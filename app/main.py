import hashlib
import hmac
import logging
import re
import secrets
import threading
from contextlib import asynccontextmanager
from urllib.parse import quote

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import config
from .db import get_db, init_db, now
from .mailer import render, send_email
from . import matching
from .matching import review_proposal, run_daily_matching

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("jobmatch")

scheduler = BackgroundScheduler(timezone=config.TIMEZONE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    if config.SCHEDULER_ENABLED:
        scheduler.add_job(run_daily_matching, CronTrigger(hour=config.MATCH_HOUR, minute=config.MATCH_MINUTE),
                          id="daily_matching", replace_existing=True, max_instances=1, coalesce=True)
        scheduler.start()
        log.info("Daily matching scheduled at %02d:%02d %s", config.MATCH_HOUR, config.MATCH_MINUTE, config.TIMEZONE)
    yield
    if scheduler.running:
        scheduler.shutdown(wait=False)


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=config.BASE_DIR / "app" / "static"), name="static")
templates = Jinja2Templates(directory=config.BASE_DIR / "app" / "templates")
templates.env.globals.update(FIELDS=config.FIELDS, JOB_TYPES=config.JOB_TYPES, WORKPLACES=config.WORKPLACES,
                             MAIL_ENABLED=config.MAIL_ENABLED)


def page(request: Request, name: str, code: int = 200, **ctx) -> HTMLResponse:
    return templates.TemplateResponse(request, name, ctx, status_code=code)


# ---------------- Validation helpers ----------------

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PHONE_RE = re.compile(r"^[0-9+\-\s()]{7,20}$")
MAX_TEXT = 3000


def _clean(form, key: str, limit: int = MAX_TEXT) -> str:
    return str(form.get(key, "")).strip()[:limit]


def _int_or_none(value: str):
    value = (value or "").strip()
    return int(value) if value.isdigit() else None


def _validate_contact(v: dict, errors: dict) -> None:
    if not v["name"]:
        errors["name"] = "יש למלא שם"
    if not PHONE_RE.match(v["phone"]):
        errors["phone"] = "מספר טלפון לא תקין"
    if not EMAIL_RE.match(v["email"]):
        errors["email"] = "כתובת אימייל לא תקינה"
    if not v.get("consent"):
        errors["consent"] = "יש לאשר את תנאי השימוש במידע"


def _send_welcome(email: str, name: str, kind: str, token: str) -> None:
    link = f"{config.BASE_URL}/manage/{kind}/{token}"
    subject = "המשרה נקלטה במאגר המעסיקים" if kind == "employer" else "הפרטים נקלטו במאגר מחפשי העבודה"
    send_email(email, subject, render("welcome.html", name=name, kind=kind, link=link))


# ---------------- Public pages ----------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return page(request, "index.html")


@app.get("/employer", response_class=HTMLResponse)
def employer_form(request: Request):
    return page(request, "employer_form.html", v={}, errors={})


@app.post("/employer", response_class=HTMLResponse)
async def employer_submit(request: Request, background: BackgroundTasks):
    form = await request.form()
    v = {k: _clean(form, k, 200) for k in ("name", "phone", "email", "field", "field_other",
                                            "job_type", "workplace", "workplace_other", "age_min", "age_max")}
    v.update({k: _clean(form, k) for k in ("description", "education", "experience", "extra")})
    v["consent"] = form.get("consent")
    v["email"] = v["email"].lower()
    errors: dict = {}
    _validate_contact(v, errors)
    if v["field"] not in config.FIELDS:
        errors["field"] = "יש לבחור תחום"
    if v["job_type"] not in config.JOB_TYPES:
        errors["job_type"] = "יש לבחור סוג משרה"
    if v["workplace"] not in config.WORKPLACES:
        errors["workplace"] = "יש לבחור מקום עבודה"
    age_min, age_max = _int_or_none(v["age_min"]), _int_or_none(v["age_max"])
    if (v["age_min"] and age_min is None) or (v["age_max"] and age_max is None):
        errors["age"] = "גיל חייב להיות מספר"
    elif age_min is not None and age_max is not None and age_min > age_max:
        errors["age"] = "גיל המינימום גדול מגיל המקסימום"
    if not v["description"]:
        errors["description"] = "יש לתאר את המשרה"
    if errors:
        return page(request, "employer_form.html", code=422, v=v, errors=errors)

    token = secrets.token_urlsafe(24)
    with get_db() as conn:
        conn.execute(
            """INSERT INTO employers (token, name, phone, email, field, field_other, job_type, workplace,
               workplace_other, age_min, age_max, description, education, experience, extra, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (token, v["name"], v["phone"], v["email"], v["field"],
             v["field_other"] if v["field"] == "אחר" else "", v["job_type"], v["workplace"],
             v["workplace_other"] if v["workplace"] == "אחר" else "", age_min, age_max,
             v["description"], v["education"], v["experience"], v["extra"], now()),
        )
    # Sent after the response, so the user does not wait for Google.
    background.add_task(_send_welcome, v["email"], v["name"], "employer", token)
    return RedirectResponse(f"/manage/employer/{token}?new=1", status_code=303)


@app.get("/seeker", response_class=HTMLResponse)
def seeker_form(request: Request):
    return page(request, "seeker_form.html", v={}, errors={})


@app.post("/seeker", response_class=HTMLResponse)
async def seeker_submit(request: Request, background: BackgroundTasks):
    form = await request.form()
    v = {k: _clean(form, k, 200) for k in ("name", "phone", "email", "seeking_field", "age")}
    v.update({k: _clean(form, k) for k in ("experience", "advantages", "extra")})
    v["consent"] = form.get("consent")
    v["email"] = v["email"].lower()
    errors: dict = {}
    _validate_contact(v, errors)
    age = _int_or_none(v["age"])
    if age is None or not 14 <= age <= 99:
        errors["age"] = "יש להזין גיל תקין"
    if not v["seeking_field"]:
        errors["seeking_field"] = "יש לציין תחום מבוקש"
    if errors:
        return page(request, "seeker_form.html", code=422, v=v, errors=errors)

    token = secrets.token_urlsafe(24)
    with get_db() as conn:
        conn.execute(
            """INSERT INTO seekers (token, name, phone, email, seeking_field, age, experience,
               advantages, extra, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (token, v["name"], v["phone"], v["email"], v["seeking_field"], age,
             v["experience"], v["advantages"], v["extra"], now()),
        )
    background.add_task(_send_welcome, v["email"], v["name"], "seeker", token)
    return RedirectResponse(f"/manage/seeker/{token}?new=1", status_code=303)


# ---------------- Personal page + "remove me" ----------------

TABLES = {"employer": "employers", "seeker": "seekers"}


def _get_by_token(kind: str, token: str):
    if kind not in TABLES:
        raise HTTPException(404)
    with get_db() as conn:
        row = conn.execute(f"SELECT * FROM {TABLES[kind]} WHERE token = ?", (token,)).fetchone()
    if row is None:
        raise HTTPException(404)
    return row


@app.get("/manage/{kind}/{token}", response_class=HTMLResponse)
def manage(request: Request, kind: str, token: str, new: int = 0):
    return page(request, "manage.html", kind=kind, item=_get_by_token(kind, token), new=new)


@app.post("/manage/{kind}/{token}/delete", response_class=HTMLResponse)
def delete_me(request: Request, kind: str, token: str):
    row = _get_by_token(kind, token)
    with get_db() as conn:
        conn.execute(f"DELETE FROM {TABLES[kind]} WHERE id = ?", (row["id"],))
    log.info("%s %s removed itself from the database", kind, row["id"])
    return page(request, "removed.html", kind=kind)



# ---------------- Admin (human representative) ----------------

security = HTTPBasic()


def admin_user(creds: HTTPBasicCredentials = Depends(security)) -> str:
    ok = bool(config.ADMIN_PASSWORD) and secrets.compare_digest(
        creds.username.encode(), config.ADMIN_USER.encode()
    ) and secrets.compare_digest(creds.password.encode(), config.ADMIN_PASSWORD.encode())
    if not ok:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Unauthorized",
                            headers={"WWW-Authenticate": "Basic"})
    return creds.username


def csrf_token(user: str) -> str:
    return hmac.new(config.SECRET_KEY.encode(), f"csrf:{user}".encode(), hashlib.sha256).hexdigest()


def _admin_redirect(msg: str) -> RedirectResponse:
    return RedirectResponse("/admin?msg=" + quote(msg), status_code=303)


def check_csrf(form, user: str) -> None:
    if not hmac.compare_digest(str(form.get("csrf", "")), csrf_token(user)):
        raise HTTPException(403, "Bad CSRF token")


def _admin_version(conn) -> str:
    """Short fingerprint of everything the admin home page shows; it changes whenever the page would."""
    parts = [
        conn.execute("SELECT group_concat(id || ':' || status) FROM proposals").fetchone()[0],
        conn.execute("SELECT group_concat(id || ':' || status || ':' || ifnull(finished_at, '')) "
                     "FROM (SELECT * FROM match_runs ORDER BY id DESC LIMIT 10)").fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM employers").fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM seekers").fetchone()[0],
    ]
    return hashlib.sha256(repr(parts).encode()).hexdigest()[:16]


@app.get("/admin/state")
def admin_state(user: str = Depends(admin_user)):
    """Polled by the admin page so it can refresh itself when something changes."""
    with get_db() as conn:
        return JSONResponse({"version": _admin_version(conn), "running": matching._run_lock.locked()},
                            headers={"Cache-Control": "no-store"})


@app.get("/admin", response_class=HTMLResponse)
def admin_home(request: Request, user: str = Depends(admin_user), msg: str = ""):
    with get_db() as conn:
        pending = conn.execute(
            """SELECT p.id, p.created_at, e.name AS employer, e.field, e.job_type,
                      (SELECT COUNT(*) FROM proposal_items WHERE proposal_id = p.id) AS n
               FROM proposals p JOIN employers e ON e.id = p.employer_id
               WHERE p.status = 'pending' ORDER BY p.created_at"""
        ).fetchall()
        history = conn.execute(
            """SELECT p.id, p.status, p.reviewed_at, p.reviewed_by, e.name AS employer,
                      (SELECT COUNT(*) FROM proposal_items WHERE proposal_id = p.id AND approved = 1) AS sent,
                      (SELECT COUNT(*) FROM proposal_items WHERE proposal_id = p.id) AS n
               FROM proposals p JOIN employers e ON e.id = p.employer_id
               WHERE p.status != 'pending' ORDER BY p.reviewed_at DESC LIMIT 30"""
        ).fetchall()
        runs = conn.execute("SELECT * FROM match_runs ORDER BY id DESC LIMIT 10").fetchall()
        counts = {
            "employers": conn.execute("SELECT COUNT(*) FROM employers").fetchone()[0],
            "seekers": conn.execute("SELECT COUNT(*) FROM seekers").fetchone()[0],
        }
        version = _admin_version(conn)
    return page(request, "admin/index.html", pending=pending, history=history, runs=runs,
                counts=counts, csrf=csrf_token(user), msg=msg, config=config,
                version=version, running=matching._run_lock.locked())


@app.post("/admin/run")
async def admin_run(request: Request, user: str = Depends(admin_user)):
    check_csrf(await request.form(), user)
    if matching._run_lock.locked():
        return _admin_redirect("הרצת התאמה קודמת עדיין פעילה. הדף יתעדכן אוטומטית כשתסתיים.")
    threading.Thread(target=run_daily_matching, kwargs={"trigger": f"manual:{user}"}, daemon=True).start()
    return _admin_redirect("ההתאמה הופעלה ברקע. הדף יתעדכן אוטומטית כשיהיו תוצאות.")


@app.get("/admin/proposals/{pid}", response_class=HTMLResponse)
def admin_proposal(request: Request, pid: int, user: str = Depends(admin_user)):
    with get_db() as conn:
        p = conn.execute("SELECT * FROM proposals WHERE id = ?", (pid,)).fetchone()
        if p is None:
            raise HTTPException(404)
        employer = conn.execute("SELECT * FROM employers WHERE id = ?", (p["employer_id"],)).fetchone()
        items = conn.execute(
            """SELECT pi.*, s.name, s.phone, s.email, s.age, s.seeking_field, s.experience,
                      s.advantages, s.extra
               FROM proposal_items pi JOIN seekers s ON s.id = pi.seeker_id
               WHERE pi.proposal_id = ? ORDER BY pi.score DESC""",
            (pid,),
        ).fetchall()
    return page(request, "admin/proposal.html", p=p, employer=employer, items=items, csrf=csrf_token(user))


@app.post("/admin/proposals/{pid}")
async def admin_proposal_decide(request: Request, pid: int, user: str = Depends(admin_user)):
    form = await request.form()
    check_csrf(form, user)
    action = form.get("action")
    if action not in ("send", "send_all", "reject"):
        raise HTTPException(400)
    if action == "send_all":
        with get_db() as conn:
            ids = {r[0] for r in conn.execute("SELECT seeker_id FROM proposal_items WHERE proposal_id=?", (pid,))}
    else:
        ids = {int(x) for x in form.getlist("seeker_id") if str(x).isdigit()}
    result = review_proposal(pid, ids, reviewer=user, action="reject" if action == "reject" else "send")
    messages = {
        "sent": (f"נשלח מייל למעסיק עם {result.get('sent')} מועמדים." if result.get("emailed") else
                 f"ההצעה אושרה עם {result.get('sent')} מועמדים. לא נשלח מייל - שליחת מיילים לא מוגדרת."),
        "rejected": "ההצעה נסגרה ללא אישור מועמדים.",
        "not_pending": "ההצעה כבר טופלה.",
        "email_failed": "שליחת המייל למעסיק נכשלה - ההצעה נשארה ממתינה. בדקו את MAIL_WEBHOOK_URL / MAIL_WEBHOOK_SECRET ואת הסקריפט ב-Google.",
    }
    return _admin_redirect(messages[result["result"]])


@app.get("/admin/data", response_class=HTMLResponse)
def admin_data(request: Request, user: str = Depends(admin_user)):
    with get_db() as conn:
        employers = conn.execute("SELECT * FROM employers ORDER BY id DESC").fetchall()
        seekers = conn.execute("SELECT * FROM seekers ORDER BY id DESC").fetchall()
    return page(request, "admin/data.html", employers=employers, seekers=seekers)
