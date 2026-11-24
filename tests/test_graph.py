"""Checks every path of the LangGraph workflow with a fake Gemini and a fake mailer (no network).

Run from the job-matcher folder:  .venv\\Scripts\\python -m tests.test_graph
"""
import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ.update(DB_PATH=os.path.join(_tmp, "t.db"), DATA_DIR=_tmp, SCHEDULER_ENABLED="false",
                  MAIL_WEBHOOK_URL="x", MAIL_WEBHOOK_SECRET="y", LANGSMITH_TRACING="false", MATCH_BATCH_SIZE="2")

from unittest import mock  # noqa: E402

from app import graph, matching  # noqa: E402
from app.db import get_db, init_db, now  # noqa: E402


def fake_ai(job, seekers_text, min_score, prompt_version=None):
    """Employer 1 matches seekers 1 and 4; employer 2 matches nobody."""
    ids = [int(x.split('"')[1]) for x in seekers_text.split("<seeker id=")[1:]]
    return [] if job == "EMPLOYER2" else [{"seeker_id": i, "score": 80, "reason": "ok"} for i in ids if i in (1, 4)]


def main():
    init_db()
    with get_db() as c:
        for i, email in enumerate(["a@x", "b@x"]):
            c.execute("""INSERT INTO employers (token,name,phone,email,field,job_type,workplace,description,created_at)
                         VALUES (?,?,?,?,?,?,?,?,?)""", (f"t{i}", f"E{i}", "050", email, "משרד", "עבודה קבועה", "משרד", "d", now()))
        for i in range(5):
            c.execute("INSERT INTO seekers (token,name,phone,email,seeking_field,age,created_at) VALUES (?,?,?,?,?,?,?)",
                      (f"s{i}", f"S{i}", "050", f"s{i}@x", "משרד", 30, now()))

    calls, sent, mail_ok = [], [], [True]

    def ai(job, st, ms, pv=None):
        calls.append(job)
        return fake_ai(job, st, ms)

    def mail(to, subject, html):
        sent.append(to)
        return mail_ok[0]

    with mock.patch.object(graph, "ask_ai", ai), mock.patch.object(graph, "send_email", mail), \
            mock.patch.object(graph, "job_text", lambda e: f"EMPLOYER{e['id']}"):
        r = matching.run_daily_matching("test")
        assert (r["proposal_pending"], r["no_match"]) == (1, 1), r
        assert len(calls) == 6, "loop: 5 seekers in batches of 2 = 3 Gemini calls per employer"
        assert sent == ["b@x"], "conditional branch: no match -> one email to employer 2"

        with get_db() as c:
            p = c.execute("SELECT * FROM proposals").fetchone()
        assert p["status"] == "pending" and graph.is_waiting(p["graph_thread_id"]), "interrupt at human_review"

        sent.clear()
        r = matching.run_daily_matching("test")
        assert r["awaiting_review"] == 1 and sent == [], "no-match email is sent only once"

        mail_ok[0] = False
        assert matching.review_proposal(p["id"], {1, 4}, "admin", "send")["result"] == "email_failed"
        assert graph.is_waiting(p["graph_thread_id"]), "failed email loops back to human_review"

        mail_ok[0] = True
        r = matching.review_proposal(p["id"], {4}, "admin", "send")
        assert (r["result"], r["sent"], r["emailed"]) == ("sent", 1, True), r
        with get_db() as c:
            assert {tuple(x) for x in c.execute("SELECT seeker_id, approved FROM proposal_items")} == {(1, 0), (4, 1)}
        assert matching.review_proposal(p["id"], {4}, "admin", "send")["result"] == "not_pending"

        with get_db() as c:  # a proposal created before the graph existed has no thread
            pid = c.execute("INSERT INTO proposals (employer_id,status,created_at) VALUES (2,'pending',?)",
                            (now(),)).lastrowid
            c.execute("INSERT INTO proposal_items (proposal_id,seeker_id,score,reason) VALUES (?,2,70,'r')", (pid,))
        assert matching.review_proposal(pid, set(), "admin", "reject")["result"] == "rejected"
    print("all graph paths OK")


if __name__ == "__main__":
    main()
