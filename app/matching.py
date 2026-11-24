"""Daily AI matching between employers and job seekers, with a human approval step.

The per-employer flow is a LangGraph graph (see graph.py):
  candidates -> Gemini scores them in batches (loop) -> no match: one-time email to the employer
                                                     -> matches: a proposal waits at interrupt()
  -> the representative decides in the admin UI -> the graph resumes and emails the employer.
"""
import logging
import threading

from langsmith import Client as LangSmithClient
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from . import config, graph
from .ai import QuotaExhausted
from .db import get_db, now

log = logging.getLogger(__name__)

_run_lock = threading.Lock()


@traceable(name="match_employer")
def match_employer(employer_id: int, run_id: int | None) -> dict:
    run_tree = get_current_run_tree()
    state = graph.start(employer_id, run_id, str(run_tree.id) if run_tree else None)
    return {"employer_id": employer_id, "result": state["result"],
            "proposal_id": state.get("proposal_id"), "matches": len(state.get("found", [])),
            "candidates_checked": len(state.get("candidate_ids", []))}


@traceable(name="daily_matching_run")
def run_daily_matching(trigger: str = "scheduler") -> dict:
    if not _run_lock.acquire(blocking=False):
        return {"status": "already_running"}
    try:
        with get_db() as conn:
            run_id = conn.execute(
                "INSERT INTO match_runs (trigger, started_at, status) VALUES (?,?,?)",
                (trigger, now(), "running"),
            ).lastrowid
            employer_ids = [r["id"] for r in conn.execute("SELECT id FROM employers ORDER BY id")]

        counts = {"proposal_pending": 0, "no_match": 0, "awaiting_review": 0, "deleted": 0, "error": 0}
        quota_hit = False
        for eid in employer_ids:
            try:
                counts[match_employer(eid, run_id)["result"]] += 1
            except QuotaExhausted:
                log.error("Daily Gemini quota for %s is used up - stopping the run", config.GEMINI_MODEL)
                quota_hit = True
                break
            except Exception:  # noqa: BLE001 - one employer must not stop the run
                log.exception("Matching failed for employer %s", eid)
                counts["error"] += 1

        summary = (f"מעסיקים: {len(employer_ids)} | הצעות חדשות לאישור: {counts['proposal_pending']} | "
                   f"ללא התאמה: {counts['no_match']} | ממתינים לבדיקה קודמת: {counts['awaiting_review']} | "
                   f"שגיאות: {counts['error']}")
        if quota_hit:
            summary = (f"המכסה היומית של Gemini ({config.GEMINI_MODEL}) נגמרה - ההרצה נעצרה. "
                       f"אפשר להחליף את GEMINI_MODEL ב-.env או לחכות לאיפוס המכסה. " + summary)
        with get_db() as conn:
            conn.execute("UPDATE match_runs SET finished_at=?, status=?, summary=? WHERE id=?",
                         (now(), "quota" if quota_hit else "error" if counts["error"] else "done", summary, run_id))
        return {"status": "quota" if quota_hit else "done", "run_id": run_id, **counts}
    finally:
        _run_lock.release()


# ---------------- Human in the loop ----------------

@traceable(name="human_review_decision")
def review_proposal(proposal_id: int, approved_seeker_ids: set[int], reviewer: str, action: str) -> dict:
    """Resumes the proposal's graph (waiting at interrupt) with the representative's decision.

    action = 'send' (email the selected seekers to the employer) or 'reject'.
    """
    with get_db() as conn:
        p = conn.execute("SELECT * FROM proposals WHERE id = ?", (proposal_id,)).fetchone()
        if p is None or p["status"] != "pending":
            return {"result": "not_pending"}
        total = conn.execute("SELECT COUNT(*) FROM proposal_items WHERE proposal_id = ?", (proposal_id,)).fetchone()[0]

    thread_id = p["graph_thread_id"]
    if not thread_id or not graph.is_waiting(thread_id):
        thread_id = graph.adopt_proposal(p)  # proposal created before the graph existed

    decision = {"action": action, "seeker_ids": sorted(approved_seeker_ids), "reviewer": reviewer}
    state = graph.resume(thread_id, decision)
    if state["waiting"]:
        return {"result": "email_failed"}  # the graph went back to human_review; proposal stays pending

    with get_db() as conn:
        sent = conn.execute("SELECT COUNT(*) FROM proposal_items WHERE proposal_id = ? AND approved = 1",
                            (proposal_id,)).fetchone()[0]
    _send_feedback(p["langsmith_run_id"], sent, total, reviewer)
    return {"result": state["result"], "sent": sent, "total": total, "emailed": state.get("emailed", False)}


def _send_feedback(run_id: str | None, approved: int, total: int, reviewer: str) -> None:
    """Attach the human decision to the AI's LangSmith trace (share of AI suggestions approved)."""
    if not (config.LANGSMITH_ENABLED and run_id and total):
        return
    try:
        LangSmithClient().create_feedback(
            run_id=run_id,
            key="human_approval_rate",
            score=approved / total,
            comment=f"{reviewer} approved {approved} of {total} suggested candidates",
        )
    except Exception:  # noqa: BLE001 - tracing must never break the review
        log.warning("Could not send LangSmith feedback", exc_info=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from .db import init_db
    init_db()
    print(run_daily_matching(trigger="cli"))
