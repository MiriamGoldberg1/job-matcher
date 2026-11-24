"""LangGraph workflow that matches one employer with job seekers, with a human approval step.

    START -> load_candidates --(nothing to do)--> END
                   |
                   v
            score_batch <--+   loop: one Gemini call per batch of seekers
                   |-------+
                   v
       (matches?) --no--> no_match_email -> END
                   | yes
                   v
            create_proposal -> human_review   <-- interrupt(): waits for the representative
                                   |    ^
                      send --------+    +---- email failed: back to the representative (loop)
                        v               |
                send_to_employer -------+
                        |
                        v
            close_proposal -> END   (also reached directly on "reject")

The graph state is saved in data/graph.db (SqliteSaver), so a proposal can wait for days and
the graph resumes from the same point when the representative decides (Command(resume=...)).
"""
import logging
import sqlite3
import uuid
from typing import Literal, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from . import config
from .ai import ask_ai, job_text, seeker_text
from .db import get_db, now
from .mailer import render, send_email

log = logging.getLogger(__name__)


class MatchState(TypedDict, total=False):
    employer_id: int
    run_id: int | None
    trace_id: str | None          # LangSmith run of match_employer, for the human feedback
    thread_id: str
    candidate_ids: list[int]
    batch_start: int              # index of the next batch to score
    found: list[dict]             # [{seeker_id, score, reason}]
    proposal_id: int
    decision: dict                # {"action": "send" | "reject", "seeker_ids": [...], "reviewer": "..."}
    email_failed: bool
    emailed: bool
    result: str                   # deleted | awaiting_review | no_match | proposal_pending | sent | rejected


# ---------------- Nodes ----------------

def load_candidates(state: MatchState) -> dict:
    """Seekers in the age range that were not already reviewed for this employer."""
    with get_db() as conn:
        employer = conn.execute("SELECT * FROM employers WHERE id = ?", (state["employer_id"],)).fetchone()
        if employer is None:
            return {"result": "deleted"}
        if conn.execute("SELECT 1 FROM proposals WHERE employer_id = ? AND status = 'pending'",
                        (employer["id"],)).fetchone():
            return {"result": "awaiting_review"}
        lo, hi = employer["age_min"], employer["age_max"]
        rows = conn.execute(
            """SELECT id FROM seekers s
               WHERE (? IS NULL OR s.age >= ?) AND (? IS NULL OR s.age <= ?)
                 AND s.id NOT IN (
                     SELECT pi.seeker_id FROM proposal_items pi JOIN proposals p ON p.id = pi.proposal_id
                     WHERE p.employer_id = ? AND p.status != 'pending')
               ORDER BY s.id""",
            (lo, lo, hi, hi, employer["id"]),
        ).fetchall()
    return {"candidate_ids": [r["id"] for r in rows], "batch_start": 0, "found": []}


def score_batch(state: MatchState) -> dict:
    """One Gemini call for the next batch of candidates; keeps the best score per seeker."""
    size = max(1, config.MATCH_BATCH_SIZE)
    start = state["batch_start"]
    ids = state["candidate_ids"][start:start + size]
    with get_db() as conn:
        employer = conn.execute("SELECT * FROM employers WHERE id = ?", (state["employer_id"],)).fetchone()
        marks = ",".join("?" * len(ids))
        seekers = conn.execute(f"SELECT * FROM seekers WHERE id IN ({marks}) ORDER BY id", ids).fetchall()
    found = {m["seeker_id"]: m for m in state.get("found", [])}
    valid = {s["id"] for s in seekers}
    if seekers:
        for m in ask_ai(job_text(employer), "\n".join(seeker_text(s) for s in seekers), config.MATCH_MIN_SCORE):
            sid, score = m.get("seeker_id"), int(m.get("score", 0))
            if sid in valid and score >= config.MATCH_MIN_SCORE and score > found.get(sid, {}).get("score", -1):
                found[sid] = {"seeker_id": sid, "score": min(score, 100), "reason": m["reason"].strip()}
    return {"batch_start": start + size, "found": list(found.values())}


def no_match_email(state: MatchState) -> dict:
    """The "no suitable candidates yet" email goes to each employer only once."""
    with get_db() as conn:
        employer = conn.execute("SELECT * FROM employers WHERE id = ?", (state["employer_id"],)).fetchone()
    if employer and not employer["no_match_sent_at"] and send_email(
            employer["email"], "עדכון ממערכת מחברים", render("no_match.html", employer=employer)):
        with get_db() as conn:
            conn.execute("UPDATE employers SET no_match_sent_at = ? WHERE id = ?", (now(), employer["id"]))
    return {"result": "no_match"}


def create_proposal(state: MatchState) -> dict:
    with get_db() as conn:
        pid = conn.execute(
            """INSERT INTO proposals (run_id, employer_id, status, langsmith_run_id, graph_thread_id, created_at)
               VALUES (?,?,?,?,?,?)""",
            (state.get("run_id"), state["employer_id"], "pending", state.get("trace_id"),
             state["thread_id"], now()),
        ).lastrowid
        conn.executemany(
            "INSERT INTO proposal_items (proposal_id, seeker_id, score, reason) VALUES (?,?,?,?)",
            [(pid, m["seeker_id"], m["score"], m["reason"])
             for m in sorted(state["found"], key=lambda x: -x["score"])],
        )
    return {"proposal_id": pid, "result": "proposal_pending"}


def human_review(state: MatchState) -> dict:
    """Human in the loop: the graph stops here until the representative decides in the admin UI.

    Nothing may run before interrupt(): LangGraph re-runs this node from the top when it resumes.
    """
    decision = interrupt({
        "proposal_id": state["proposal_id"],
        "candidates": len(state["found"]),
        "email_failed_before": state.get("email_failed", False),
    })
    return {"decision": decision, "email_failed": False}


def send_to_employer(state: MatchState) -> dict:
    if not config.MAIL_ENABLED:
        return {"emailed": False}
    items = _approved_items(state["proposal_id"], set(state["decision"]["seeker_ids"]))
    with get_db() as conn:
        employer = conn.execute("SELECT * FROM employers WHERE id = ?", (state["employer_id"],)).fetchone()
    ok = bool(items) and employer is not None and send_email(
        employer["email"], "נמצאו מועמדים מתאימים למשרה שלכם",
        render("candidates.html", employer=employer, candidates=items))
    return {"emailed": ok, "email_failed": not ok}


def close_proposal(state: MatchState) -> dict:
    decision = state["decision"]
    approved = []
    if decision["action"] == "send":
        approved = _approved_items(state["proposal_id"], set(decision.get("seeker_ids", [])))
    status = "sent" if approved else "rejected"
    approved_ids = {i["id"] for i in approved}
    with get_db() as conn:
        for item_id, in conn.execute("SELECT id FROM proposal_items WHERE proposal_id = ?",
                                     (state["proposal_id"],)).fetchall():
            conn.execute("UPDATE proposal_items SET approved = ? WHERE id = ?",
                         (1 if item_id in approved_ids else 0, item_id))
        conn.execute("UPDATE proposals SET status=?, reviewed_at=?, reviewed_by=? WHERE id=?",
                     (status, now(), decision.get("reviewer", ""), state["proposal_id"]))
    return {"result": status}


def _approved_items(proposal_id: int, seeker_ids: set[int]) -> list:
    with get_db() as conn:
        items = conn.execute(
            """SELECT pi.*, s.name, s.phone, s.email, s.age, s.seeking_field, s.experience,
                      s.advantages, s.extra
               FROM proposal_items pi JOIN seekers s ON s.id = pi.seeker_id
               WHERE pi.proposal_id = ? ORDER BY pi.score DESC""",
            (proposal_id,),
        ).fetchall()
    return [i for i in items if i["seeker_id"] in seeker_ids]


# ---------------- Edges (the decisions) ----------------

def after_load(state: MatchState) -> Literal["score_batch", "no_match_email", "__end__"]:
    if state.get("result"):                       # deleted / still waiting for an earlier review
        return END
    return "score_batch" if state["candidate_ids"] else "no_match_email"


def after_score(state: MatchState) -> Literal["score_batch", "create_proposal", "no_match_email"]:
    if state["batch_start"] < len(state["candidate_ids"]):
        return "score_batch"                      # loop: more batches to score
    return "create_proposal" if state["found"] else "no_match_email"


def after_review(state: MatchState) -> Literal["send_to_employer", "close_proposal"]:
    d = state["decision"]
    return "send_to_employer" if d["action"] == "send" and d.get("seeker_ids") else "close_proposal"


def after_send(state: MatchState) -> Literal["human_review", "close_proposal"]:
    return "human_review" if state.get("email_failed") else "close_proposal"


def build_graph(checkpointer=None):
    g = StateGraph(MatchState)
    for node in (load_candidates, score_batch, no_match_email, create_proposal,
                 human_review, send_to_employer, close_proposal):
        g.add_node(node.__name__, node)
    g.add_edge(START, "load_candidates")
    g.add_conditional_edges("load_candidates", after_load)
    g.add_conditional_edges("score_batch", after_score)
    g.add_edge("no_match_email", END)
    g.add_edge("create_proposal", "human_review")
    g.add_conditional_edges("human_review", after_review)
    g.add_conditional_edges("send_to_employer", after_send)
    g.add_edge("close_proposal", END)
    return g.compile(checkpointer=checkpointer)


_graph = None


def graph():
    global _graph
    if _graph is None:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(config.DATA_DIR / "graph.db", check_same_thread=False)
        _graph = build_graph(SqliteSaver(conn))
    return _graph


def _cfg(thread_id: str) -> dict:
    # Each loop pass is a step; allow many batches before LangGraph's recursion guard stops the run.
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 1000}


def start(employer_id: int, run_id: int | None, trace_id: str | None) -> dict:
    """Runs the graph until it ends or stops at human_review."""
    thread_id = f"employer-{employer_id}-{uuid.uuid4().hex[:12]}"
    return graph().invoke({"employer_id": employer_id, "run_id": run_id, "trace_id": trace_id,
                           "thread_id": thread_id}, _cfg(thread_id))


def resume(thread_id: str, decision: dict) -> dict:
    """Continues a graph that waits at human_review with the representative's decision."""
    g = graph()
    g.invoke(Command(resume=decision), _cfg(thread_id))
    snapshot = g.get_state(_cfg(thread_id))
    state = dict(snapshot.values)
    state["waiting"] = "human_review" in snapshot.next
    return state


def is_waiting(thread_id: str) -> bool:
    return "human_review" in graph().get_state(_cfg(thread_id)).next


def adopt_proposal(p) -> str:
    """Gives a pending proposal that has no graph thread (created before the graph existed) a thread
    that waits at human_review, as if create_proposal had just run. Returns the new thread id."""
    with get_db() as conn:
        items = conn.execute("SELECT seeker_id, score, reason FROM proposal_items WHERE proposal_id = ?",
                             (p["id"],)).fetchall()
    thread_id = f"employer-{p['employer_id']}-{uuid.uuid4().hex[:12]}"
    g, cfg = graph(), _cfg(thread_id)
    g.update_state(cfg, {"employer_id": p["employer_id"], "run_id": p["run_id"], "thread_id": thread_id,
                         "trace_id": p["langsmith_run_id"], "proposal_id": p["id"],
                         "found": [dict(i) for i in items], "result": "proposal_pending"},
                   as_node="create_proposal")
    g.invoke(None, cfg)  # runs up to the interrupt in human_review
    with get_db() as conn:
        conn.execute("UPDATE proposals SET graph_thread_id = ? WHERE id = ?", (thread_id, p["id"]))
    return thread_id
