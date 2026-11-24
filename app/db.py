import contextlib
import sqlite3
from datetime import datetime, timezone

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS employers (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    token           TEXT NOT NULL UNIQUE,
    name            TEXT NOT NULL,
    phone           TEXT NOT NULL,
    email           TEXT NOT NULL,
    field           TEXT NOT NULL,
    field_other     TEXT DEFAULT '',
    job_type        TEXT NOT NULL,
    workplace       TEXT NOT NULL,
    workplace_other TEXT DEFAULT '',
    age_min         INTEGER,
    age_max         INTEGER,
    description     TEXT DEFAULT '',
    education       TEXT DEFAULT '',
    experience      TEXT DEFAULT '',
    extra           TEXT DEFAULT '',
    no_match_sent_at TEXT,          -- when the one-time "no match yet" email went out
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS seekers (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    token         TEXT NOT NULL UNIQUE,
    name          TEXT NOT NULL,
    phone         TEXT NOT NULL,
    email         TEXT NOT NULL,
    seeking_field TEXT DEFAULT '',
    age           INTEGER,
    experience    TEXT DEFAULT '',
    advantages    TEXT DEFAULT '',
    extra         TEXT DEFAULT '',
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS match_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger     TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    status      TEXT NOT NULL,
    summary     TEXT DEFAULT ''
);

-- One proposal = the AI's list of candidates for one employer, waiting for a human.
CREATE TABLE IF NOT EXISTS proposals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER REFERENCES match_runs(id) ON DELETE SET NULL,
    employer_id     INTEGER NOT NULL REFERENCES employers(id) ON DELETE CASCADE,
    status          TEXT NOT NULL DEFAULT 'pending',   -- pending / sent / rejected
    langsmith_run_id TEXT,
    graph_thread_id TEXT,          -- LangGraph thread that waits at human_review for this proposal
    created_at      TEXT NOT NULL,
    reviewed_at     TEXT,
    reviewed_by     TEXT
);

CREATE TABLE IF NOT EXISTS proposal_items (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    proposal_id INTEGER NOT NULL REFERENCES proposals(id) ON DELETE CASCADE,
    seeker_id   INTEGER NOT NULL REFERENCES seekers(id) ON DELETE CASCADE,
    score       INTEGER NOT NULL,
    reason      TEXT NOT NULL,
    approved    INTEGER            -- NULL = not reviewed, 1 = sent, 0 = not sent
);

CREATE INDEX IF NOT EXISTS idx_proposals_employer ON proposals(employer_id, status);
CREATE INDEX IF NOT EXISTS idx_items_proposal ON proposal_items(proposal_id);
CREATE INDEX IF NOT EXISTS idx_employers_email ON employers(email);
CREATE INDEX IF NOT EXISTS idx_seekers_email ON seekers(email);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextlib.contextmanager
def get_db():
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with get_db() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(SCHEMA)
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(employers)")}
        if "no_match_sent_at" not in cols:  # databases created before this column existed
            conn.execute("ALTER TABLE employers ADD COLUMN no_match_sent_at TEXT")
        if "graph_thread_id" not in {r["name"] for r in conn.execute("PRAGMA table_info(proposals)")}:
            conn.execute("ALTER TABLE proposals ADD COLUMN graph_thread_id TEXT")
        # A run still marked 'running' at startup was cut off when the server stopped.
        conn.execute("UPDATE match_runs SET status='interrupted', finished_at=? WHERE status='running'", (now(),))
