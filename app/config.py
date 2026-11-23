import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "jobmatch.db"))

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000").rstrip("/")
SECRET_KEY = os.getenv("SECRET_KEY") or secrets.token_hex(32)
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")

# Email goes out through a Google Apps Script web app (see README). Both values are required.
MAIL_WEBHOOK_URL = os.getenv("MAIL_WEBHOOK_URL", "").strip()
MAIL_WEBHOOK_SECRET = os.getenv("MAIL_WEBHOOK_SECRET", "").strip()
MAIL_ENABLED = bool(MAIL_WEBHOOK_URL and MAIL_WEBHOOK_SECRET)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
# Seconds to wait after each Gemini call (free tier allows only a few requests per minute).
GEMINI_CALL_DELAY = float(os.getenv("GEMINI_CALL_DELAY", "6"))
MATCH_MIN_SCORE = int(os.getenv("MATCH_MIN_SCORE", "60"))
MATCH_BATCH_SIZE = int(os.getenv("MATCH_BATCH_SIZE", "40"))

SCHEDULER_ENABLED = _bool("SCHEDULER_ENABLED", True)
MATCH_HOUR = int(os.getenv("MATCH_HOUR", "8"))
MATCH_MINUTE = int(os.getenv("MATCH_MINUTE", "0"))
TIMEZONE = os.getenv("TIMEZONE", "Asia/Jerusalem")

LANGSMITH_ENABLED = _bool("LANGSMITH_TRACING", False) and bool(os.getenv("LANGSMITH_API_KEY"))

# ---- Form options (from the spec) ----
FIELDS = [
    "הוראה וחינוך רגיל",
    "משרד",
    "כוח אדם",
    "פיננסי",
    "עיצוב",
    "ניהול פרויקט",
    "תרבות",
    "חינוך מיוחד",
    "תחומי הטיפול",
    "אחר",
]
JOB_TYPES = ["עבודה קבועה", "פרויקט חד פעמי", "מילוי מקום"]
WORKPLACES = ["מוסד", "משרד", "עבודה יחידנית", "אחר"]
