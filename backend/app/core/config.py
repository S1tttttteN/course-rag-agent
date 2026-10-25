"""Configuration independent of the working directory."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")
DATA_DIR = Path(os.getenv("COURSE_AGENT_DATA_DIR", "").strip() or str(PROJECT_ROOT / "data")).resolve()
DATABASE_PATH = DATA_DIR / "app.sqlite3"
ALLOWED_ORIGINS = ["http://127.0.0.1:5173", "http://localhost:5173"]
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "course_indexes"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
