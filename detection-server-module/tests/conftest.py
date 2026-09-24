"""Test offline: không gọi Groq, MongoDB, RAG server hay tải checkpoint."""

import os
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

# app.config đọc các biến này lúc import; không cần .env thật để chạy test.
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("MONGO_URI", "mongodb://127.0.0.1:1")
os.environ.setdefault("SAVE_SEGMENTATION_PREVIEW", "false")
