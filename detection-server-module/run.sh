#!/bin/sh
# Chạy app bằng .venv của module. Cổng mặc định 8005, đổi bằng PORT.
# Tạo venv lần đầu: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cd "$(dirname "$0")"
if [ ! -f rag/chroma_db/chroma.sqlite3 ]; then
    .venv/bin/python -m rag.ingest --reset
fi
exec .venv/bin/uvicorn app.api:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8005}"
