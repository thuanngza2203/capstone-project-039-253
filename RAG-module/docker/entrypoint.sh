#!/bin/sh
# Lần chạy đầu (volume index còn trống) hoặc index dở dang: tự chạy `python main.py index`
# rồi mới bật API. Thư mục index lấy theo CHUNKING_STRATEGY / CHROMA_DIR trong .env như khi chạy tay.
set -e

if ! python - <<'EOF'
import sys

from config import COLLECTION_NAME, get_index_directory
from index_manifest import read_manifest

directory = get_index_directory()
try:
    ready = read_manifest(directory, COLLECTION_NAME) is not None
except RuntimeError:
    ready = False
print(f"Index {directory}: {'sẵn sàng' if ready else 'chưa có, sẽ tạo'}", flush=True)
sys.exit(0 if ready else 1)
EOF
then
    python main.py index
fi

exec "$@"
