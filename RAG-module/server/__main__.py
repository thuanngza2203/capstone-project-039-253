"""Chạy RAG API hoặc xuất OpenAPI schema.

    python -m server                             # theo RAG_API_HOST / RAG_API_PORT
    python -m server --host 0.0.0.0 --port 8010  # public; không có RAG_API_KEY thì chỉ cảnh báo
    python -m server --export-openapi openapi.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config import LOOPBACK_HOSTS, get_api_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAG bệnh cây API (FastAPI + Swagger).")
    parser.add_argument("--host", help="Ghi đè RAG_API_HOST.")
    parser.add_argument("--port", type=int, help="Ghi đè RAG_API_PORT.")
    parser.add_argument("--export-openapi", type=Path, metavar="FILE",
                        help="Ghi OpenAPI schema ra file rồi thoát; không nạp model.")
    args = parser.parse_args(argv)

    from server.app import create_app

    try:
        if args.export_openapi:
            # Schema không phụ thuộc key hay index; không mở Chroma, không nạp model.
            schema = create_app(api_key="").openapi()
            args.export_openapi.parent.mkdir(parents=True, exist_ok=True)
            args.export_openapi.write_text(
                json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
            )
            print(f"Đã ghi OpenAPI schema: {args.export_openapi.resolve()}")
            return 0

        settings = get_api_settings()
        host = args.host or settings.host
        port = args.port or settings.port
        if not 1 <= port <= 65535:
            raise ValueError("Cổng phải nằm trong 1–65535.")
    except ValueError as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1

    if not settings.api_key and host not in LOOPBACK_HOSTS:
        # API public là lựa chọn đã chốt (plan kiến trúc 24/09); chỉ nhắc để không bật nhầm.
        print(f"Cảnh báo: nghe trên {host} và RAG_API_KEY trống: ai tới được cổng này cũng gọi "
              "được mọi API, kể cả /v1/answer (dùng GPU của LLM).", file=sys.stderr, flush=True)

    import uvicorn

    print(f"Swagger: http://{'127.0.0.1' if host in LOOPBACK_HOSTS else host}:{port}/docs", flush=True)
    # Một process duy nhất: embedding và index nằm trong bộ nhớ của process này.
    uvicorn.run(create_app(api_key=settings.api_key), host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
