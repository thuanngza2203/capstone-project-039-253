"""Command line interface cho RAG-module-2."""

from __future__ import annotations

import argparse
import sys

from config import TOP_K
from rag import ask, build_index, retrieve


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAG LangChain tối giản cho tài liệu bệnh cây tiếng Việt."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("index", help="Đọc data/**/*.txt và build lại Chroma.")

    search_parser = commands.add_parser(
        "search", help="Kiểm tra các chunk được retrieval mà không gọi Gemini."
    )
    search_parser.add_argument("question", help="Câu hỏi cần tìm trong kho dữ liệu.")
    search_parser.add_argument("--top-k", type=int, default=TOP_K)

    ask_parser = commands.add_parser(
        "ask", help="Retrieval rồi dùng Gemini để tạo câu trả lời."
    )
    ask_parser.add_argument("question", help="Câu hỏi gửi tới hệ thống RAG.")
    ask_parser.add_argument("--top-k", type=int, default=TOP_K)
    return parser


def run(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    try:
        if args.command == "index":
            file_count, chunk_count = build_index()
            print(
                f"Đã index {file_count} tài liệu thành {chunk_count} chunk "
                "trong chroma_db/."
            )
            return 0

        if args.command == "search":
            documents = retrieve(args.question, k=args.top_k)
            if not documents:
                print("Không tìm thấy chunk liên quan.")
                return 0
            for index, document in enumerate(documents, start=1):
                source = document.metadata.get("source", "không rõ nguồn")
                print(f"\n[Nguồn {index}: {source}]")
                print(document.page_content)
            return 0

        answer, sources = ask(args.question, k=args.top_k)
        print(answer)
        if sources:
            print("\nNguồn đã truy xuất:")
            for source in sources:
                print(f"- {source}")
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports provider errors.
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(run())
