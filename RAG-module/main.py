"""CLI một câu hỏi hoặc phiên hỏi liên tục giữ model trong bộ nhớ."""

from __future__ import annotations

import argparse
import json
import sys
from time import perf_counter

from config import TOP_K, get_chunking_settings, get_index_directory
from rag import RAGSession, ask, build_index, retrieve, retrieve_with_debug


def add_retrieval_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index-dir", help="Index cần đọc; ghi đè CHROMA_DIR/thư mục theo strategy.")
    parser.add_argument(
        "--mode", choices=["semantic", "bm25", "hybrid"],
        help="Ghi đè RETRIEVAL_MODE cho lệnh này.",
    )
    parser.add_argument(
        "--rerank", action=argparse.BooleanOptionalAction, default=None,
        help="Bật/tắt reranker cho lệnh này (--rerank / --no-rerank).",
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAG LangChain tối giản cho tài liệu bệnh cây tiếng Việt."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    index_parser = commands.add_parser("index", help="Đọc data/**/*.txt và build lại Chroma.")
    index_parser.add_argument("--strategy", choices=["recursive", "structure"], help="Ghi đè CHUNKING_STRATEGY.")
    index_parser.add_argument("--index-dir", help="Nơi ghi index; mặc định tách theo strategy.")
    index_parser.add_argument("--data-dir", help="Corpus khác, ví dụ snapshot trước migration.")

    preview_parser = commands.add_parser("preview-chunks", help="Xem chunk/token/span mà không tạo index.")
    preview_parser.add_argument("--strategy", choices=["recursive", "structure"])
    preview_parser.add_argument("--data-dir")
    preview_parser.add_argument("--source", help="Đường tương đối, ví dụ apple/apple_black_rot.txt.")
    preview_parser.add_argument("--output", help="Ghi JSONL và file thống kê bên cạnh.")

    search_parser = commands.add_parser(
        "search", help="Kiểm tra các chunk được retrieval mà không gọi LLM."
    )
    search_parser.add_argument("question", help="Câu hỏi cần tìm trong kho dữ liệu.")
    search_parser.add_argument("--top-k", type=int, default=TOP_K)
    add_retrieval_options(search_parser)
    search_parser.add_argument(
        "--debug", action="store_true", help="In thứ hạng và điểm từng nhánh retrieval."
    )

    ask_parser = commands.add_parser(
        "ask", help="Retrieval rồi dùng LLM đã cấu hình để tạo câu trả lời."
    )
    ask_parser.add_argument("question", help="Câu hỏi gửi tới hệ thống RAG.")
    ask_parser.add_argument("--top-k", type=int, default=TOP_K)
    add_retrieval_options(ask_parser)

    chat_parser = commands.add_parser(
        "chat", help="Hỏi liên tục trong cùng process, dùng lại model trong RAM/VRAM."
    )
    chat_parser.add_argument("--top-k", type=int, default=TOP_K)
    add_retrieval_options(chat_parser)
    chat_parser.add_argument(
        "--search-only", action="store_true", help="Chỉ retrieval, không gọi LLM."
    )
    chat_parser.add_argument(
        "--debug", action="store_true", help="In query đã viết lại và trace retrieval."
    )
    chat_parser.add_argument(
        "--history-turns", type=int, default=None,
        help="Số lượt hỏi/đáp gần nhất được giữ; 0 để tắt lịch sử.",
    )
    return parser


def print_documents(documents) -> None:
    if not documents:
        print("Không tìm thấy chunk liên quan.")
        return
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "không rõ nguồn")
        print(f"\n[Nguồn {index}: {source}]")
        print(document.page_content)


def print_answer(answer: str, sources: list[str]) -> None:
    print(answer)
    if sources:
        print("\nNguồn đã truy xuất:")
        for source in sources:
            print(f"- {source}")


def run_chat(args: argparse.Namespace, options: dict) -> int:
    """Một session cho cả vòng lặp; lỗi một câu hỏi không làm mất model đã nạp."""
    if args.history_turns is not None:
        options = {**options, "history_turns": args.history_turns}
    session = RAGSession(k=args.top_k, **options)
    print("Đang mở index và nạp model cần dùng...", flush=True)
    started = perf_counter()
    session.warmup()
    print(f"Chuẩn bị xong trong {perf_counter() - started:.2f}s.")
    print("Nhập câu hỏi; /reset để xóa lịch sử; /exit hoặc Ctrl+C để thoát.")
    if args.search_only:
        print("Chế độ search-only: các query độc lập, không dùng lịch sử hoặc gọi LLM.")

    while True:
        try:
            question = input("\nBạn> ").strip()
            if question.casefold() in {"/exit", "/quit"}:
                return 0
            if question.casefold() == "/reset":
                session.clear_history()
                print("Đã xóa lịch sử hội thoại. Model vẫn được giữ trong bộ nhớ.")
                continue
            if not question:
                continue
            started = perf_counter()
            if args.search_only:
                result = session.search(question)
                if args.debug:
                    print(json.dumps(result.to_debug_dict(), ensure_ascii=False, indent=2))
                print_documents(result.documents)
            elif args.debug:
                result = session.ask_with_debug(question)
                print(json.dumps(result.to_debug_dict(), ensure_ascii=False, indent=2))
                print_answer(result.answer, result.sources)
            else:
                print_answer(*session.ask(question))
            print(f"\nThời gian xử lý: {perf_counter() - started:.2f}s.")
        except (EOFError, KeyboardInterrupt):
            print("\nĐã kết thúc phiên.")
            return 0
        except Exception as exc:  # Lỗi query/provider: báo lỗi rồi nhận câu tiếp theo.
            print(f"Lỗi: {exc}", file=sys.stderr)


def run(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    try:
        if args.command == "index":
            options = {}
            if args.strategy is not None:
                options["strategy"] = args.strategy
            if args.index_dir is not None:
                options["persist_directory"] = args.index_dir
            if args.data_dir is not None:
                options["data_dir"] = args.data_dir
            file_count, chunk_count = build_index(**options)
            directory = get_index_directory(args.index_dir, strategy=args.strategy)
            print(
                f"Đã index {file_count} tài liệu thành {chunk_count} chunk "
                f"trong {directory}/ ({get_chunking_settings(strategy=args.strategy).strategy})."
            )
            return 0

        if args.command == "preview-chunks":
            from chunk_preview import export_preview, preview_records, summarize
            options = {"strategy": args.strategy, "source": args.source}
            if args.data_dir:
                options["data_dir"] = args.data_dir
            records = preview_records(**options)
            if args.output:
                print(f"Đã export: {export_preview(records, args.output).resolve()}")
            else:
                for record in records:
                    print(json.dumps(record, ensure_ascii=False, indent=2))
            print(json.dumps(summarize(records), ensure_ascii=False, indent=2))
            return 0

        options = {}
        if args.index_dir is not None:
            options["persist_directory"] = args.index_dir
        if args.mode is not None:
            options["mode"] = args.mode
        if args.rerank is not None:
            options["rerank"] = args.rerank

        if args.command == "chat":
            return run_chat(args, options)

        if args.command == "search":
            if args.debug:
                result = retrieve_with_debug(args.question, k=args.top_k, **options)
                print(json.dumps(result.to_debug_dict(), ensure_ascii=False, indent=2))
                documents = result.documents
            else:
                documents = retrieve(args.question, k=args.top_k, **options)
            print_documents(documents)
            return 0

        answer, sources = ask(args.question, k=args.top_k, **options)
        print_answer(answer, sources)
        return 0
    except KeyboardInterrupt:
        print("\nĐã dừng chương trình.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports provider errors.
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(run())
