"""Thử memory với Ollama thật: python scripts/smoke_chat.py --mode bm25.

Script này gọi LLM thật, không nằm trong bộ pytest offline. BM25 mặc định giúp
kiểm tra hội thoại mà không cần nạp thêm embedding; có thể chọn hybrid.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_llm_settings
from rag import RAGSession


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["bm25", "semantic", "hybrid"], default="bm25")
    args = parser.parse_args()
    settings = get_llm_settings()
    if settings.provider != "ollama":
        raise RuntimeError("Script này cần LLM_PROVIDER=ollama trong .env.")

    print(
        f"Provider: {settings.provider}; model: {settings.ollama_model}; "
        f"mode: {args.mode}",
        flush=True,
    )
    session = RAGSession(mode=args.mode, rerank=False, history_turns=4)
    session.warmup()
    for question in [
        "Triệu chứng bệnh ghẻ táo là gì? Trả lời ngắn gọn.",
        "Vậy tác nhân gây bệnh đó là gì?",
        "Cháy lá sớm trên khoai tây do tác nhân nào?",
    ]:
        print(f"\nĐang hỏi: {question}", flush=True)
        started = perf_counter()
        result = session.ask_with_debug(question)
        print(json.dumps({
            "question": result.question,
            "retrieval_query": result.retrieval_query,
            "history_turns_used": result.history_turns_used,
            "sources": result.sources,
            "answer": result.answer,
            "elapsed_seconds": round(perf_counter() - started, 2),
        }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
