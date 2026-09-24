"""So số liệu tự động giữa các variant: độ trễ, token, câu bị cắt, câu từ chối, trích dẫn sai.

    python -m experiments.stats experiments/runs/qwen4b-structure.jsonl experiments/runs/qwen4b-recursive.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.generate import load_rows, summarize

COLUMNS = (
    ("Variant", "variant"), ("Model", "model"), ("Index", "index"), ("Câu", "questions"),
    ("Lỗi", "errors"), ("Từ chối dựng sẵn", "refused"), ("Bị cắt", "truncated"),
    ("Trích dẫn sai", "invalid_citations"), ("Sinh (ms, trung vị)", "generate_ms_median"),
    ("Sinh (ms, P95)", "generate_ms_p95"), ("Token ra (TB)", "output_tokens_mean"),
    ("Độ dài (ký tự, trung vị)", "answer_chars_median"),
)


def describe(path: Path) -> dict[str, Any]:
    rows = load_rows(path)
    run_path = path.with_suffix(".run.json")
    run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {}
    return {"variant": run.get("variant", path.stem), "model": run.get("model", "?"),
            "index": run.get("index", "?"), **summarize(list(rows.values()))}


def table(results: list[dict[str, Any]]) -> str:
    lines = ["| " + " | ".join(title for title, _ in COLUMNS) + " |",
             "| " + " | ".join("---" if key in {"variant", "model", "index"} else "---:" for _, key in COLUMNS) + " |"]
    for result in results:
        cells = ["—" if result[key] is None else str(result[key]) for _, key in COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="Các file runs/<variant>.jsonl.")
    args = parser.parse_args(argv)
    print(table([describe(path) for path in args.runs]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
