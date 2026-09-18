"""Export chunk để đọc và so sánh; không mở Chroma hay tải model weights."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from statistics import median

from chunking import load_tokenizer, structure_records, token_counter
from config import DATA_DIR, EMBEDDING_MODEL, get_chunking_settings


def preview_records(*, data_dir=DATA_DIR, strategy=None, source=None, tokenizer=None):
    from rag import load_documents, split_documents

    documents = load_documents(data_dir)
    if source:
        documents = [doc for doc in documents if doc.metadata["source"] == source.replace("\\", "/")]
        if not documents:
            raise ValueError(f"Không tìm thấy --source '{source}' trong data-dir.")
    settings = get_chunking_settings(strategy=strategy)
    tokenizer = tokenizer if tokenizer is not None else load_tokenizer(settings.tokenizer_model or EMBEDDING_MODEL)
    if settings.strategy == "structure":
        return structure_records(documents, settings, tokenizer=tokenizer)
    count = token_counter(tokenizer)
    sources = {doc.metadata["source"]: doc.page_content for doc in documents}
    records = []
    for chunk in split_documents(documents, strategy="recursive"):
        header, body = chunk.page_content.split("\n\n", 1)
        metadata = dict(chunk.metadata)
        text = sources[metadata["source"]]
        metadata.update(start_line=text.count("\n", 0, metadata["start_index"]) + 1,
                        end_line=text.count("\n", 0, metadata["end_index"] - 1) + 1)
        records.append({"kind": "chunk", "header": header, "body": body,
                        "text": chunk.page_content, "tokens": count(chunk.page_content),
                        "header_tokens": count(header), "metadata": metadata})
    return records


def summarize(records):
    chunks = [record for record in records if record["kind"] == "chunk"]
    tokens = sorted(record["tokens"] for record in chunks)
    return {
        "chunks": len(chunks), "metadata_records": sum(r["kind"] == "metadata" for r in records),
        "warnings": [r["message"] for r in records if r["kind"] == "warning"],
        "tokens": {"min": min(tokens, default=0), "median": median(tokens) if tokens else 0,
                   "p95": tokens[max(0, math.ceil(len(tokens) * .95) - 1)] if tokens else 0,
                   "max": max(tokens, default=0)},
        # Chỉ là tỷ lệ đếm riêng header; token(header+body) không có tính cộng chính xác.
        "header_token_ratio_estimate": sum(r["header_tokens"] for r in chunks) / max(1, sum(tokens)),
        "chunks_per_file": dict(Counter(r["metadata"]["source"] for r in chunks)),
    }


def export_preview(records, output: Path | str) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    path.with_suffix(".summary.json").write_text(
        json.dumps(summarize(records), ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return path
