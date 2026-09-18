"""So sánh hai index thật bằng evidence trong source, không dùng chunk ID làm nhãn.

python scripts/benchmark_chunking.py --build --output artifacts/chunking/benchmark.json
Mặc định A=recursive trên snapshot raw, B=structure trên data hiện tại, hybrid,
không reranker/LLM. --rerank đánh giá riêng C/D bằng model reranker thật.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chunking import load_tokenizer, token_counter
from config import COLLECTION_NAME, EMBEDDING_MODEL, get_index_directory, get_retrieval_settings
from index_manifest import read_manifest
from rag import RAGSession, build_index, load_documents


@dataclass(frozen=True)
class EvidenceSpan:
    source: str
    start: int
    end: int
    positions: frozenset[int]  # Chỉ ký tự có nội dung; bỏ whitespace giữa hai chunk.


def resolve_evidence(rows, documents):
    """Map quote nguyên văn sang vị trí trong từng corpus (marker làm offset thay đổi)."""
    sources = {d.metadata["source"]: d.page_content for d in documents}
    resolved = {}
    for row in rows:
        resolved[row["id"]] = {}
        for field in ("expected_evidence", "required_context"):
            spans = []
            for label in row[field]:
                text = sources[label["source"]]
                if text.count(label["quote"]) != 1:
                    raise ValueError(f"{row['id']}: quote phải khớp duy nhất trong {label['source']}")
                start = text.index(label["quote"])
                positions = frozenset(start + i for i, char in enumerate(label["quote"]) if not char.isspace())
                if not positions:
                    raise ValueError(f"{row['id']}: evidence không có nội dung.")
                spans.append(EvidenceSpan(label["source"], start, start + len(label["quote"]), positions))
            resolved[row["id"]][field] = spans
    return resolved


def coverage(label, documents):
    """Union span: overlap không được tính điểm hai lần."""
    covered = set()
    for document in documents:
        meta = document.metadata
        if meta.get("source") != label.source or meta.get("start_index", -1) < 0:
            continue
        left, right = max(label.start, meta["start_index"]), min(label.end, meta["end_index"])
        covered.update(range(left, right))
    return len(label.positions.intersection(covered)) / len(label.positions)


def score_documents(documents, labels):
    evidence, context = labels["expected_evidence"], labels["required_context"]
    fractions = [coverage(label, documents) for label in evidence]
    complete = [fraction >= 1 for fraction in fractions]
    hit1 = any(coverage(label, documents[:1]) >= 1 for label in evidence)
    first_rank = next((rank for rank in range(1, len(documents) + 1)
                       if any(coverage(label, documents[rank - 1:rank]) >= 1 for label in evidence)), None)
    expected_sources = {label.source for label in evidence}
    wrong_section = sum(d.metadata["source"] in expected_sources
                        and not any(coverage(label, [d]) > 0 for label in evidence + context) for d in documents)
    duplicates = sum(max(0, sum(coverage(label, [d]) >= 1 for d in documents) - 1) for label in evidence)
    return {
        "hit_at_1": int(hit1), "recall_at_k": mean(complete) if complete else None,
        "evidence_character_coverage": mean(fractions) if fractions else None,
        "mrr_at_k": (1 / first_rank if first_rank else 0) if len(evidence) == 1 else None,
        "required_context_recall": mean(coverage(label, documents) >= 1 for label in context) if context else None,
        "wrong_section_hits": wrong_section, "duplicate_evidence_hits": duplicates,
    }


def distribution(values):
    values = sorted(values)
    return {"median": median(values) if values else None,
            "p95": values[math.ceil(len(values) * .95) - 1] if values else None}


def aggregate(results, rows):
    output = {}
    for split in ("dev", "heldout", "all"):
        chosen = [results[row["id"]] for row in rows if row["answerable"] and (split == "all" or row["split"] == split)]
        metrics = {"answerable_queries": len(chosen)}
        for name in ("hit_at_1", "recall_at_k", "mrr_at_k", "evidence_character_coverage",
                     "required_context_recall", "wrong_section_hits", "duplicate_evidence_hits"):
            values = [item[name] for item in chosen if item[name] is not None]
            metrics[name] = mean(values) if values else None
        metrics["hit_at_1_count"] = sum(item["hit_at_1"] for item in chosen)
        metrics["latency_seconds"] = distribution([item["latency_seconds"] for item in chosen])
        metrics["context_tokens"] = distribution([item["context_tokens"] for item in chosen])
        output[split] = metrics
    return output


def run(args):
    rows = [json.loads(line) for line in Path(args.queries).read_text(encoding="utf-8").splitlines() if line.strip()]
    count = token_counter(load_tokenizer(EMBEDDING_MODEL))
    settings = get_retrieval_settings(mode=args.mode, rerank=args.rerank)
    report = {"metric_version": "evidence-non-whitespace-v1", "k": args.top_k,
              "retrieval": asdict(settings), "embedding_tokenizer": EMBEDDING_MODEL,
              "queries_sha256": hashlib.sha256(Path(args.queries).read_bytes()).hexdigest(), "runs": {}}
    for name, strategy, data_dir, index_dir in (
        ("A", "recursive", args.data_a, args.index_a), ("B", "structure", args.data_b, args.index_b),
    ):
        index_dir = get_index_directory(index_dir)
        started = perf_counter()
        if args.build:
            print(f"Building {name}: {strategy} -> {index_dir}", flush=True)
            build_index(data_dir, index_dir, strategy=strategy)
        index_seconds = perf_counter() - started if args.build else None
        documents = load_documents(data_dir)
        manifest = read_manifest(Path(index_dir), COLLECTION_NAME)
        if manifest is None:
            raise ValueError("Benchmark cần index có manifest; dùng --build để tạo snapshot riêng.")
        if manifest["chunking"]["strategy"] != strategy:
            raise ValueError(f"Index {name} phải dùng strategy {strategy}.")
        actual_hashes = {d.metadata["source"]: hashlib.sha256(d.page_content.encode()).hexdigest() for d in documents}
        if actual_hashes != {f["source"]: f["sha256"] for f in manifest["files"]}:
            raise ValueError(f"Corpus {name} không khớp manifest index; hãy build lại đúng data-dir.")
        labels = resolve_evidence(rows, documents)
        session = RAGSession(persist_directory=index_dir, mode=args.mode, rerank=args.rerank, k=args.top_k)
        session.warmup()
        results = {}
        pair_tokenizer = load_tokenizer(settings.reranker_model) if args.rerank else None
        for index, row in enumerate(rows, 1):
            started = perf_counter()
            found = session.search(row["query"])
            duration = perf_counter() - started
            score = score_documents(found.documents, labels[row["id"]])
            score.update(latency_seconds=duration, context_tokens=count("\n\n".join(d.page_content for d in found.documents)))
            score["hits"] = [{"metadata": d.metadata, "text": d.page_content} for d in found.documents]
            if pair_tokenizer is not None:
                # Đếm chính cặp mà cross encoder nhận. Đây là cờ audit, chưa khẳng định
                # phần bị cắt có/không chứa evidence (cần đọc từng cặp vượt ngưỡng).
                pair_lengths = [len(pair_tokenizer(row["query"], d.page_content, truncation=False)["input_ids"])
                                for d in found.documents]
                score["reranker_pair_tokens"] = pair_lengths
                score["reranker_pairs_over_512"] = sum(length > 512 for length in pair_lengths)
            results[row["id"]] = score
            if index % 10 == 0:
                print(f"{name}: {index}/{len(rows)} queries", flush=True)
        report["runs"][name] = {"manifest": manifest, "index_seconds": index_seconds,
                                 "summary": aggregate(results, rows), "queries": results}
    report["comparison"] = []
    for row in rows:
        if not row["answerable"]:
            continue
        left = report["runs"]["A"]["queries"][row["id"]]
        right = report["runs"]["B"]["queries"][row["id"]]
        a, b = (left["recall_at_k"], left["hit_at_1"]), (right["recall_at_k"], right["hit_at_1"])
        report["comparison"].append({"id": row["id"], "query": row["query"], "split": row["split"],
                                     "change": "better" if b > a else "worse" if b < a else "same",
                                     "a": a, "b": b})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: data["summary"] for name, data in report["runs"].items()}, ensure_ascii=False, indent=2))
    print(f"Report: {output.resolve()}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--data-a", default=str(ROOT / "artifacts/chunking/baseline/data"))
    parser.add_argument("--data-b", default=str(ROOT / "data"))
    parser.add_argument("--index-a", default=str(ROOT / "artifacts/chunking/recursive_db"))
    parser.add_argument("--index-b", default=str(ROOT / "artifacts/chunking/structure_db"))
    parser.add_argument("--queries", default=str(ROOT / "eval/chunking_queries.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "artifacts/chunking/benchmark.json"))
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--mode", choices=["semantic", "bm25", "hybrid"], default="hybrid")
    parser.add_argument("--rerank", action="store_true")
    run(parser.parse_args())
