"""So sánh ba cấu hình chunking cho báo cáo.

    A  dữ liệu trước tái cấu trúc + recursive
    B  dữ liệu sau tái cấu trúc  + recursive
    C  dữ liệu sau tái cấu trúc  + structure

A→B tách tác động của việc viết lại dữ liệu, B→C tách tác động của cách chunk.
Ba index được build lại bằng cùng code trong artifacts/ (không đụng index đang dùng),
rồi đo đặc trưng index, retrieval (nhiều k, nhiều ngân sách token, ba mode) và kiểm
định thống kê theo cặp trên bộ eval có nhãn bằng chứng.

    python scripts/compare_chunking.py
    python scripts/compare_chunking.py --reuse-indexes   # chỉ đo lại, không build
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as package_metadata
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chunking import load_tokenizer, parse_document, token_counter  # noqa: E402
from config import (  # noqa: E402
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL,
    create_embeddings,
    get_chunking_settings,
    get_retrieval_settings,
)
from index_manifest import corpus_fingerprint, read_manifest  # noqa: E402
from rag import _load_vector_store, build_index, load_documents  # noqa: E402
from retrieval import search_store  # noqa: E402
from scripts.benchmark_chunking import coverage, resolve_evidence  # noqa: E402

MODES = ("hybrid", "semantic", "bm25")
K_VALUES = (1, 2, 3, 4, 5, 6, 8, 10)
BUDGETS = (256, 512, 768, 1024, 1536, 2048, 3072)
MAX_K = 20
SYSTEM_K = 4
BOOTSTRAP = 10_000
SEED = 20260924
SENTENCE_END = ".!?:;…)\"”»"


@dataclass(frozen=True)
class Config:
    key: str
    slug: str
    label: str
    data_dir: Path
    strategy: str


CONFIGS = (
    Config("A", "old-recursive", "Dữ liệu cũ + recursive", ROOT / "artifacts/chunking/baseline/data", "recursive"),
    Config("B", "new-recursive", "Dữ liệu mới + recursive", ROOT / "data", "recursive"),
    Config("C", "new-structure", "Dữ liệu mới + structure", ROOT / "data", "structure"),
)
CONTRASTS = (
    ("A", "B", "Viết lại dữ liệu (cùng recursive)"),
    ("B", "C", "Đổi cách chunk (cùng dữ liệu mới)"),
    ("A", "C", "Tổng hợp cả hai"),
)
# Độ đo chính để kiểm định; (tên, kiểu) với kiểu binary dùng McNemar, còn lại Wilcoxon.
TESTED_METRICS = (
    ("hit_at_1", "binary"), ("doc_hit_at_1", "binary"), ("recall_at_4", "continuous"),
    ("rr_at_10", "continuous"), ("ndcg_at_10", "continuous"), ("coverage_at_4", "continuous"),
    ("precision_at_4", "continuous"), ("evidence_density_at_4", "continuous"), ("recall_budget_1024", "continuous"),
    ("recall_budget_1536", "continuous"), ("context_tokens_at_4", "continuous"),
)
MAIN_METRICS = (
    ("hit_at_1", "Hit@1"), ("recall_at_4", "Recall@4"), ("coverage_at_4", "Coverage@4"),
    ("rr_at_10", "MRR@10"), ("ndcg_at_4", "nDCG@4"), ("ndcg_at_10", "nDCG@10"),
    ("precision_at_4", "Context precision@4"), ("evidence_density_at_4", "Mật độ bằng chứng@4"),
    ("doc_hit_at_1", "Doc Hit@1"),
    ("doc_recall_at_4", "Doc Recall@4"), ("doc_precision_at_4", "Doc precision@4"),
    ("recall_budget_1024", "Recall@1024 token"), ("recall_budget_1536", "Recall@1536 token"),
)


# --- Tiện ích ----------------------------------------------------------------------

def vn(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def pct(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else vn(value * 100, digits) + "%"


def quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q)) if values else float("nan")


def directory_size_mb(path: Path) -> float:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / 1_048_576


def git_revision() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                               text=True, check=True).stdout.strip()
        return commit + ("+thay đổi chưa commit" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "không rõ"


def version(package: str) -> str:
    try:
        return package_metadata.version(package)
    except package_metadata.PackageNotFoundError:
        return "không có"


# --- Đặc trưng index ------------------------------------------------------------------

def clean_start(text: str, start: int) -> bool:
    if start <= 0 or text[start - 1] == "\n":
        return True
    before = text[:start].rstrip()
    return not before or before[-1] in SENTENCE_END or before.endswith("\n")


def clean_end(text: str, end: int) -> bool:
    if end >= len(text) or text[end] == "\n" or text[end - 1] in SENTENCE_END:
        return True
    return text[end:].lstrip(" \t")[:1] in {"\n", ""}


def section_maps(documents) -> dict[str, list] | None:
    """Unit (thân mục) của từng tài liệu; None khi dữ liệu không có heading marker."""
    maps = {}
    for document in documents:
        try:
            maps[document.metadata["source"]] = parse_document(document).units
        except ValueError:
            return None
    return maps


def spans_touched(text: str, start: int, end: int, units, attribute: str) -> int:
    touched = set()
    for unit in units:
        left, right = max(start, unit.start_index), min(end, unit.end_index)
        if left < right and text[left:right].strip():
            touched.add(getattr(unit, attribute))
    return len(touched)


def chunk_records(config: Config, store, documents, count) -> list[dict[str, Any]]:
    texts = {d.metadata["source"]: d.page_content for d in documents}
    sections = section_maps(documents)
    raw = store.get(include=["documents", "metadatas"])
    records = []
    for chunk_id, content, meta in zip(raw["ids"], raw["documents"], raw["metadatas"]):
        text, start, end = texts[meta["source"]], int(meta["start_index"]), int(meta["end_index"])
        body = text[start:end]
        record = {
            "config": config.key, "chunk_id": chunk_id, "source": meta["source"],
            "heading_path": meta.get("heading_path", ""), "start": start, "end": end,
            "tokens": count(content), "body_tokens": count(body), "chars": len(content),
            "body_in_content": body.strip()[:80] in content,
            "clean_start": clean_start(text, start), "clean_end": clean_end(text, end),
            "sections": None, "units": None,
        }
        if sections is not None:
            units = sections[meta["source"]]
            record["sections"] = spans_touched(text, start, end, units, "section_id")
            record["units"] = spans_touched(text, start, end, units, "unit_id")
        records.append(record)
    return records


def corpus_coverage(documents, records, *, content_only: bool = False) -> float:
    """Tỉ lệ ký tự khác khoảng trắng của corpus nằm trong ít nhất một chunk.

    `content_only` bỏ dòng heading và thân các trường metadata (TÊN TÀI LIỆU, CÂY TRỒNG...):
    structure đưa hai phần này vào header/metadata của chunk thay vì thân chunk.
    """
    covered, total = 0, 0
    by_source: dict[str, list[tuple[int, int]]] = {}
    for record in records:
        by_source.setdefault(record["source"], []).append((record["start"], record["end"]))
    for document in documents:
        text = document.page_content
        mask = np.zeros(len(text), dtype=bool)
        for start, end in by_source.get(document.metadata["source"], []):
            mask[start:end] = True
        solid = np.array([not c.isspace() for c in text], dtype=bool)
        if content_only:
            offset = 0
            for line in text.splitlines(keepends=True):
                if re.match(r"^#{1,6}\s", line):
                    solid[offset:offset + len(line)] = False
                offset += len(line)
            try:
                for unit in parse_document(document).units:
                    if unit.metadata_field:
                        solid[unit.start_index:unit.end_index] = False
            except ValueError:
                pass  # dữ liệu cũ không có heading: không có trường metadata tách riêng
        covered += int((mask & solid).sum())
        total += int(solid.sum())
    return covered / total


def index_stats(config: Config, documents, records, count, build_seconds, size_mb) -> dict[str, Any]:
    tokens = [r["tokens"] for r in records]
    body = [r["body_tokens"] for r in records]
    corpus_tokens = sum(count(d.page_content) for d in documents)
    with_sections = [r for r in records if r["sections"] is not None]
    return {
        "config": config.key, "label": config.label, "strategy": config.strategy,
        "documents": len(documents), "corpus_chars": sum(len(d.page_content) for d in documents),
        "corpus_tokens": corpus_tokens, "chunks": len(records), "chunks_per_document": len(records) / len(documents),
        "tokens_mean": statistics.fmean(tokens), "tokens_median": statistics.median(tokens),
        "tokens_std": statistics.pstdev(tokens), "tokens_p05": quantile(tokens, 0.05),
        "tokens_p95": quantile(tokens, 0.95), "tokens_max": max(tokens), "body_tokens_median": statistics.median(body),
        "header_overhead": 1 - sum(body) / sum(tokens),
        "over_512_tokens": sum(t > 512 for t in tokens) / len(tokens),
        "redundancy": sum(body) / corpus_tokens,
        "corpus_coverage": corpus_coverage(documents, records),
        "content_coverage": corpus_coverage(documents, records, content_only=True),
        "clean_start": sum(r["clean_start"] for r in records) / len(records),
        "clean_end": sum(r["clean_end"] for r in records) / len(records),
        "clean_both": sum(r["clean_start"] and r["clean_end"] for r in records) / len(records),
        "single_section": (sum(r["sections"] <= 1 for r in with_sections) / len(with_sections)) if with_sections else None,
        "single_unit": (sum(r["units"] <= 1 for r in with_sections) / len(with_sections)) if with_sections else None,
        "sections_per_chunk": statistics.fmean(r["sections"] for r in with_sections) if with_sections else None,
        "body_matches": sum(r["body_in_content"] for r in records) / len(records),
        "build_seconds": build_seconds, "index_size_mb": size_mb,
    }


def evidence_rows(config: Config, rows, labels, records) -> list[dict[str, Any]]:
    """Bằng chứng có nằm trọn trong một chunk nào của index không (giới hạn trên của Hit@1)."""
    docs = [_Chunk(r) for r in records]
    output = []
    for row in rows:
        for number, label in enumerate(labels[row["id"]]["expected_evidence"], start=1):
            same_source = [d for d in docs if d.metadata["source"] == label.source]
            fractions = [coverage(label, [d]) for d in same_source]
            output.append({
                "config": config.key, "id": row["id"], "evidence": number, "source": label.source,
                "evidence_chars": len(label.positions),
                "contained": any(f >= 1 for f in fractions),
                "chunks_overlapping": sum(f > 0 for f in fractions),
                "best_single_chunk": max(fractions, default=0.0),
            })
    return output


class _Chunk:
    """Đủ thuộc tính để dùng lại coverage() của benchmark."""

    def __init__(self, record: dict[str, Any]) -> None:
        self.metadata = {"source": record["source"], "start_index": record["start"], "end_index": record["end"]}


# --- Retrieval ------------------------------------------------------------------------

def relevance(chunk, evidence) -> float:
    return max((coverage(label, [chunk]) for label in evidence), default=0.0)


def dcg(gains: list[float]) -> float:
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def evidence_density(chunks, labels, texts) -> float:
    """Tỉ lệ ký tự trong thân các chunk là bằng chứng/ngữ cảnh đã gán nhãn (precision mức ký tự).

    Không thiên vị chunk to như Context precision: phần chữ thừa trong chunk bị tính vào mẫu số.
    Overlap giữa các chunk được tính lặp, vì LLM cũng đọc lặp.
    """
    covered: dict[str, set[int]] = {}
    total = 0
    for chunk in chunks:
        source, start, end = chunk.metadata["source"], chunk.metadata["start_index"], chunk.metadata["end_index"]
        total += sum(not character.isspace() for character in texts[source][start:end])
        covered.setdefault(source, set()).update(range(start, end))
    relevant: set[tuple[str, int]] = set()
    for label in labels:
        relevant.update((label.source, position) for position in label.positions & covered.get(label.source, set()))
    return len(relevant) / total if total else 0.0


def score_query(row, labels, ranked, token_of, ideal_gains, texts) -> dict[str, Any]:
    evidence, context = labels["expected_evidence"], labels["required_context"]
    sources = {label.source for label in evidence}
    related = evidence + context
    gains = [relevance(chunk, evidence) for chunk in ranked]
    single = [any(coverage(label, [chunk]) >= 1 for label in evidence) for chunk in ranked]
    first = next((rank for rank, ok in enumerate(single, start=1) if ok), None)
    score: dict[str, Any] = {
        "hit_at_1": int(bool(single[:1] and single[0])),
        "rr_at_10": 1 / first if first and first <= 10 else 0.0,
        "first_full_rank": first,
    }
    for k in K_VALUES:
        top = ranked[:k]
        score[f"recall_at_{k}"] = statistics.fmean(coverage(label, top) >= 1 for label in evidence)
        score[f"coverage_at_{k}"] = statistics.fmean(coverage(label, top) for label in evidence)
    for k in (4, 10):
        ideal = dcg(ideal_gains[:k])
        score[f"ndcg_at_{k}"] = dcg(gains[:k]) / ideal if ideal else 0.0
    top4 = ranked[:SYSTEM_K]
    score["precision_at_4"] = statistics.fmean(
        any(coverage(label, [chunk]) > 0 for label in related) for chunk in top4)
    score["evidence_density_at_4"] = evidence_density(top4, related, texts)
    score["doc_hit_at_1"] = int(bool(ranked) and ranked[0].metadata["source"] in sources)
    score["doc_recall_at_4"] = int(sources <= {chunk.metadata["source"] for chunk in top4})
    score["doc_precision_at_4"] = statistics.fmean(chunk.metadata["source"] in sources for chunk in top4)
    score["wrong_section_at_4"] = sum(
        chunk.metadata["source"] in sources and not any(coverage(label, [chunk]) > 0 for label in related)
        for chunk in top4)
    score["context_tokens_at_4"] = sum(token_of(chunk) for chunk in top4)
    score["required_context_recall_at_4"] = (
        statistics.fmean(coverage(label, top4) >= 1 for label in context) if context else None)
    for budget in BUDGETS:
        used, chosen = 0, []
        for chunk in ranked:
            size = token_of(chunk)
            if used + size > budget:
                break
            used += size
            chosen.append(chunk)
        score[f"recall_budget_{budget}"] = statistics.fmean(coverage(label, chosen) >= 1 for label in evidence)
        score[f"coverage_budget_{budget}"] = statistics.fmean(coverage(label, chosen) for label in evidence)
        score[f"chunks_budget_{budget}"] = len(chosen)
    return score


def run_retrieval(config, store, rows, labels, records, token_cache, texts) -> tuple[list[dict], list[dict]]:
    chunks = [_Chunk(r) for r in records]

    def token_of(document) -> int:
        return token_cache[document.page_content]

    per_query, top_hits = [], []
    for mode in MODES:
        settings = get_retrieval_settings(mode=mode, rerank=False)
        search_store("khởi động", store, k=MAX_K, settings=settings)  # dựng BM25/cache, không tính giờ
        for row in rows:
            evidence = labels[row["id"]]["expected_evidence"]
            ideal = sorted((relevance(chunk, evidence) for chunk in chunks
                            if chunk.metadata["source"] in {label.source for label in evidence}), reverse=True)
            started = perf_counter()
            result = search_store(row["query"], store, k=MAX_K, settings=settings)
            latency = (perf_counter() - started) * 1000
            ranked = result.documents
            score = score_query(row, labels[row["id"]], ranked, token_of, ideal, texts)
            per_query.append({"config": config.key, "mode": mode, "id": row["id"], "split": row["split"],
                              "category": row["category"], "evidence_count": len(evidence),
                              "latency_ms": latency, **score})
            if mode == "hybrid":
                top_hits.append({"config": config.key, "id": row["id"], "query": row["query"], "hits": [
                    {"rank": rank, "source": d.metadata["source"], "heading_path": d.metadata.get("heading_path"),
                     "start": d.metadata["start_index"], "end": d.metadata["end_index"],
                     "relevance": round(relevance(d, evidence), 3), "tokens": token_of(d),
                     "preview": d.page_content[:240]}
                    for rank, d in enumerate(ranked[:SYSTEM_K], start=1)]})
    return per_query, top_hits


# --- Thống kê -------------------------------------------------------------------------

def bootstrap_mean(values: np.ndarray, rng) -> tuple[float, float]:
    samples = values[rng.integers(0, len(values), size=(BOOTSTRAP, len(values)))].mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def mcnemar(left: np.ndarray, right: np.ndarray) -> tuple[int, int, float]:
    only_right = int(((right == 1) & (left == 0)).sum())
    only_left = int(((left == 1) & (right == 0)).sum())
    total = only_left + only_right
    if total == 0:
        return only_left, only_right, 1.0
    tail = sum(math.comb(total, i) for i in range(min(only_left, only_right) + 1)) / 2 ** total
    return only_left, only_right, min(1.0, 2 * tail)


def wilcoxon(left: np.ndarray, right: np.ndarray) -> float:
    if np.allclose(left, right):
        return 1.0
    return float(scipy_stats.wilcoxon(right, left, zero_method="wilcox").pvalue)


def holm(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted, running = [0.0] * len(p_values), 0.0
    for position, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - position) * p_values[index]))
        adjusted[index] = running
    return adjusted


def metric_matrix(per_query, config, mode, metric, ids) -> np.ndarray:
    lookup = {(r["config"], r["mode"], r["id"]): r[metric] for r in per_query}
    return np.array([lookup[(config, mode, qid)] for qid in ids], dtype=float)


def significance(per_query, ids) -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED)
    output = []
    for metric, kind in TESTED_METRICS:
        tests = []
        for left, right, meaning in CONTRASTS:
            x = metric_matrix(per_query, left, "hybrid", metric, ids)
            y = metric_matrix(per_query, right, "hybrid", metric, ids)
            diff = y - x
            low, high = bootstrap_mean(diff, rng)
            test = {"metric": metric, "contrast": f"{left}→{right}", "meaning": meaning,
                    "left_mean": float(x.mean()), "right_mean": float(y.mean()), "diff": float(diff.mean()),
                    "diff_ci_low": low, "diff_ci_high": high,
                    "better": int((diff > 0).sum()), "worse": int((diff < 0).sum()), "same": int((diff == 0).sum())}
            if kind == "binary":
                only_left, only_right, p = mcnemar(x, y)
                test.update(test_name="McNemar chính xác", p_value=p, only_left=only_left, only_right=only_right)
            else:
                test.update(test_name="Wilcoxon signed-rank", p_value=wilcoxon(x, y))
            tests.append(test)
        for test, adjusted in zip(tests, holm([t["p_value"] for t in tests])):
            test["p_holm"] = adjusted
        output.extend(tests)
    return output


def aggregate(per_query, config, mode, ids, rng) -> dict[str, Any]:
    result = {"config": config, "mode": mode, "queries": len(ids)}
    for metric, _ in MAIN_METRICS + (("context_tokens_at_4", ""), ("wrong_section_at_4", "")):
        values = metric_matrix(per_query, config, mode, metric, ids)
        result[metric] = float(values.mean())
        result[f"{metric}_ci"] = bootstrap_mean(values, rng)
    tokens = metric_matrix(per_query, config, mode, "context_tokens_at_4", ids)
    result["context_tokens_at_4_median"] = float(np.median(tokens))
    result["context_tokens_at_4_p95"] = float(np.percentile(tokens, 95))
    latency = metric_matrix(per_query, config, mode, "latency_ms", ids)
    result["latency_ms_median"], result["latency_ms_p95"] = float(np.median(latency)), float(np.percentile(latency, 95))
    rcr = [r["required_context_recall_at_4"] for r in per_query
           if r["config"] == config and r["mode"] == mode and r["required_context_recall_at_4"] is not None]
    result["required_context_recall_at_4"] = statistics.fmean(rcr) if rcr else None
    result["required_context_queries"] = len(rcr)
    return result


# --- Xuất kết quả ---------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:  # utf-8-sig: Excel đọc đúng tiếng Việt
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, tuple, dict)) else v)
                             for k, v in row.items()})


def figures(out: Path, per_query, stats_rows, records_by_config, aggregates, ids) -> list[str]:
    try:
        import matplotlib
    except ImportError:
        print("Bỏ qua biểu đồ: chưa cài matplotlib (`python -m pip install matplotlib`).", flush=True)
        return []

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = out / "figures"
    folder.mkdir(exist_ok=True)
    colors = {"A": "#9e9e9e", "B": "#1f77b4", "C": "#d62728"}
    labels = {c.key: f"{c.key}: {c.label}" for c in CONFIGS}
    written = []

    def save(fig, name: str) -> None:
        fig.tight_layout()
        fig.savefig(folder / name, dpi=200)
        plt.close(fig)
        written.append(f"figures/{name}")

    fig, ax = plt.subplots(figsize=(6.4, 4))
    for config in CONFIGS:
        ax.plot(K_VALUES, [metric_matrix(per_query, config.key, "hybrid", f"recall_at_{k}", ids).mean() for k in K_VALUES],
                marker="o", color=colors[config.key], label=labels[config.key])
    ax.set(xlabel="Số chunk lấy về (k)", ylabel="Recall@k", title="Recall theo số chunk (hybrid)", ylim=(0, 1.02))
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, "recall_at_k.png")

    fig, ax = plt.subplots(figsize=(6.4, 4))
    for config in CONFIGS:
        ax.plot(BUDGETS, [metric_matrix(per_query, config.key, "hybrid", f"recall_budget_{b}", ids).mean() for b in BUDGETS],
                marker="o", color=colors[config.key], label=labels[config.key])
    ax.set(xlabel="Ngân sách ngữ cảnh (token)", ylabel="Recall", title="Recall theo ngân sách token (hybrid)", ylim=(0, 1.02))
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, "recall_at_budget.png")

    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.boxplot([[r["tokens"] for r in records_by_config[c.key]] for c in CONFIGS],
               tick_labels=[labels[c.key] for c in CONFIGS], showfliers=True)
    ax.set(ylabel="Token mỗi chunk (tokenizer embedding)", title="Phân bố kích thước chunk")
    ax.grid(alpha=0.3, axis="y")
    ax.tick_params(axis="x", labelsize=8)
    save(fig, "chunk_tokens.png")

    def grouped_bars(ax, shown) -> None:
        width = 0.27
        for offset, config in enumerate(CONFIGS):
            agg = aggregates[(config.key, "hybrid")]
            means = [agg[m] for m, _ in shown]
            errors = np.array([[agg[m] - agg[f"{m}_ci"][0] for m, _ in shown],
                               [agg[f"{m}_ci"][1] - agg[m] for m, _ in shown]])
            ax.bar(np.arange(len(shown)) + (offset - 1) * width, means, width, yerr=errors, capsize=2,
                   color=colors[config.key], label=labels[config.key])
        ax.set_xticks(np.arange(len(shown)), [name for _, name in shown])
        ax.grid(alpha=0.3, axis="y")

    shown = (("hit_at_1", "Hit@1"), ("recall_at_4", "Recall@4"), ("rr_at_10", "MRR@10"),
             ("ndcg_at_10", "nDCG@10"), ("recall_budget_1024", "Recall@\n1024 token"),
             ("recall_budget_1536", "Recall@\n1536 token"))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    grouped_bars(ax, shown)
    ax.set(ylim=(0, 1.05), title="Độ đo chính (hybrid, khoảng tin cậy 95% bootstrap)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    save(fig, "main_metrics.png")

    fig, (left, right) = plt.subplots(1, 2, figsize=(8, 3.8))
    grouped_bars(left, (("context_tokens_at_4", "Token ngữ cảnh top-4"),))
    left.set(title="Chi phí ngữ cảnh (thấp hơn là rẻ hơn)")
    grouped_bars(right, (("evidence_density_at_4", "Mật độ bằng chứng@4"),))
    right.set(title="Tỉ lệ ngữ cảnh là bằng chứng")
    handles, names_ = left.get_legend_handles_labels()
    fig.legend(handles, names_, fontsize=8, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(folder / "cost_density.png", dpi=200)
    plt.close(fig)
    written.append("figures/cost_density.png")

    categories = sorted({r["category"] for r in per_query if r["id"] in set(ids)})
    fig, ax = plt.subplots(figsize=(7.5, 6))
    height = 0.27
    for offset, config in enumerate(CONFIGS):
        values = [statistics.fmean(r["recall_at_4"] for r in per_query if r["config"] == config.key
                                   and r["mode"] == "hybrid" and r["category"] == cat) for cat in categories]
        ax.barh(np.arange(len(categories)) + (offset - 1) * height, values, height,
                color=colors[config.key], label=labels[config.key])
    counts = {cat: sum(1 for r in per_query if r["config"] == "A" and r["mode"] == "hybrid" and r["category"] == cat)
              for cat in categories}
    ax.set_yticks(np.arange(len(categories)), [f"{cat} (n={counts[cat]})" for cat in categories], fontsize=8)
    ax.set(xlabel="Recall@4", xlim=(0, 1.05), title="Recall@4 theo loại câu hỏi (hybrid)")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.4, -0.09), ncol=3, frameon=False)
    save(fig, "recall_by_category.png")
    return written


def markdown_table(headers: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or "l" + "r" * (len(headers) - 1)
    marks = ["---:" if a == "r" else "---" for a in align]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(marks) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def results_markdown(summary: dict[str, Any], index_rows, aggregates, per_query, stats_rows, evidence, ids) -> str:
    keys = [c.key for c in CONFIGS]
    names = {c.key: f"{c.key}. {c.label}" for c in CONFIGS}
    idx = {row["config"]: row for row in index_rows}
    out = [
        "# Kết quả so sánh chunking (tự sinh)",
        "",
        f"Sinh bởi `scripts/compare_chunking.py` lúc {summary['generated_at']}. Đừng sửa tay: chạy lại script sẽ ghi đè.",
        "Phần nhận xét nằm ở `README.md`.",
        "",
        "## 1. Thiết lập",
        "",
        markdown_table(["Cấu hình", "Dữ liệu", "Chunking", "Số tài liệu", "Fingerprint dữ liệu"], [
            [names[c.key], c.data_dir.relative_to(ROOT).as_posix(),
             "recursive 1.000 ký tự, overlap 150" if c.strategy == "recursive" else "structure ≤ 400 token, overlap 40",
             str(idx[c.key]["documents"]), summary["configs"][c.key]["fingerprint"][:12]] for c in CONFIGS], "lllrl"),
        "",
        f"- Bộ eval: `eval/chunking_queries.jsonl`, {summary['queries_total']} câu, tính điểm trên "
        f"**{len(ids)} câu có đáp án** ({summary['splits']['dev']} dev, {summary['splits']['heldout']} heldout; "
        f"{summary['evidence_spans']} đoạn bằng chứng). 2 câu ngoài corpus bị loại khỏi điểm.",
        f"- Embedding `{EMBEDDING_MODEL}` ({EMBEDDING_DEVICE}); token đếm bằng tokenizer của embedding.",
        f"- Retrieval chính: hybrid (semantic + BM25, RRF k={summary['retrieval']['rrf_k']}, "
        f"{summary['retrieval']['candidate_k']} ứng viên mỗi nhánh), không reranker; hệ thống dùng k = {SYSTEM_K}.",
        f"- Khoảng tin cậy 95%: bootstrap {BOOTSTRAP:,} lần theo câu hỏi (seed {SEED}).".replace(",", "."),
        f"- Git `{summary['git']}`; Python {summary['versions']['python']}, langchain-core "
        f"{summary['versions']['langchain-core']}, chromadb {summary['versions']['chromadb']}.",
        "",
        "## 2. Đặc trưng index",
        "",
    ]
    rows = [
        ("Số chunk", lambda r: vn(r["chunks"], 0)),
        ("Chunk / tài liệu", lambda r: vn(r["chunks_per_document"], 1)),
        ("Token / chunk: trung bình", lambda r: vn(r["tokens_mean"], 0)),
        ("Token / chunk: trung vị", lambda r: vn(r["tokens_median"], 0)),
        ("Token / chunk: P5–P95", lambda r: f"{vn(r['tokens_p05'], 0)}–{vn(r['tokens_p95'], 0)}"),
        ("Token / chunk: lớn nhất", lambda r: vn(r["tokens_max"], 0)),
        ("Độ lệch chuẩn token", lambda r: vn(r["tokens_std"], 0)),
        ("Tỉ lệ token dành cho header", lambda r: pct(r["header_overhead"])),
        ("Chunk > 512 token", lambda r: pct(r["over_512_tokens"])),
        ("Hệ số lặp (token thân chunk / token corpus)", lambda r: vn(r["redundancy"], 2)),
        ("Phủ corpus (ký tự nằm trong thân ≥ 1 chunk)", lambda r: pct(r["corpus_coverage"])),
        ("Phủ nội dung (không tính dòng heading, trường metadata)", lambda r: pct(r["content_coverage"])),
        ("Bắt đầu ở đầu dòng/câu", lambda r: pct(r["clean_start"])),
        ("Kết thúc ở cuối dòng/câu", lambda r: pct(r["clean_end"])),
        ("Không cắt ngang câu (cả hai đầu)", lambda r: pct(r["clean_both"])),
        ("Nằm trong đúng 1 mục H2", lambda r: pct(r["single_section"])),
        ("Nằm trong đúng 1 mục lá", lambda r: pct(r["single_unit"])),
        ("Số mục H2 trung bình / chunk", lambda r: vn(r["sections_per_chunk"], 2)),
        ("Bằng chứng nằm trọn trong 1 chunk", lambda r: pct(summary["configs"][r["config"]]["evidence_contained"])),
        ("Số chunk chạm 1 bằng chứng (TB)", lambda r: vn(summary["configs"][r["config"]]["evidence_chunks_overlapping"], 2)),
        ("Thời gian build index (s)", lambda r: vn(r["build_seconds"], 1)),
        ("Dung lượng index (MB)", lambda r: vn(r["index_size_mb"], 1)),
    ]
    out.append(markdown_table(["Độ đo"] + [names[k] for k in keys],
                              [[name] + [fn(idx[k]) for k in keys] for name, fn in rows]))
    out += ["", "Mục H2/mục lá chỉ đo được trên dữ liệu mới (dữ liệu cũ không có heading marker).",
            "Thời gian build gồm tách chunk, embedding và ghi Chroma; model đã nạp sẵn trước khi bấm giờ.", "",
            "## 3. Retrieval chính (hybrid, k = 4)", ""]
    out.append(markdown_table(["Độ đo"] + [names[k] for k in keys], [
        [name] + [f"{vn(aggregates[(k, 'hybrid')][m])} [{vn(aggregates[(k, 'hybrid')][m + '_ci'][0])}–"
                  f"{vn(aggregates[(k, 'hybrid')][m + '_ci'][1])}]" for k in keys]
        for m, name in MAIN_METRICS]))
    out.append("")
    out.append(markdown_table(["Chi phí"] + [names[k] for k in keys], [
        ["Token ngữ cảnh top-4: trung bình"] + [vn(aggregates[(k, "hybrid")]["context_tokens_at_4"], 0) for k in keys],
        ["Token ngữ cảnh top-4: trung vị"] + [vn(aggregates[(k, "hybrid")]["context_tokens_at_4_median"], 0) for k in keys],
        ["Token ngữ cảnh top-4: P95"] + [vn(aggregates[(k, "hybrid")]["context_tokens_at_4_p95"], 0) for k in keys],
        ["Chunk đúng tài liệu nhưng sai mục (TB / 4)"] + [vn(aggregates[(k, "hybrid")]["wrong_section_at_4"], 2) for k in keys],
        ["Độ trễ retrieval: trung vị (ms)"] + [vn(aggregates[(k, "hybrid")]["latency_ms_median"], 1) for k in keys],
        ["Độ trễ retrieval: P95 (ms)"] + [vn(aggregates[(k, "hybrid")]["latency_ms_p95"], 1) for k in keys],
        [f"Required context recall@4 ({aggregates[('A', 'hybrid')]['required_context_queries']} câu)"]
        + [pct(aggregates[(k, "hybrid")]["required_context_recall_at_4"]) for k in keys],
    ]))
    out += ["", "Số trong ngoặc vuông là khoảng tin cậy 95%.", "", "## 4. Theo số chunk k (hybrid)", ""]
    out.append(markdown_table(["k"] + [f"Recall {k}" for k in keys] + [f"Coverage {k}" for k in keys], [
        [str(kv)] + [vn(metric_matrix(per_query, k, "hybrid", f"recall_at_{kv}", ids).mean()) for k in keys]
        + [vn(metric_matrix(per_query, k, "hybrid", f"coverage_at_{kv}", ids).mean()) for k in keys]
        for kv in K_VALUES]))
    out += ["", "## 5. Cùng ngân sách token ngữ cảnh (hybrid)", "",
            "Lấy chunk theo thứ hạng cho tới khi chunk kế tiếp làm vượt ngân sách. So sánh này công bằng hơn "
            "so cùng k, vì chunk của các cấu hình dài ngắn khác nhau.", ""]
    out.append(markdown_table(["Ngân sách"] + [f"Recall {k}" for k in keys] + [f"Số chunk TB {k}" for k in keys], [
        [vn(b, 0)] + [vn(metric_matrix(per_query, k, "hybrid", f"recall_budget_{b}", ids).mean()) for k in keys]
        + [vn(metric_matrix(per_query, k, "hybrid", f"chunks_budget_{b}", ids).mean(), 1) for k in keys]
        for b in BUDGETS]))
    out += ["", "## 6. Kiểm định theo cặp (hybrid)", "",
            "Nhị phân: McNemar chính xác; liên tục: Wilcoxon signed-rank. p Holm hiệu chỉnh cho 3 phép so sánh "
            "của cùng một độ đo. Chênh lệch = cấu hình sau − cấu hình trước, kèm khoảng tin cậy 95% bootstrap.", ""]
    labels_metric = dict(MAIN_METRICS) | {"context_tokens_at_4": "Token ngữ cảnh top-4"}
    out.append(markdown_table(["Độ đo", "So sánh", "Trước", "Sau", "Chênh lệch [KTC 95%]", "Tốt hơn/Kém hơn/Bằng", "p", "p Holm"], [
        [labels_metric[s["metric"]], f"{s['contrast']} ({s['meaning']})",
         vn(s["left_mean"], 3 if s["metric"] != "context_tokens_at_4" else 0),
         vn(s["right_mean"], 3 if s["metric"] != "context_tokens_at_4" else 0),
         f"{vn(s['diff'], 3 if s['metric'] != 'context_tokens_at_4' else 0)} "
         f"[{vn(s['diff_ci_low'], 3 if s['metric'] != 'context_tokens_at_4' else 0)}; "
         f"{vn(s['diff_ci_high'], 3 if s['metric'] != 'context_tokens_at_4' else 0)}]",
         f"{s['better']}/{s['worse']}/{s['same']}", vn(s["p_value"]), vn(s["p_holm"])]
        for s in stats_rows], "llrrrrrr"))
    out += ["", "Với Token ngữ cảnh, \"tốt hơn\" nghĩa là nhiều token hơn (tức tốn hơn).", "",
            "## 7. Theo tập dev/heldout (hybrid)", ""]
    split_rows = []
    for split in ("dev", "heldout"):
        chosen = [r["id"] for r in per_query if r["config"] == "A" and r["mode"] == "hybrid" and r["split"] == split]
        split_rows.append([f"{split} (n={len(chosen)})"] + [
            f"{vn(metric_matrix(per_query, k, 'hybrid', 'hit_at_1', chosen).mean())} / "
            f"{vn(metric_matrix(per_query, k, 'hybrid', 'recall_at_4', chosen).mean())} / "
            f"{vn(metric_matrix(per_query, k, 'hybrid', 'rr_at_10', chosen).mean())}" for k in keys])
    out.append(markdown_table(["Tập"] + [f"{names[k]}: Hit@1 / Recall@4 / MRR" for k in keys], split_rows))
    out += ["", "## 8. Theo loại câu hỏi (hybrid, Hit@1 / Recall@4)", ""]
    categories = sorted({r["category"] for r in per_query if r["mode"] == "hybrid"})
    category_rows = []
    for category in categories:
        chosen = [r["id"] for r in per_query if r["config"] == "A" and r["mode"] == "hybrid" and r["category"] == category]
        category_rows.append([f"{category} (n={len(chosen)})"] + [
            f"{vn(metric_matrix(per_query, k, 'hybrid', 'hit_at_1', chosen).mean(), 2)} / "
            f"{vn(metric_matrix(per_query, k, 'hybrid', 'recall_at_4', chosen).mean(), 2)}" for k in keys])
    out.append(markdown_table(["Loại"] + [names[k] for k in keys], category_rows))
    out += ["", "Mỗi loại chỉ có vài câu: dùng để tìm điểm yếu, không để kết luận.", "",
            "## 9. Ảnh hưởng của phương pháp tìm (k = 4)", ""]
    out.append(markdown_table(["Mode", "Độ đo"] + [names[k] for k in keys], [
        [mode, name] + [vn(aggregates[(k, mode)][metric]) for k in keys]
        for mode in MODES for metric, name in (("hit_at_1", "Hit@1"), ("recall_at_4", "Recall@4"),
                                               ("rr_at_10", "MRR@10"), ("ndcg_at_10", "nDCG@10"))], "llrrr"))
    out += ["", "## 10. Định nghĩa độ đo", "", DEFINITIONS]
    return "\n".join(out) + "\n"


DEFINITIONS = """- **Bằng chứng**: đoạn trích nguyên văn trong tài liệu, gán nhãn cho từng câu hỏi. Một chunk *chứa trọn*
  bằng chứng khi mọi ký tự khác khoảng trắng của đoạn đó nằm trong khoảng [start, end) của chunk.
- **Hit@1**: chunk xếp hạng 1 chứa trọn ít nhất một bằng chứng.
- **Recall@k**: tỉ lệ bằng chứng được phủ trọn bởi hợp của k chunk đầu (một bằng chứng có thể nằm vắt qua hai chunk).
- **Coverage@k**: như Recall@k nhưng tính điểm từng phần theo tỉ lệ ký tự được phủ.
- **MRR@10**: nghịch đảo thứ hạng của chunk đầu tiên chứa trọn một bằng chứng (0 nếu không có trong top 10).
- **nDCG@k**: độ lợi của một chunk = tỉ lệ bằng chứng mà riêng chunk đó phủ; chuẩn hóa theo thứ tự lý tưởng
  của mọi chunk trong index.
- **Context precision@4**: tỉ lệ chunk trong top-4 chạm vào bằng chứng hoặc ngữ cảnh bắt buộc. Thiên vị chunk
  to: chunk càng dài càng dễ chạm bằng chứng, và bằng chứng vắt qua hai chunk được đếm hai lần.
- **Mật độ bằng chứng@4**: tỉ lệ ký tự trong thân top-4 chunk là bằng chứng/ngữ cảnh đã gán nhãn (precision
  mức ký tự); phần chữ thừa bị tính vào mẫu số nên không thiên vị chunk to.
- **Doc Hit@1 / Doc Recall@4 / Doc precision@4**: như trên nhưng ở mức tài liệu (đúng file nguồn).
- **Recall@N token**: lấy chunk theo thứ hạng cho tới khi tổng token vượt N; đo recall trên phần đã lấy.
- **Chunk đúng tài liệu nhưng sai mục**: chunk thuộc tài liệu đúng nhưng không chạm bằng chứng/ngữ cảnh.
- **Required context recall@4**: tỉ lệ đoạn ngữ cảnh bắt buộc (điều kiện, ngoại lệ) được phủ trọn trong top-4.
- **Hệ số lặp**: tổng token thân chunk / tổng token corpus; > 1 do overlap, < 1 khi dòng heading và trường
  metadata nằm trong header thay vì thân chunk.
- **Phủ corpus / phủ nội dung**: tỉ lệ ký tự khác khoảng trắng nằm trong thân ít nhất một chunk; bản "nội dung"
  không tính dòng heading và thân các trường metadata (structure đưa chúng vào header/metadata của chunk).
- **Không cắt ngang câu**: chunk bắt đầu ở đầu dòng hoặc sau dấu kết câu, và kết thúc ở cuối dòng hoặc sau dấu kết câu.
- **Nằm trong 1 mục H2 / mục lá**: phần thân chunk chỉ chạm một mục `##` / một mục nhỏ nhất theo heading.
"""


# --- Chạy -----------------------------------------------------------------------------

def build_or_reuse(config: Config, index_dir: Path, reuse: bool, fingerprint: str) -> float | None:
    timing_file = index_dir / "build_seconds.json"
    manifest = read_manifest(index_dir, COLLECTION_NAME) if index_dir.exists() else None
    if reuse and manifest and manifest["corpus_sha256"] == fingerprint and manifest["chunking"]["strategy"] == config.strategy:
        print(f"{config.key}: dùng lại {index_dir}", flush=True)
        return json.loads(timing_file.read_text(encoding="utf-8"))["seconds"] if timing_file.exists() else None
    print(f"{config.key}: build {config.strategy} từ {config.data_dir}", flush=True)
    started = perf_counter()
    build_index(config.data_dir, index_dir, strategy=config.strategy)
    seconds = perf_counter() - started
    timing_file.write_text(json.dumps({"seconds": seconds}), encoding="utf-8")
    return seconds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/2026-09-24-chunking-comparison")
    parser.add_argument("--index-root", type=Path, default=ROOT / "artifacts/chunking/compare-2026-09-24")
    parser.add_argument("--queries", type=Path, default=ROOT / "eval/chunking_queries.jsonl")
    parser.add_argument("--reuse-indexes", action="store_true")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", message="Token indices sequence length")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    rows_all = [json.loads(line) for line in args.queries.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows_all if row["answerable"]]
    ids = [row["id"] for row in rows]
    count = token_counter(load_tokenizer(EMBEDDING_MODEL))
    create_embeddings().embed_query("khởi động")  # nạp model trước khi bấm giờ build
    if get_chunking_settings(strategy="structure").tokenizer_model != EMBEDDING_MODEL:
        load_tokenizer(get_chunking_settings(strategy="structure").tokenizer_model)

    args.out.mkdir(parents=True, exist_ok=True)
    per_query, top_hits, index_rows, evidence, chunk_rows = [], [], [], [], []
    records_by_config: dict[str, list] = {}
    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git": git_revision(),
        "queries_total": len(rows_all), "queries_scored": len(rows),
        "queries_sha256": hashlib.sha256(args.queries.read_bytes()).hexdigest(),
        "splits": {s: sum(r["split"] == s for r in rows) for s in ("dev", "heldout")},
        "evidence_spans": sum(len(r["expected_evidence"]) for r in rows),
        "embedding": {"model": EMBEDDING_MODEL, "device": EMBEDDING_DEVICE},
        "chunking": {"recursive": {"chunk_size_chars": CHUNK_SIZE, "overlap_chars": CHUNK_OVERLAP},
                     "structure": {k: v for k, v in vars(get_chunking_settings(strategy="structure")).items()}},
        "retrieval": {k: v for k, v in vars(get_retrieval_settings(mode="hybrid", rerank=False)).items()},
        "versions": {"python": platform.python_version(), "langchain-core": version("langchain-core"),
                     "chromadb": version("chromadb"), "sentence-transformers": version("sentence-transformers")},
        "configs": {},
    }
    for config in CONFIGS:
        documents = load_documents(config.data_dir)
        fingerprint = corpus_fingerprint(documents)
        labels = resolve_evidence(rows, documents)
        index_dir = args.index_root / config.slug
        seconds = build_or_reuse(config, index_dir, args.reuse_indexes, fingerprint)
        store = _load_vector_store(index_dir, None, COLLECTION_NAME)
        records = chunk_records(config, store, documents, count)
        records_by_config[config.key] = records
        token_cache = {}
        raw = store.get(include=["documents"])
        for content in raw["documents"]:
            token_cache[content] = count(content)
        stats = index_stats(config, documents, records, count, seconds, directory_size_mb(index_dir))
        index_rows.append(stats)
        spans = evidence_rows(config, rows, labels, records)
        evidence.extend(spans)
        chunk_rows.extend({k: v for k, v in r.items() if k != "body_in_content"} for r in records)
        print(f"{config.key}: {stats['chunks']} chunk, đo retrieval...", flush=True)
        texts = {d.metadata["source"]: d.page_content for d in documents}
        query_scores, hits = run_retrieval(config, store, rows, labels, records, token_cache, texts)
        per_query.extend(query_scores)
        top_hits.extend(hits)
        summary["configs"][config.key] = {
            "label": config.label, "data_dir": config.data_dir.relative_to(ROOT).as_posix(),
            "strategy": config.strategy, "fingerprint": fingerprint, "index_dir": str(index_dir),
            "evidence_contained": statistics.fmean(s["contained"] for s in spans),
            "evidence_chunks_overlapping": statistics.fmean(s["chunks_overlapping"] for s in spans),
            "index": stats,
        }

    rng = np.random.default_rng(SEED)
    aggregates = {(c.key, mode): aggregate(per_query, c.key, mode, ids, rng) for c in CONFIGS for mode in MODES}
    stats_rows = significance(per_query, ids)
    summary["retrieval_results"] = {f"{c}/{m}": v for (c, m), v in aggregates.items()}
    summary["significance"] = stats_rows

    write_csv(args.out / "index_stats.csv", index_rows)
    write_csv(args.out / "retrieval_summary.csv", list(aggregates.values()))
    write_csv(args.out / "significance.csv", stats_rows)
    write_csv(args.out / "per_query.csv", per_query)
    write_csv(args.out / "evidence_containment.csv", evidence)
    write_csv(args.out / "chunks.csv", chunk_rows)
    (args.out / "top_hits.jsonl").write_text(
        "".join(json.dumps(h, ensure_ascii=False) + "\n" for h in top_hits), encoding="utf-8")
    summary["figures"] = figures(args.out, per_query, stats_rows, records_by_config, aggregates, ids)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
                                           encoding="utf-8")
    (args.out / "RESULTS.md").write_text(
        results_markdown(summary, index_rows, aggregates, per_query, stats_rows, evidence, ids), encoding="utf-8")
    print(f"Xong: {args.out}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
