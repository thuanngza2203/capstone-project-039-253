"""So sánh semantic search và hybrid search (semantic + BM25, gộp bằng RRF) cho báo cáo.

Cấu hình chính: dữ liệu mới + structure (cấu hình hệ thống đang dùng). Lặp lại trên dữ liệu
mới + recursive để xem kết luận có phụ thuộc cách chunk không. BM25 đơn lẻ chỉ để tham khảo
(giải thích hybrid lấy thêm gì so với semantic), không nằm trong phép so sánh chính.

Dùng chung độ đo và index với scripts/compare_chunking.py.

    python scripts/compare_retrieval_modes.py
    python scripts/compare_retrieval_modes.py --reuse-indexes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chunking import load_tokenizer, token_counter  # noqa: E402
from config import COLLECTION_NAME, EMBEDDING_DEVICE, EMBEDDING_MODEL, create_embeddings, get_retrieval_settings  # noqa: E402
from index_manifest import corpus_fingerprint  # noqa: E402
from rag import _load_vector_store, load_documents  # noqa: E402
from retrieval import search_store  # noqa: E402
from scripts.benchmark_chunking import coverage, resolve_evidence  # noqa: E402
from scripts.compare_chunking import (  # noqa: E402
    BOOTSTRAP, BUDGETS, CONFIGS, DEFINITIONS, K_VALUES, MAX_K, SEED, _Chunk, bootstrap_mean, build_or_reuse,
    git_revision, holm, markdown_table, mcnemar, metric_matrix, pct, relevance, score_query, version, vn,
    wilcoxon, write_csv,
)

MODES = ("semantic", "hybrid", "bm25")
MODE_LABEL = {"semantic": "Semantic", "hybrid": "Hybrid", "bm25": "BM25 (tham khảo)"}
CONFIG_KEYS = ("C", "B")  # C: cấu hình chính của hệ thống; B: kiểm tra độ vững
LATENCY_REPEATS = 5
MISSING_RANK = MAX_K + 1  # "không có trong top-20"

# Màu theo thực thể (đã chạy validate_palette.js: pass cả 3 ô, all-pairs).
COLOR = {"semantic": "#2a78d6", "hybrid": "#eb6834", "bm25": "#1baf7a"}
INK, INK_2, GRID, SURFACE, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#c9c8c3"

TESTED = (
    ("hit_at_1", "binary", "Hit@1"), ("doc_hit_at_1", "binary", "Doc Hit@1"),
    ("recall_at_4", "continuous", "Recall@4"), ("coverage_at_4", "continuous", "Coverage@4"),
    ("rr_at_10", "continuous", "MRR@10"), ("ndcg_at_10", "continuous", "nDCG@10"),
    ("evidence_density_at_4", "continuous", "Mật độ bằng chứng@4"),
    ("recall_budget_1024", "continuous", "Recall@1024 token"),
    ("latency_ms", "continuous", "Độ trễ (ms)"),
)
MAIN = (
    ("hit_at_1", "Hit@1"), ("recall_at_1", "Recall@1"), ("recall_at_4", "Recall@4"),
    ("recall_at_10", "Recall@10"), ("coverage_at_4", "Coverage@4"), ("rr_at_10", "MRR@10"),
    ("ndcg_at_4", "nDCG@4"), ("ndcg_at_10", "nDCG@10"), ("evidence_density_at_4", "Mật độ bằng chứng@4"),
    ("precision_at_4", "Context precision@4"), ("doc_hit_at_1", "Doc Hit@1"),
    ("doc_precision_at_4", "Doc precision@4"), ("recall_budget_1024", "Recall@1024 token"),
)


# Chỉ giữ định nghĩa của các độ đo có trong báo cáo này (bỏ các độ đo riêng của so sánh chunking).
_KEPT_TERMS = ("**Bằng chứng**", "**Hit@1**", "**Recall@k**", "**Coverage@k**", "**MRR@10**", "**nDCG@k**",
               "**Context precision@4**", "**Mật độ bằng chứng@4**", "**Doc Hit@1", "**Recall@N token**")
MODE_DEFINITIONS = "\n".join(
    "- " + bullet for bullet in ("\n" + DEFINITIONS.strip()).split("\n- ")[1:] if bullet.startswith(_KEPT_TERMS))


def branch_profile(result, evidence) -> dict[str, Any]:
    """Hybrid lấy top-4 từ nhánh nào, và chunk đúng đầu tiên đứng hạng mấy ở mỗi nhánh."""
    hits = result.hits
    top4 = hits[:4]
    first = next((hit for hit in hits if any(coverage(label, [hit.document]) >= 1 for label in evidence)), None)
    return {
        "top4_both": sum(h.semantic_rank is not None and h.bm25_rank is not None for h in top4),
        "top4_semantic_only": sum(h.semantic_rank is not None and h.bm25_rank is None for h in top4),
        "top4_bm25_only": sum(h.semantic_rank is None and h.bm25_rank is not None for h in top4),
        "correct_semantic_rank": first.semantic_rank if first else None,
        "correct_bm25_rank": first.bm25_rank if first else None,
    }


def measure(config, rows, count, reuse: bool, index_root: Path) -> tuple[list[dict], list[dict], dict]:
    documents = load_documents(config.data_dir)
    labels = resolve_evidence(rows, documents)
    fingerprint = corpus_fingerprint(documents)
    index_dir = index_root / config.slug
    build_or_reuse(config, index_dir, reuse, fingerprint)
    store = _load_vector_store(index_dir, None, COLLECTION_NAME)
    raw = store.get(include=["documents", "metadatas"])
    chunks = [_Chunk({"source": m["source"], "start": int(m["start_index"]), "end": int(m["end_index"])})
              for m in raw["metadatas"]]
    tokens = {content: count(content) for content in raw["documents"]}
    texts = {d.metadata["source"]: d.page_content for d in documents}

    per_query, top_hits = [], []
    for mode in MODES:
        settings = get_retrieval_settings(mode=mode, rerank=False)
        for _ in range(3):  # dựng BM25/cache và làm nóng GPU, không tính giờ
            search_store("khởi động", store, k=MAX_K, settings=settings)
        for row in rows:
            evidence = labels[row["id"]]["expected_evidence"]
            sources = {label.source for label in evidence}
            ideal = sorted((relevance(c, evidence) for c in chunks if c.metadata["source"] in sources), reverse=True)
            timings = []
            for _ in range(LATENCY_REPEATS):
                started = perf_counter()
                result = search_store(row["query"], store, k=MAX_K, settings=settings)
                timings.append((perf_counter() - started) * 1000)
            ranked = result.documents
            score = score_query(row, labels[row["id"]], ranked, lambda d: tokens[d.page_content], ideal, texts)
            record = {"config": config.key, "mode": mode, "id": row["id"], "split": row["split"],
                      "category": row["category"], "query": row["query"],
                      "latency_ms": statistics.median(timings), "latency_min_ms": min(timings), **score}
            if mode == "hybrid":
                record.update(branch_profile(result, evidence))
            per_query.append(record)
            if config.key == "C" and mode in {"semantic", "hybrid"}:
                top_hits.append({"config": config.key, "mode": mode, "id": row["id"], "query": row["query"], "hits": [
                    {"rank": rank, "source": h.document.metadata["source"],
                     "heading_path": h.document.metadata.get("heading_path"),
                     "semantic_rank": h.semantic_rank, "bm25_rank": h.bm25_rank,
                     "relevance": round(relevance(h.document, evidence), 3), "preview": h.document.page_content[:240]}
                    for rank, h in enumerate(result.hits[:4], start=1)]})
    info = {"label": config.label, "strategy": config.strategy, "data_dir": config.data_dir.relative_to(ROOT).as_posix(),
            "fingerprint": fingerprint, "chunks": len(chunks), "index_dir": str(index_dir)}
    return per_query, top_hits, info


def rank(value) -> int:
    return MISSING_RANK if value in (None, "") else int(value)


def query_changes(per_query, config: str, ids) -> list[dict[str, Any]]:
    lookup = {(r["mode"], r["id"]): r for r in per_query if r["config"] == config}
    rows = []
    for qid in ids:
        semantic, hybrid, bm25 = (lookup[(m, qid)] for m in ("semantic", "hybrid", "bm25"))
        s, h = rank(semantic["first_full_rank"]), rank(hybrid["first_full_rank"])
        if s <= 4 < h:
            change = "hybrid làm mất (semantic có trong top-4, hybrid không)"
        elif h <= 4 < s:
            change = "hybrid cứu (semantic không có trong top-4, hybrid có)"
        elif h < s:
            change = "hybrid xếp cao hơn"
        elif h > s:
            change = "hybrid xếp thấp hơn"
        else:
            change = "như nhau"
        rows.append({"config": config, "id": qid, "category": semantic["category"], "query": semantic["query"],
                     "semantic_rank": None if s == MISSING_RANK else s, "hybrid_rank": None if h == MISSING_RANK else h,
                     "bm25_rank": None if rank(bm25["first_full_rank"]) == MISSING_RANK else rank(bm25["first_full_rank"]),
                     "recall_at_4_semantic": semantic["recall_at_4"], "recall_at_4_hybrid": hybrid["recall_at_4"],
                     "change": change})
    return rows


def significance(per_query, ids) -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED)
    output = []
    for config in CONFIG_KEYS:
        tests = []
        for metric, kind, label in TESTED:
            x = metric_matrix(per_query, config, "semantic", metric, ids)
            y = metric_matrix(per_query, config, "hybrid", metric, ids)
            diff = y - x
            low, high = bootstrap_mean(diff, rng)
            test = {"config": config, "metric": metric, "label": label, "semantic": float(x.mean()),
                    "hybrid": float(y.mean()), "diff": float(diff.mean()), "diff_ci_low": low, "diff_ci_high": high,
                    "hybrid_higher": int((diff > 0).sum()), "hybrid_lower": int((diff < 0).sum()),
                    "same": int((diff == 0).sum())}
            if kind == "binary":
                only_semantic, only_hybrid, p = mcnemar(x, y)
                test.update(test_name="McNemar chính xác", p_value=p, only_semantic=only_semantic, only_hybrid=only_hybrid)
            else:
                test.update(test_name="Wilcoxon signed-rank", p_value=wilcoxon(x, y))
            tests.append(test)
        # Một họ kiểm định cho mỗi cấu hình: "hybrid có khác semantic không" trên 9 độ đo.
        for test, adjusted in zip(tests, holm([t["p_value"] for t in tests])):
            test["p_holm"] = adjusted
        output.extend(tests)
    return output


def aggregate(per_query, config, mode, ids, rng) -> dict[str, Any]:
    result = {"config": config, "mode": mode, "queries": len(ids)}
    for metric, _ in MAIN + (("context_tokens_at_4", ""),):
        values = metric_matrix(per_query, config, mode, metric, ids)
        result[metric] = float(values.mean())
        result[f"{metric}_ci"] = bootstrap_mean(values, rng)
    latency = metric_matrix(per_query, config, mode, "latency_ms", ids)
    result.update(latency_ms_median=float(np.median(latency)), latency_ms_p95=float(np.percentile(latency, 95)),
                  latency_ms_mean=float(latency.mean()))
    return result


# --- Biểu đồ ---------------------------------------------------------------------------

def figures(out: Path, per_query, aggregates, changes, ids) -> list[str]:
    try:
        import matplotlib
    except ImportError:
        print("Bỏ qua biểu đồ: chưa cài matplotlib.", flush=True)
        return []
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": NEUTRAL, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "font.size": 9.5, "legend.frameon": False, "lines.linewidth": 2,
    })
    folder = out / "figures"
    folder.mkdir(exist_ok=True)
    written = []
    config_title = {"C": "Dữ liệu mới + structure (chính)", "B": "Dữ liệu mới + recursive"}

    def save(fig, name):
        fig.savefig(folder / name, dpi=200, bbox_inches="tight")
        plt.close(fig)
        written.append(f"figures/{name}")

    def curve(ax, xs, config, metric_of, title, xlabel):
        ends = []
        for mode in MODES:
            ys = [metric_matrix(per_query, config, mode, metric_of(x), ids).mean() for x in xs]
            dashed = mode == "bm25"
            ax.plot(xs, ys, color=COLOR[mode], linestyle="--" if dashed else "-", linewidth=1.5 if dashed else 2,
                    marker="o", markersize=5, label=MODE_LABEL[mode], zorder=3)
            ends.append([ys[-1], MODE_LABEL[mode].split(" ")[0]])
        # Nhãn cuối đường: giãn ra khi các giá trị cuối gần nhau (ví dụ cùng 0,940) để không đè chữ.
        ends.sort()
        for lower, upper in zip(ends, ends[1:]):
            upper[0] = max(upper[0], lower[0] + 0.045)
        span = xs[-1] - xs[0]
        for y, name in ends:
            ax.text(xs[-1] + span * 0.03, y, name, va="center", fontsize=8.5, color=INK_2)
        ax.set(title=title, xlabel=xlabel, ylim=(0, 1.08))
        ax.grid(axis="x", visible=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    for ax, config in zip(axes, CONFIG_KEYS):
        curve(ax, K_VALUES, config, lambda k: f"recall_at_{k}", config_title[config], "Số chunk lấy về (k)")
    axes[0].set_ylabel("Recall@k")
    axes[0].legend(loc="lower right")
    fig.suptitle("Recall theo số chunk", x=0.02, ha="left", fontsize=11)
    save(fig, "recall_at_k.png")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    for ax, config in zip(axes, CONFIG_KEYS):
        curve(ax, BUDGETS, config, lambda b: f"recall_budget_{b}", config_title[config], "Ngân sách ngữ cảnh (token)")
    axes[0].set_ylabel("Recall")
    axes[0].legend(loc="lower right")
    fig.suptitle("Recall theo ngân sách token ngữ cảnh", x=0.02, ha="left", fontsize=11)
    save(fig, "recall_at_budget.png")

    shown = (("hit_at_1", "Hit@1"), ("recall_at_4", "Recall@4"), ("rr_at_10", "MRR@10"),
             ("ndcg_at_10", "nDCG@10"), ("recall_budget_1024", "Recall@\n1024 token"), ("doc_hit_at_1", "Doc Hit@1"))
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    width = 0.36
    for offset, mode in enumerate(("semantic", "hybrid")):
        agg = aggregates[("C", mode)]
        means = np.array([agg[m] for m, _ in shown])
        errors = np.array([[agg[m] - agg[f"{m}_ci"][0] for m, _ in shown], [agg[f"{m}_ci"][1] - agg[m] for m, _ in shown]])
        x = np.arange(len(shown)) + (offset - 0.5) * width
        ax.bar(x, means, width - 0.04, color=COLOR[mode], label=MODE_LABEL[mode], zorder=2)
        ax.errorbar(x, means, yerr=errors, fmt="none", ecolor=INK_2, elinewidth=1, capsize=2, zorder=3)
        for xi, value in zip(x, means):
            ax.text(xi, 0.02, f"{value:.2f}".replace(".", ","), ha="center", va="bottom", fontsize=7.5, color=SURFACE)
    ax.set_xticks(np.arange(len(shown)), [name for _, name in shown])
    ax.set(ylim=(0, 1.05), title="Độ đo chính, dữ liệu mới + structure (khoảng tin cậy 95% bootstrap)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    save(fig, "main_metrics.png")

    rows = [r for r in changes if r["config"] == "C"]
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    rng = np.random.default_rng(1)
    # "Không có trong top-20" vẽ ở 23 để tách khỏi mốc 20.
    shown_missing = MISSING_RANK + 2
    s = np.array([rank(r["semantic_rank"]) for r in rows], dtype=float)
    h = np.array([rank(r["hybrid_rank"]) for r in rows], dtype=float)
    s[s == MISSING_RANK], h[h == MISSING_RANK] = shown_missing, shown_missing
    jitter = lambda n: rng.uniform(-0.18, 0.18, n)
    better, worse = h < s, h > s
    for mask, color, label in ((better, COLOR["hybrid"], "hybrid xếp cao hơn"), (worse, COLOR["semantic"], "semantic xếp cao hơn"),
                               (~better & ~worse, NEUTRAL, "như nhau")):
        ax.scatter(s[mask] + jitter(mask.sum()), h[mask] + jitter(mask.sum()), s=34, color=color,
                   edgecolor=SURFACE, linewidth=1, label=f"{label} ({int(mask.sum())})", zorder=3)
    ax.plot([0.5, shown_missing + 0.5], [0.5, shown_missing + 0.5], color=INK_2, linewidth=1, linestyle=":")
    ticks = [1, 2, 3, 4, 5, 10, 15, 20, shown_missing]
    names = [str(t) for t in ticks[:-1]] + ["không\ncó"]
    ax.set_xticks(ticks, names)
    ax.set_yticks(ticks, names)
    ax.set(xlim=(0.4, shown_missing + 0.6), ylim=(0.4, shown_missing + 0.6),
           xlabel="Hạng chunk đúng đầu tiên: semantic", ylabel="Hạng chunk đúng đầu tiên: hybrid",
           title="Thứ hạng từng câu (dữ liệu mới + structure)")
    ax.text(shown_missing - 0.3, 1.2, "dưới đường chéo:\nhybrid xếp cao hơn", ha="right", va="bottom",
            fontsize=8, color=INK_2)
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "rank_shift.png")

    categories = sorted({r["category"] for r in per_query if r["config"] == "C"})
    deltas, counts = [], []
    for category in categories:
        chosen = [r["id"] for r in per_query if r["config"] == "C" and r["mode"] == "hybrid" and r["category"] == category]
        deltas.append(metric_matrix(per_query, "C", "hybrid", "recall_at_4", chosen).mean()
                      - metric_matrix(per_query, "C", "semantic", "recall_at_4", chosen).mean())
        counts.append(len(chosen))
    order = np.argsort(deltas)
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    for position, index in enumerate(order):
        delta = deltas[index]
        color = COLOR["hybrid"] if delta > 0 else COLOR["semantic"] if delta < 0 else NEUTRAL
        ax.barh(position, delta if delta else 0.004, height=0.62, color=color, zorder=2)
        ax.text(delta + (0.01 if delta >= 0 else -0.01), position, f"{delta:+.2f}".replace(".", ","),
                va="center", ha="left" if delta >= 0 else "right", fontsize=8, color=INK_2)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks(range(len(order)), [f"{categories[i]} (n={counts[i]})" for i in order], fontsize=8.5)
    limit = max(0.3, max(abs(d) for d in deltas) + 0.12)
    ax.set(xlim=(-limit, limit), xlabel="Recall@4 hybrid − semantic",
           title="Chênh lệch theo loại câu hỏi (dữ liệu mới + structure)")
    ax.grid(axis="y", visible=False)
    ax.text(0.99, -0.13, "cam: hybrid tốt hơn · xanh: semantic tốt hơn", transform=ax.transAxes,
            ha="right", fontsize=8, color=INK_2)
    save(fig, "category_delta.png")

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    data = [metric_matrix(per_query, "C", mode, "latency_ms", ids) for mode in MODES]
    box = ax.boxplot(data, orientation="horizontal", widths=0.5, patch_artist=True, showfliers=True,
                     medianprops={"color": INK, "linewidth": 1.5}, flierprops={"markersize": 3, "markeredgecolor": INK_2})
    for patch, mode in zip(box["boxes"], MODES):
        patch.set(facecolor=COLOR[mode], alpha=0.85, edgecolor=SURFACE)
    ax.set_yticks([1, 2, 3], [MODE_LABEL[m] for m in MODES])
    for position, values in enumerate(data, start=1):
        ax.text(np.median(values), position + 0.34, f"trung vị {np.median(values):.0f} ms", ha="center", fontsize=8, color=INK_2)
    ax.set(xlabel="ms mỗi truy vấn (trung vị của 5 lần chạy)", title="Độ trễ retrieval (dữ liệu mới + structure)")
    ax.grid(axis="y", visible=False)
    save(fig, "latency.png")
    return written


# --- Báo cáo markdown ------------------------------------------------------------------

def results_markdown(summary, aggregates, stats_rows, changes, per_query, ids) -> str:
    names = {"C": "C. Dữ liệu mới + structure", "B": "B. Dữ liệu mới + recursive"}
    out = [
        "# Kết quả so sánh semantic và hybrid search (tự sinh)", "",
        f"Sinh bởi `scripts/compare_retrieval_modes.py` lúc {summary['generated_at']}. Chạy lại script sẽ ghi đè; "
        "phần nhận xét nằm ở `README.md`.", "",
        "## 1. Thiết lập", "",
        markdown_table(["Cấu hình", "Dữ liệu", "Chunking", "Số chunk", "Vai trò"], [
            [names[k], summary["configs"][k]["data_dir"], summary["configs"][k]["strategy"],
             str(summary["configs"][k]["chunks"]), "chính (hệ thống đang dùng)" if k == "C" else "kiểm tra độ vững"]
            for k in CONFIG_KEYS], "lllrl"),
        "",
        "- **Semantic**: tìm theo embedding `AITeamVN/Vietnamese_Embedding`, khoảng cách cosine.",
        f"- **Hybrid**: semantic + BM25, mỗi nhánh lấy {summary['retrieval']['candidate_k']} ứng viên, gộp bằng "
        f"Reciprocal Rank Fusion (k = {summary['retrieval']['rrf_k']}). Không reranker.",
        "- **BM25** chạy riêng chỉ để tham khảo: giải thích hybrid lấy thêm gì so với semantic.",
        f"- Bộ eval: `eval/chunking_queries.jsonl`, {len(ids)} câu có nhãn bằng chứng "
        f"({summary['splits']['dev']} dev, {summary['splits']['heldout']} heldout). Hệ thống dùng k = 4.",
        f"- Độ trễ: trung vị của {LATENCY_REPEATS} lần chạy mỗi câu, sau khi làm nóng; embedding trên {EMBEDDING_DEVICE}.",
        f"- Khoảng tin cậy 95%: bootstrap {BOOTSTRAP:,} lần theo câu (seed {SEED}).".replace(",", "."),
        f"- Git `{summary['git']}`; Python {summary['versions']['python']}, chromadb {summary['versions']['chromadb']}.",
        "",
    ]
    for number, config in enumerate(CONFIG_KEYS, start=2):
        out += [f"## {number}. Kết quả chính: {names[config]} (k = 4)", ""]
        out.append(markdown_table(["Độ đo", "Semantic", "Hybrid", "BM25 (tham khảo)"], [
            [label] + [f"{vn(aggregates[(config, m)][metric])} [{vn(aggregates[(config, m)][metric + '_ci'][0])}–"
                       f"{vn(aggregates[(config, m)][metric + '_ci'][1])}]" if m != "bm25"
                       else vn(aggregates[(config, m)][metric]) for m in MODES]
            for metric, label in MAIN]))
        out.append("")
        out.append(markdown_table(["Chi phí", "Semantic", "Hybrid", "BM25 (tham khảo)"], [
            ["Độ trễ trung vị (ms)"] + [vn(aggregates[(config, m)]["latency_ms_median"], 1) for m in MODES],
            ["Độ trễ P95 (ms)"] + [vn(aggregates[(config, m)]["latency_ms_p95"], 1) for m in MODES],
            ["Token ngữ cảnh top-4 (TB)"] + [vn(aggregates[(config, m)]["context_tokens_at_4"], 0) for m in MODES],
        ]))
        out += ["", "Số trong ngoặc vuông là khoảng tin cậy 95%.", ""]
    out += ["## 4. Kiểm định hybrid so với semantic", "",
            "Nhị phân: McNemar chính xác; liên tục: Wilcoxon signed-rank. Chênh lệch = hybrid − semantic, kèm KTC 95% "
            "bootstrap. p Holm hiệu chỉnh trên 9 độ đo của cùng một cấu hình. Với độ trễ, chênh lệch dương = hybrid chậm hơn.", ""]
    out.append(markdown_table(["Cấu hình", "Độ đo", "Semantic", "Hybrid", "Chênh lệch [KTC 95%]",
                               "Hybrid cao hơn/thấp hơn/bằng", "p", "p Holm"], [
        [s["config"], s["label"], vn(s["semantic"], 1 if s["metric"] == "latency_ms" else 3),
         vn(s["hybrid"], 1 if s["metric"] == "latency_ms" else 3),
         f"{vn(s['diff'], 1 if s['metric'] == 'latency_ms' else 3)} [{vn(s['diff_ci_low'], 1 if s['metric'] == 'latency_ms' else 3)}; "
         f"{vn(s['diff_ci_high'], 1 if s['metric'] == 'latency_ms' else 3)}]",
         f"{s['hybrid_higher']}/{s['hybrid_lower']}/{s['same']}", vn(s["p_value"]), vn(s["p_holm"])]
        for s in stats_rows], "llrrrrrr"))
    out += ["", "## 5. Theo k và theo ngân sách token (dữ liệu mới + structure)", ""]
    out.append(markdown_table(["k"] + [f"Recall {MODE_LABEL[m]}" for m in MODES], [
        [str(k)] + [vn(metric_matrix(per_query, "C", m, f"recall_at_{k}", ids).mean()) for m in MODES] for k in K_VALUES]))
    out.append("")
    out.append(markdown_table(["Ngân sách token"] + [f"Recall {MODE_LABEL[m]}" for m in MODES], [
        [vn(b, 0)] + [vn(metric_matrix(per_query, "C", m, f"recall_budget_{b}", ids).mean()) for m in MODES] for b in BUDGETS]))
    out += ["", "## 6. Hybrid lấy kết quả từ nhánh nào (dữ liệu mới + structure)", ""]
    hybrid = [r for r in per_query if r["config"] == "C" and r["mode"] == "hybrid"]
    both = statistics.fmean(r["top4_both"] for r in hybrid)
    semantic_only = statistics.fmean(r["top4_semantic_only"] for r in hybrid)
    bm25_only = statistics.fmean(r["top4_bm25_only"] for r in hybrid)
    found = [r for r in hybrid if r["first_full_rank"] is not None and r["first_full_rank"] <= 4]
    only_bm25_found = sum(1 for r in found if r["correct_semantic_rank"] is None)
    only_semantic_found = sum(1 for r in found if r["correct_bm25_rank"] is None)
    out.append(markdown_table(["Chỉ số", "Giá trị"], [
        ["Trong top-4 của hybrid: chunk có ở cả hai nhánh (TB / 4)", vn(both, 2)],
        ["Chỉ có ở nhánh semantic (TB / 4)", vn(semantic_only, 2)],
        ["Chỉ có ở nhánh BM25 (TB / 4)", vn(bm25_only, 2)],
        [f"Câu hybrid tìm được chunk đúng trong top-4", f"{len(found)}/{len(ids)}"],
        ["… trong đó chunk đúng không có trong top-20 của semantic (nhờ BM25)", str(only_bm25_found)],
        ["… trong đó chunk đúng không có trong top-20 của BM25 (nhờ semantic)", str(only_semantic_found)],
    ], "lr"))
    out += ["", "## 7. Thay đổi từng câu (dữ liệu mới + structure)", "",
            "Hạng = thứ hạng của chunk đầu tiên chứa trọn bằng chứng; trống = không có trong top-20.", ""]
    rows = [r for r in changes if r["config"] == "C"]
    counts = {}
    for r in rows:
        counts[r["change"]] = counts.get(r["change"], 0) + 1
    out.append(markdown_table(["Thay đổi", "Số câu"], [[k, str(v)] for k, v in sorted(counts.items(), key=lambda kv: -kv[1])], "lr"))
    out.append("")
    changed = [r for r in rows if r["change"] != "như nhau"]
    out.append(markdown_table(["Câu", "Loại", "Nội dung", "Hạng semantic", "Hạng hybrid", "Hạng BM25", "Thay đổi"], [
        [r["id"], r["category"], r["query"], str(r["semantic_rank"] or "—"), str(r["hybrid_rank"] or "—"),
         str(r["bm25_rank"] or "—"), r["change"]] for r in sorted(changed, key=lambda r: r["change"])], "lllrrrl"))
    out += ["", "## 8. Theo loại câu hỏi (dữ liệu mới + structure, Hit@1 / Recall@4)", ""]
    categories = sorted({r["category"] for r in per_query if r["config"] == "C"})
    table = []
    for category in categories:
        chosen = [r["id"] for r in hybrid if r["category"] == category]
        table.append([f"{category} (n={len(chosen)})"] + [
            f"{vn(metric_matrix(per_query, 'C', m, 'hit_at_1', chosen).mean(), 2)} / "
            f"{vn(metric_matrix(per_query, 'C', m, 'recall_at_4', chosen).mean(), 2)}" for m in MODES])
    out.append(markdown_table(["Loại"] + [MODE_LABEL[m] for m in MODES], table))
    out += ["", "## 9. Theo tập dev/heldout (Hit@1 / Recall@4 / MRR@10)", ""]
    table = []
    for config in CONFIG_KEYS:
        for split in ("dev", "heldout"):
            chosen = [r["id"] for r in per_query if r["config"] == config and r["mode"] == "hybrid" and r["split"] == split]
            table.append([f"{config} · {split} (n={len(chosen)})"] + [
                " / ".join(vn(metric_matrix(per_query, config, m, metric, chosen).mean()) for metric in ("hit_at_1", "recall_at_4", "rr_at_10"))
                for m in ("semantic", "hybrid")])
    out.append(markdown_table(["Tập", "Semantic", "Hybrid"], table))
    out += ["", "## 10. Định nghĩa độ đo", "", MODE_DEFINITIONS,
            "- **Độ trễ**: thời gian gọi `search_store` (embedding câu hỏi + tìm + gộp), trung vị 5 lần; không gồm LLM.",
            "- **Hạng chunk đúng đầu tiên**: thứ hạng nhỏ nhất của chunk chứa trọn một bằng chứng; dùng cho MRR và mục 7."]
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/2026-09-24-retrieval-semantic-vs-hybrid")
    parser.add_argument("--index-root", type=Path, default=ROOT / "artifacts/chunking/compare-2026-09-24")
    parser.add_argument("--queries", type=Path, default=ROOT / "eval/chunking_queries.jsonl")
    parser.add_argument("--reuse-indexes", action="store_true")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", message="Token indices sequence length")

    rows_all = [json.loads(line) for line in args.queries.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows_all if row["answerable"]]
    ids = [row["id"] for row in rows]
    count = token_counter(load_tokenizer(EMBEDDING_MODEL))
    create_embeddings().embed_query("khởi động")
    args.out.mkdir(parents=True, exist_ok=True)

    per_query, top_hits, infos = [], [], {}
    for config in (c for c in CONFIGS if c.key in CONFIG_KEYS):
        print(f"{config.key}: đo {', '.join(MODES)}...", flush=True)
        scores, hits, info = measure(config, rows, count, args.reuse_indexes, args.index_root)
        per_query.extend(scores)
        top_hits.extend(hits)
        infos[config.key] = info

    rng = np.random.default_rng(SEED)
    aggregates = {(c, m): aggregate(per_query, c, m, ids, rng) for c in CONFIG_KEYS for m in MODES}
    stats_rows = significance(per_query, ids)
    changes = [row for config in CONFIG_KEYS for row in query_changes(per_query, config, ids)]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git": git_revision(),
        "queries_scored": len(ids), "queries_sha256": hashlib.sha256(args.queries.read_bytes()).hexdigest(),
        "splits": {s: sum(r["split"] == s for r in rows) for s in ("dev", "heldout")},
        "embedding": {"model": EMBEDDING_MODEL, "device": EMBEDDING_DEVICE},
        "retrieval": {k: v for k, v in vars(get_retrieval_settings(mode="hybrid", rerank=False)).items()},
        "latency_repeats": LATENCY_REPEATS,
        "versions": {"python": platform.python_version(), "chromadb": version("chromadb"),
                     "langchain-core": version("langchain-core")},
        "configs": infos,
        "results": {f"{c}/{m}": v for (c, m), v in aggregates.items()},
        "significance": stats_rows,
    }
    write_csv(args.out / "retrieval_summary.csv", list(aggregates.values()))
    write_csv(args.out / "significance.csv", stats_rows)
    write_csv(args.out / "per_query.csv", per_query)
    write_csv(args.out / "query_changes.csv", changes)
    (args.out / "top_hits.jsonl").write_text("".join(json.dumps(h, ensure_ascii=False) + "\n" for h in top_hits),
                                             encoding="utf-8")
    summary["figures"] = figures(args.out, per_query, aggregates, changes, ids)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
                                           encoding="utf-8")
    (args.out / "RESULTS.md").write_text(results_markdown(summary, aggregates, stats_rows, changes, per_query, ids),
                                         encoding="utf-8")
    print(f"Xong: {args.out}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
