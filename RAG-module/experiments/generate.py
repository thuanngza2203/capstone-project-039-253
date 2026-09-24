"""Chạy một bộ câu hỏi qua RAG API và ghi câu trả lời của một variant.

Một variant = một cấu hình cố định: index (cách chunk) + LLM (model đang host) +
tham số retrieval. Mỗi variant ghi vào `runs/<variant>.jsonl` và `runs/<variant>.run.json`.

    python -m experiments.generate --questions experiments/questions.jsonl \\
        --variant qwen4b-structure --index structure --llm-provider vllm

Chạy lại cùng lệnh thì chỉ chạy các câu chưa có kết quả hoặc đã lỗi. Công cụ dừng nếu
model thật (sau alias `rag-llm`), index hay tham số retrieval khác với lần chạy trước.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx

EXPERIMENTS_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = EXPERIMENTS_DIR / "runs"
VARIANT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
# Trường của câu hỏi được chuyển thẳng vào /v1/answer; trường khác (topic, note...) chỉ để ghi chú.
FORWARDED_FIELDS = ("plant_type", "disease", "retrieval_query", "subject_context")
RETRYABLE_STATUS = frozenset({502, 503, 504})
# Các trường trong meta phải giữ nguyên suốt một variant.
LOCKED_SETTINGS = ("index", "chunking_strategy", "retrieval_mode", "reranker_enabled", "top_k")


class ExperimentError(RuntimeError):
    """Lỗi phải dừng cả lượt chạy (cấu hình lệch, index cũ, LLM không tới được)."""


@dataclass(frozen=True)
class RunConfig:
    variant: str
    index: str | None = None
    llm_provider: str | None = None
    top_k: int | None = None
    mode: str | None = None
    rerank: bool | None = None


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_questions(path: Path) -> list[dict[str, Any]]:
    questions, seen = [], set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExperimentError(f"{path}:{number}: không phải JSON ({exc.msg}).") from exc
        qid, question = str(item.get("id", "")).strip(), str(item.get("question", "")).strip()
        if not qid or not question:
            raise ExperimentError(f"{path}:{number}: cần có `id` và `question`.")
        if qid in seen:
            raise ExperimentError(f"{path}:{number}: trùng id {qid!r}.")
        seen.add(qid)
        questions.append({**item, "id": qid, "question": question})
    if not questions:
        raise ExperimentError(f"{path}: không có câu hỏi nào.")
    return questions


def load_rows(path: Path) -> dict[str, dict[str, Any]]:
    """Dòng sau cùng của mỗi câu thắng: lần chạy lại một câu lỗi ghi đè lỗi cũ."""
    rows: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["question_id"]] = row
    return rows


def model_identity(info: dict[str, Any]) -> str:
    """Tên model thật: vLLM lấy `root` sau alias, Ollama kèm digest, Gemini lấy tên."""
    served = info.get("served") or []
    if info["provider"] == "vllm":
        match = next((model for model in served if model["id"] == info.get("model")), None)
        if match and match.get("root"):
            return match["root"]
    if info["provider"] == "ollama" and served:
        entry = served[0]
        return f"{entry['id']}@{entry['digest']}" if entry.get("digest") else entry["id"]
    return info.get("model") or "không rõ"


class Runner:
    def __init__(
        self, client: httpx.Client, config: RunConfig, out_dir: Path, *,
        allow_stale: bool = False, retries: int = 2, backoff_seconds: float = 3.0,
        log: Callable[[str], None] = print,
    ) -> None:
        if not VARIANT_PATTERN.match(config.variant):
            raise ExperimentError("--variant chỉ gồm chữ, số, '.', '_', '-' (tối đa 64 ký tự).")
        self.client, self.config, self.log = client, config, log
        self.allow_stale, self.retries, self.backoff = allow_stale, retries, backoff_seconds
        self.rows_path = out_dir / f"{config.variant}.jsonl"
        self.run_path = out_dir / f"{config.variant}.run.json"
        out_dir.mkdir(parents=True, exist_ok=True)

    # --- Kiểm tra trước khi chạy ------------------------------------------------

    def _get(self, path: str, **params: Any) -> dict[str, Any]:
        try:
            response = self.client.get(path, params=params or None)
        except httpx.HTTPError as exc:
            raise ExperimentError(f"Không gọi được RAG API ({path}): {exc}. Server đã chạy chưa?") from exc
        if response.status_code != 200:
            raise ExperimentError(f"RAG API {path} trả {response.status_code}: {response.text[:300]}")
        return response.json()

    def check_index(self) -> dict[str, Any]:
        status = self._get("/v1/status")
        name = self.config.index or status["default_index"]
        index = next(item for item in status["indexes"] if item["name"] == name)
        if index["matches_data"] is not True and not self.allow_stale:
            reason = index.get("detail") or "không xác minh được index build từ data/ hiện tại"
            raise ExperimentError(f"Index {name}: {reason}. Thêm --allow-stale nếu cố ý.")
        return {"name": name, **{k: index[k] for k in ("strategy", "chunk_count", "built_at", "matches_data")},
                "embedding_model": status["embedding_model"]}

    def check_llm(self) -> dict[str, Any]:
        params = {"probe": "true"}
        if self.config.llm_provider:
            params["provider"] = self.config.llm_provider
        info = self._get("/v1/llm", **params)
        if info["reachable"] is False:
            raise ExperimentError(f"LLM {info['provider']} không tới được: {info['detail']}")
        if info["provider"] == "vllm" and info.get("detail"):
            raise ExperimentError(f"vLLM: {info['detail']}")
        return {**info, "identity": model_identity(info)}

    def prepare(self, questions_path: Path, questions: list[dict[str, Any]]) -> dict[str, Any]:
        index, llm = self.check_index(), self.check_llm()
        fingerprint = {"index": index["name"], "llm_provider": llm["provider"], "model": llm["identity"]}
        run = json.loads(self.run_path.read_text(encoding="utf-8")) if self.run_path.exists() else None
        if run is None:
            run = {
                "variant": self.config.variant, "created_at": now(), **fingerprint,
                "request": {k: getattr(self.config, k) for k in ("top_k", "mode", "rerank")},
                "settings": None, "index_info": index, "llm_info": llm, "sessions": [],
            }
        else:
            changed = {k: (run[k], v) for k, v in fingerprint.items() if run[k] != v}
            request = {k: getattr(self.config, k) for k in ("top_k", "mode", "rerank")}
            changed.update({k: (run["request"][k], v) for k, v in request.items() if run["request"][k] != v})
            if changed:
                detail = "; ".join(f"{k}: {old!r} → {new!r}" for k, (old, new) in changed.items())
                raise ExperimentError(
                    f"Variant {self.config.variant!r} đã chạy với cấu hình khác ({detail}). "
                    "Đặt --variant mới để không trộn kết quả của hai cấu hình."
                )
        run["sessions"].append({
            "started_at": now(), "questions_file": str(questions_path),
            "questions_sha256": hashlib.sha256(questions_path.read_bytes()).hexdigest(),
            "question_count": len(questions),
        })
        self._write_run(run)
        return run

    def _write_run(self, run: dict[str, Any]) -> None:
        temporary = self.run_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.run_path)

    # --- Chạy -------------------------------------------------------------------

    def payload(self, question: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {"query": question["question"], "debug": True}
        body.update({k: question[k] for k in FORWARDED_FIELDS if question.get(k)})
        options = {"index": self.config.index, "llm_provider": self.config.llm_provider,
                   "top_k": self.config.top_k, "mode": self.config.mode, "rerank": self.config.rerank}
        body.update({k: v for k, v in options.items() if v is not None})
        return body

    def ask(self, question: dict[str, Any]) -> dict[str, Any]:
        body = self.payload(question)
        row: dict[str, Any] = {"question_id": question["id"], "variant": self.config.variant,
                               "question": question["question"], "request": body}
        error: dict[str, Any] | None = None
        for attempt in range(self.retries + 1):
            if attempt:
                time.sleep(self.backoff * attempt)
            started = time.perf_counter()
            try:
                response = self.client.post("/v1/answer", json=body)
            except httpx.HTTPError as exc:
                error = {"status_code": None, "detail": f"{type(exc).__name__}: {exc}"}
                continue
            row["wall_ms"] = round((time.perf_counter() - started) * 1000)
            if response.status_code == 200:
                data = response.json()
                row.update(
                    answer=data["answer"], grounded=data["grounded"], sources=data["sources"],
                    retrieval_query=data["retrieval_query"], scope_status=data["scope"]["status"],
                    citations=data["citations"], meta=data["meta"],
                    chunks=[{k: chunk[k] for k in ("rank", "source", "heading_path", "content")}
                            for chunk in data.get("chunks") or []],
                    created_at=now(),
                )
                return row
            error = {"status_code": response.status_code, "detail": response.text[:500]}
            if response.status_code not in RETRYABLE_STATUS:
                break
        row.update(error=error, created_at=now())
        return row

    @staticmethod
    def locked(meta: dict[str, Any]) -> dict[str, Any]:
        values = {k: meta.get(k) for k in LOCKED_SETTINGS}
        values["llm_provider"] = (meta.get("llm") or {}).get("provider")
        return values

    def check_settings(self, run: dict[str, Any], row: dict[str, Any]) -> None:
        """Server đổi cấu hình giữa chừng (sửa .env rồi khởi động lại) thì dừng."""
        current = self.locked(row["meta"])
        if run["settings"] is None:
            if current["retrieval_mode"] is not None and current["llm_provider"] is not None:
                run["settings"] = current
                self._write_run(run)
            return
        changed = {k: (run["settings"][k], v) for k, v in current.items()
                   if v is not None and run["settings"][k] is not None and v != run["settings"][k]}
        if changed:
            detail = "; ".join(f"{k}: {old!r} → {new!r}" for k, (old, new) in changed.items())
            raise ExperimentError(f"Cấu hình server đổi giữa chừng ({detail}). Dừng để không trộn kết quả.")

    def run(self, questions_path: Path, questions: list[dict[str, Any]], *, limit: int | None = None) -> dict[str, Any]:
        run = self.prepare(questions_path, questions)
        rows = load_rows(self.rows_path)
        todo = [q for q in questions if q["id"] not in rows or "error" in rows[q["id"]]]
        pending = todo if limit is None else todo[:limit]
        skipped = f" (--limit: còn {len(todo) - len(pending)} câu cho lần sau)" if len(pending) < len(todo) else ""
        self.log(f"{self.config.variant}: {len(questions) - len(todo)}/{len(questions)} câu đã có, "
                 f"chạy {len(pending)} câu{skipped}. "
                 f"Model: {run['model']} ({run['llm_provider']}), index: {run['index']}.")
        with self.rows_path.open("a", encoding="utf-8") as handle:
            for number, question in enumerate(pending, start=1):
                row = self.ask(question)
                if "meta" in row:
                    self.check_settings(run, row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                rows[question["id"]] = row
                self.log(f"[{number}/{len(pending)}] {question['id']}: " + (
                    f"LỖI {row['error']['status_code']}" if "error" in row else
                    f"{row['wall_ms']} ms" + (" · BỊ CẮT" if (row["meta"].get("llm") or {}).get("truncated") else "")
                ))

        # Probe lại: model bị đổi trên Vast trong lúc chạy thì phải biết.
        try:
            final_model = self.check_llm()["identity"]
        except ExperimentError as exc:
            final_model = None
            self.log(f"Không probe lại được LLM sau khi chạy ({exc}); kết quả vẫn được giữ.")
        session = run["sessions"][-1]
        session.update(finished_at=now(), model_at_end=final_model)
        if final_model is not None and final_model != run["model"]:
            session["model_changed_during_run"] = True
            self.log(f"CẢNH BÁO: model đổi từ {run['model']} sang {final_model} trong lúc chạy.")
        self._write_run(run)
        self.compact(questions, rows)
        return summarize(list(rows.values()))

    def compact(self, questions: list[dict[str, Any]], rows: dict[str, dict[str, Any]]) -> None:
        """Ghi lại file theo thứ tự câu hỏi, mỗi câu một dòng (bỏ các lần lỗi cũ)."""
        order = {q["id"]: i for i, q in enumerate(questions)}
        ordered = sorted(rows.values(), key=lambda row: order.get(row["question_id"], len(order)))
        temporary = self.rows_path.with_suffix(".tmp")
        temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered), encoding="utf-8")
        temporary.replace(self.rows_path)


def percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if "error" not in row]
    llm = [row["meta"]["llm"] for row in ok if row["meta"].get("llm")]
    generate = [row["meta"]["timing_ms"]["generate"] for row in ok if row["meta"]["timing_ms"]["generate"] is not None]
    output_tokens = [item["output_tokens"] for item in llm if item.get("output_tokens") is not None]
    return {
        "questions": len(rows), "answered": len(ok), "errors": len(rows) - len(ok),
        "refused": sum(1 for row in ok if not row["grounded"]),
        "truncated": sum(1 for item in llm if item.get("truncated")),
        "invalid_citations": sum(1 for row in ok if row["citations"]["invalid"]),
        "generate_ms_median": round(statistics.median(generate)) if generate else None,
        "generate_ms_p95": percentile(generate, 0.95),
        "output_tokens_mean": round(statistics.fmean(output_tokens)) if output_tokens else None,
        "answer_chars_median": round(statistics.median(len(row["answer"]) for row in ok)) if ok else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", required=True, type=Path, help="File JSONL, mỗi dòng {id, question, ...}.")
    parser.add_argument("--variant", required=True, help="Tên cấu hình, ví dụ qwen32b-structure.")
    parser.add_argument("--index", choices=["recursive", "structure"], help="Bỏ trống = index mặc định của server.")
    parser.add_argument("--llm-provider", choices=["ollama", "gemini", "vllm"], help="Bỏ trống = LLM_PROVIDER của server.")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--mode", choices=["semantic", "bm25", "hybrid"])
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--api", help="Mặc định http://127.0.0.1:<RAG_API_PORT>.")
    parser.add_argument("--api-key", help="Mặc định RAG_API_KEY trong .env.")
    parser.add_argument("--out", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--timeout", type=float, default=300, help="Giây cho mỗi câu (model lớn cần lâu).")
    parser.add_argument("--retries", type=int, default=2, help="Số lần thử lại khi mạng/LLM lỗi (502/503/504).")
    parser.add_argument("--limit", type=int, help="Chỉ chạy N câu đầu còn thiếu (thử nhanh).")
    parser.add_argument("--allow-stale", action="store_true", help="Cho chạy trên index không khớp data/.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from config import get_api_settings  # nạp .env của RAG-module

    settings = get_api_settings()
    api = (args.api or f"http://127.0.0.1:{settings.port}").rstrip("/")
    key = args.api_key if args.api_key is not None else settings.api_key
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    config = RunConfig(args.variant, args.index, args.llm_provider, args.top_k, args.mode, args.rerank)
    try:
        questions = load_questions(args.questions)
        with httpx.Client(base_url=api, headers=headers, timeout=args.timeout) as client:
            summary = Runner(client, config, args.out, allow_stale=args.allow_stale,
                             retries=args.retries).run(args.questions, questions, limit=args.limit)
    except ExperimentError as exc:
        print(f"Dừng: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["truncated"]:
        print(f"Có {summary['truncated']} câu bị cắt ở giới hạn token: tăng VLLM_MAX_TOKENS rồi chạy "
              "variant mới trước khi khảo sát.", file=sys.stderr)
    return 0 if summary["errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
