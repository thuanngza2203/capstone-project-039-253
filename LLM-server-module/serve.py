"""Đọc cấu hình riêng của server và khởi động API có sẵn của vLLM.

Không import RAG hoặc tự viết lại chat API. vLLM giữ model trong GPU và xử lý
request; Python process này được thay bằng vLLM khi khởi động thành công.
"""

from __future__ import annotations

import argparse
import math
import os
import shlex
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values

MODULE_DIR = Path(__file__).resolve().parent


def read_environment(path: Path) -> dict[str, str]:
    """Biến shell có ưu tiên hơn file, giống cách RAG-module đọc cấu hình."""
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {path}. Tạo .env hoặc kiểm tra --env-file.")
    # utf-8-sig đọc được cả UTF-8 thường và file có BOM do editor Windows tạo.
    values = {
        key: value for key, value in dotenv_values(path, encoding="utf-8-sig").items()
        if value is not None
    }
    return {**values, **os.environ}


def positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    try:
        value = int(env.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} phải là số nguyên lớn hơn 0.") from exc
    if value < 1:
        raise ValueError(f"{name} phải là số nguyên lớn hơn 0.")
    return value


@dataclass(frozen=True)
class ServerSettings:
    model_id: str
    served_model_name: str
    api_key: str = field(repr=False)
    port: int
    max_model_len: int
    max_num_seqs: int
    tensor_parallel_size: int
    gpu_memory_utilization: float
    dtype: str
    reasoning_parser: str
    language_model_only: bool
    cache_dir: Path

    @classmethod
    def from_environment(cls, env: Mapping[str, str]) -> ServerSettings:
        model = env.get("LLM_MODEL_ID", "Qwen/Qwen3.5-4B").strip()
        name = env.get("LLM_SERVED_MODEL_NAME", "qwen3.5-4b").strip()
        key = env.get("LLM_API_KEY", "").strip()
        if not model or not name:
            raise ValueError("LLM_MODEL_ID và LLM_SERVED_MODEL_NAME không được để trống.")
        if not key or any(character.isspace() for character in key):
            raise ValueError("Điền LLM_API_KEY không chứa khoảng trắng vào .env của server.")
        port = positive_int(env, "LLM_PORT", 8000)
        if port > 65535:
            raise ValueError("LLM_PORT phải nằm trong 1–65535.")
        try:
            memory = float(env.get("LLM_GPU_MEMORY_UTILIZATION", "0.80"))
        except ValueError as exc:
            raise ValueError("LLM_GPU_MEMORY_UTILIZATION phải nằm trong (0, 1].") from exc
        if not math.isfinite(memory) or not 0 < memory <= 1:
            raise ValueError("LLM_GPU_MEMORY_UTILIZATION phải nằm trong (0, 1].")
        language_only = env.get("LLM_LANGUAGE_MODEL_ONLY", "true").strip().lower()
        if language_only not in {"true", "false"}:
            raise ValueError("LLM_LANGUAGE_MODEL_ONLY phải là true hoặc false.")
        cache = Path(env.get("LLM_CACHE_DIR", "models/huggingface").strip() or "models/huggingface")
        if not cache.is_absolute():
            cache = MODULE_DIR / cache
        return cls(
            model_id=model,
            served_model_name=name,
            api_key=key,
            port=port,
            max_model_len=positive_int(env, "LLM_MAX_MODEL_LEN", 8192),
            max_num_seqs=positive_int(env, "LLM_MAX_NUM_SEQS", 1),
            tensor_parallel_size=positive_int(env, "LLM_TENSOR_PARALLEL_SIZE", 1),
            gpu_memory_utilization=memory,
            dtype=env.get("LLM_DTYPE", "auto").strip() or "auto",
            reasoning_parser=env.get("LLM_REASONING_PARSER", "qwen3").strip(),
            language_model_only=language_only == "true",
            cache_dir=cache.resolve(),
        )


def build_command(settings: ServerSettings) -> list[str]:
    """Truyền argv trực tiếp, không nối model/config thành shell command."""
    command = [
        "vllm", "serve", settings.model_id,
        "--host", "127.0.0.1",
        "--port", str(settings.port),
        "--served-model-name", settings.served_model_name,
        "--max-model-len", str(settings.max_model_len),
        "--max-num-seqs", str(settings.max_num_seqs),
        "--tensor-parallel-size", str(settings.tensor_parallel_size),
        "--gpu-memory-utilization", str(settings.gpu_memory_utilization),
        "--dtype", settings.dtype,
    ]
    if settings.language_model_only:
        command.append("--language-model-only")
    if settings.reasoning_parser:
        command.extend(["--reasoning-parser", settings.reasoning_parser])
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chạy riêng LLM trên Linux/Vast.ai bằng vLLM.")
    parser.add_argument("--env-file", type=Path, default=MODULE_DIR / ".env")
    parser.add_argument("--dry-run", action="store_true", help="Kiểm tra config/in lệnh, không nạp model.")
    args = parser.parse_args(argv)
    try:
        env = read_environment(args.env_file)
        settings = ServerSettings.from_environment(env)
        command = build_command(settings)
        print(shlex.join(command), flush=True)
        print("API key: đã cấu hình (ẩn).", flush=True)
        if args.dry_run:
            return 0
        if sys.platform != "linux":
            raise ValueError("Khởi động vLLM trên Linux; Windows chỉ dùng --dry-run hoặc check_api.py.")
        executable = shutil.which("vllm")
        if executable is None:
            raise ValueError("Không tìm thấy vllm. Chạy bash install.sh và activate .venv của module này.")
        settings.cache_dir.mkdir(parents=True, exist_ok=True)
        # vLLM hỗ trợ key qua environment; không đưa secret vào argv/log.
        env["VLLM_API_KEY"] = settings.api_key
        env["HF_HOME"] = str(settings.cache_dir)
        os.execvpe(executable, command, env)
    except (ValueError, OSError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
