"""Đọc cấu hình riêng của server và khởi động API có sẵn của vLLM.

Không import RAG hoặc tự viết lại chat API. vLLM giữ model trong GPU và xử lý
request; Python process này được thay bằng vLLM khi khởi động thành công.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_SERVED_MODEL_NAME = "rag-llm"
# Kho chứng chỉ của hệ điều hành: Debian/Ubuntu (image Vast), rồi RHEL, Alpine.
SYSTEM_CA_BUNDLES = (
    Path("/etc/ssl/certs/ca-certificates.crt"),
    Path("/etc/pki/tls/certs/ca-bundle.crt"),
    Path("/etc/ssl/cert.pem"),
)
# Mặc định của huggingface_hub là 10 giây: mạng Vast chậm là hết giờ, transformers nuốt lỗi
# và vLLM chỉ báo "Can't load tokenizer". Đặt trong .env thì giữ giá trị đó.
HF_TIMEOUTS = {"HF_HUB_DOWNLOAD_TIMEOUT": "60", "HF_HUB_ETAG_TIMEOUT": "30"}
DOWNLOAD_HINTS = """Không tải được model {model} từ HuggingFace. Lỗi thật in ở trên; cách xử lý theo lỗi:
- "429" / "Too Many Requests": HF giới hạn máy chưa đăng nhập. Tạo token (huggingface.co/settings/tokens),
  thêm HF_TOKEN=<token> vào .env.
- "401" / "403" / "gated": model cần token hoặc phải chấp nhận điều khoản trên trang model; thêm HF_TOKEN.
- "404" / "Repository Not Found": sai LLM_MODEL_ID.
- "timed out" / "Connection": mạng chậm hoặc chặn HuggingFace. Tăng HF_HUB_DOWNLOAD_TIMEOUT trong .env,
  hoặc tải qua mirror: HF_ENDPOINT=https://hf-mirror.com.
- "No space left on device": ổ đĩa đầy (model mặc định cần khoảng 32 GB trống).
Sửa .env rồi chạy lại python serve.py: phần đã tải được giữ lại."""


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


def chat_template_kwargs(env: Mapping[str, str]) -> dict:
    """Đọc tùy chọn template của model; để trống thì dùng mặc định của vLLM."""
    raw = env.get("LLM_CHAT_TEMPLATE_KWARGS", "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("Cần JSON object.")
        # json.loads mặc định chấp nhận NaN/Infinity dù JSON chuẩn không có.
        json.dumps(value, allow_nan=False)
    except ValueError as exc:
        raise ValueError("LLM_CHAT_TEMPLATE_KWARGS phải là JSON object hợp lệ.") from exc
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
    quantization: str
    chat_template_file: Path | None
    chat_template_kwargs: dict

    @classmethod
    def from_environment(cls, env: Mapping[str, str]) -> ServerSettings:
        model = env.get("LLM_MODEL_ID", "Qwen/Qwen3.8-27B-FP8").strip()
        name = env.get("LLM_SERVED_MODEL_NAME", DEFAULT_SERVED_MODEL_NAME).strip()
        key = env.get("LLM_API_KEY", "").strip()
        if not model or not name:
            raise ValueError("LLM_MODEL_ID và LLM_SERVED_MODEL_NAME không được để trống.")
        if not key or any(character.isspace() for character in key):
            raise ValueError("Điền LLM_API_KEY không chứa khoảng trắng vào .env của server.")
        port = positive_int(env, "LLM_PORT", 8000)
        if port > 65535:
            raise ValueError("LLM_PORT phải nằm trong 1–65535.")
        try:
            memory = float(env.get("LLM_GPU_MEMORY_UTILIZATION", "0.90"))
        except ValueError as exc:
            raise ValueError("LLM_GPU_MEMORY_UTILIZATION phải nằm trong (0, 1].") from exc
        if not math.isfinite(memory) or not 0 < memory <= 1:
            raise ValueError("LLM_GPU_MEMORY_UTILIZATION phải nằm trong (0, 1].")
        language_only = env.get("LLM_LANGUAGE_MODEL_ONLY", "false").strip().lower()
        if language_only not in {"true", "false"}:
            raise ValueError("LLM_LANGUAGE_MODEL_ONLY phải là true hoặc false.")
        cache = Path(env.get("LLM_CACHE_DIR", "models/huggingface").strip() or "models/huggingface")
        if not cache.is_absolute():
            cache = MODULE_DIR / cache
        template_path = env.get("LLM_CHAT_TEMPLATE_FILE", "").strip()
        template = Path(template_path) if template_path else None
        if template is not None:
            if not template.is_absolute():
                template = MODULE_DIR / template
            template = template.resolve()
            if not template.is_file():
                raise ValueError(f"LLM_CHAT_TEMPLATE_FILE không tìm thấy file: {template}")
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
            reasoning_parser=env.get("LLM_REASONING_PARSER", "").strip(),
            language_model_only=language_only == "true",
            cache_dir=cache.resolve(),
            quantization=env.get("LLM_QUANTIZATION", "").strip(),
            chat_template_file=template,
            chat_template_kwargs=chat_template_kwargs(env),
        )


def use_system_certificates(env: dict[str, str], bundles: tuple[Path, ...] | None = None) -> str | None:
    """Cho vLLM tải model bằng kho chứng chỉ của hệ điều hành.

    Một số máy Vast đi qua proxy HTTPS có CA riêng, CA đó chỉ nằm trong kho của hệ điều hành.
    huggingface_hub (httpx) mặc định chỉ tin certifi nên tải model báo CERTIFICATE_VERIFY_FAILED
    ("Can't load tokenizer"). Máy bình thường thì hai kho có cùng CA nên không đổi gì.
    Đã đặt SSL_CERT_FILE/REQUESTS_CA_BUNDLE (shell hoặc .env) thì giữ nguyên.
    """
    bundle = env.get("SSL_CERT_FILE", "").strip()
    if not bundle:
        candidates = SYSTEM_CA_BUNDLES if bundles is None else bundles
        found = next((path for path in candidates if path.is_file() and path.stat().st_size > 0), None)
        if found is None:
            return None
        bundle = env["SSL_CERT_FILE"] = str(found)
    if not env.get("REQUESTS_CA_BUNDLE", "").strip():
        env["REQUESTS_CA_BUNDLE"] = bundle
    return bundle


def is_true(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def download_model(model_id: str, env: dict[str, str], hf: str | None, run=None) -> None:
    """Tải đủ model vào HF_HOME bằng `hf download` trước khi bật vLLM, rồi cho vLLM chạy offline.

    vLLM/transformers nuốt lỗi tải và chỉ báo "Can't load tokenizer"; `hf download` in lỗi thật, có
    thanh tiến độ và tải tiếp phần còn thiếu khi chạy lại. Xet hỏng (ví dụ "CAS Client Error ... 401")
    thì tải lại bằng HTTPS thường (HF_HUB_DISABLE_XET=1). Bỏ qua khi model là thư mục trên máy,
    khi đã đặt HF_HUB_OFFLINE=1 (model có sẵn trong cache) hoặc khi venv không có lệnh `hf`.
    """
    if hf is None or is_true(env.get("HF_HUB_OFFLINE")) or Path(model_id).is_dir():
        return
    run = run or subprocess.run
    print(f"Tải model {model_id} vào {env.get('HF_HOME', 'cache HuggingFace')} "
          "(lần đầu mất vài chục GB; chạy lại thì tải tiếp phần còn thiếu)...", flush=True)
    for name, value in HF_TIMEOUTS.items():
        if not env.get(name, "").strip():
            env[name] = value
    attempts = [dict(env)]
    if not is_true(env.get("HF_HUB_DISABLE_XET")):
        attempts.append({**env, "HF_HUB_DISABLE_XET": "1"})
    for number, attempt in enumerate(attempts):
        if number:
            print("Tải qua Xet lỗi; tải lại bằng HTTPS thường (HF_HUB_DISABLE_XET=1)...", flush=True)
        if run([hf, "download", model_id], env=attempt).returncode == 0:
            env.update(attempt)
            # Model đã đủ trong cache: vLLM không gọi mạng nữa, không còn lỗi tải bị che.
            env["HF_HUB_OFFLINE"] = "1"
            return
    raise ValueError(DOWNLOAD_HINTS.format(model=model_id))


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
    if settings.quantization:
        command.extend(["--quantization", settings.quantization])
    if settings.chat_template_file is not None:
        command.extend(["--chat-template", str(settings.chat_template_file)])
    if settings.chat_template_kwargs:
        command.extend([
            "--default-chat-template-kwargs",
            json.dumps(settings.chat_template_kwargs, ensure_ascii=False, allow_nan=False),
        ])
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
        # Vast template có thể cài sẵn vLLM khác trong PATH. Chỉ dùng bản đi
        # cùng Python đang chạy để giữ đúng dependencies của venv đã chọn.
        executable = shutil.which("vllm", path=str(Path(sys.executable).parent))
        if executable is None:
            raise ValueError("Không tìm thấy vllm. Chạy bash install.sh và activate .venv của module này.")
        settings.cache_dir.mkdir(parents=True, exist_ok=True)
        # vLLM hỗ trợ key qua environment; không đưa secret vào argv/log.
        env["VLLM_API_KEY"] = settings.api_key
        env["HF_HOME"] = str(settings.cache_dir)
        bundle = use_system_certificates(env)
        if bundle:
            print(f"Chứng chỉ TLS khi tải model: {bundle}", flush=True)
        download_model(settings.model_id, env, shutil.which("hf", path=str(Path(sys.executable).parent)))
        os.execvpe(executable, command, env)
    except (ValueError, OSError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
