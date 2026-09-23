"""Kiểm tra model ID, API key và chat API; không cần embedding hay Chroma."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from serve import DEFAULT_SERVED_MODEL_NAME, MODULE_DIR, positive_int, read_environment


class RejectRedirects(HTTPRedirectHandler):
    """Chỉ gửi Bearer key đến endpoint người dùng cấu hình."""

    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


def open_request(request: Request, *, timeout: int):
    # urllib mặc định chuyển tiếp Authorization khi redirect; chặn và yêu cầu
    # người dùng đặt URL cuối cùng thay vì gửi key sang địa chỉ khác.
    return build_opener(RejectRedirects()).open(request, timeout=timeout)


def request_json(url: str, api_key: str, *, payload: dict | None = None, timeout: int = 120) -> dict:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=data, headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    })
    with open_request(request, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise ValueError("API trả JSON không đúng định dạng object.")
    return result


def check_api(
    base_url: str, model: str, api_key: str, *, timeout: int = 120,
    max_tokens: int = 800, think: bool | None = None,
) -> str:
    base_url = base_url.strip().rstrip("/")
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or not parsed.path.endswith("/v1"):
        raise ValueError("Base URL phải là http(s)://host[:port]/v1.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Base URL không chứa credentials, query hoặc fragment.")
    if not api_key.strip() or not model.strip() or timeout < 1:
        raise ValueError("Điền model/API key và dùng timeout lớn hơn 0.")
    if any(character.isspace() for character in api_key):
        raise ValueError("API key không được chứa khoảng trắng hoặc ký tự xuống dòng.")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise ValueError("max_tokens phải là số nguyên lớn hơn 0.")
    if think is not None and not isinstance(think, bool):
        raise ValueError("think phải là true, false hoặc None.")
    models = request_json(f"{base_url}/models", api_key, timeout=timeout)
    entries = models.get("data")
    if not isinstance(entries, list):
        raise ValueError("Models API thiếu danh sách data.")
    available = [item.get("id") for item in entries if isinstance(item, dict)]
    if model not in available:
        raise ValueError(f"Không tìm thấy model {model!r}. API đang có: {available}")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Trả lời ngắn bằng tiếng Việt: bạn đã sẵn sàng chưa?"}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    # Không ép tùy chọn riêng của Qwen lên mọi model. Nếu không override,
    # request dùng default-chat-template-kwargs của server đang chạy.
    if think is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": think}
    response = request_json(f"{base_url}/chat/completions", api_key, timeout=timeout, payload=payload)
    try:
        choice = response["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Chat API thiếu choices[0].message.content.") from exc
    if choice.get("finish_reason") == "length":
        raise ValueError(
            "LLM đã hết giới hạn output trước khi trả lời xong. "
            "Tăng --max-tokens hoặc chỉnh thinking theo model."
        )
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM trả content rỗng; kiểm tra thinking, model và giới hạn output.")
    return content.strip()


def client_base_url(env: dict[str, str]) -> str:
    """Cùng thứ tự với RAG-module: VLLM_BASE_URL, rồi VLLM_SCHEME/HOST/PORT.

    Chép lại thay vì import vì module server không phụ thuộc RAG. Không có biến
    phía client nào thì dùng LLM_PORT của .env server như trước.
    """
    explicit = env.get("VLLM_BASE_URL", "").strip()
    host = env.get("VLLM_HOST", "").strip()
    port = env.get("VLLM_PORT", "").strip()
    if explicit:
        if host or port:
            raise ValueError(
                "Chỉ dùng một cách: điền VLLM_HOST/VLLM_PORT, hoặc điền VLLM_BASE_URL "
                "đầy đủ. Để trống cách còn lại trong .env."
            )
        return explicit
    if not host:
        if port:
            raise ValueError("Đã điền VLLM_PORT thì phải điền cả VLLM_HOST.")
        return f"http://127.0.0.1:{env.get('LLM_PORT', '8000')}/v1"
    scheme = env.get("VLLM_SCHEME", "").strip().casefold() or "http"
    if scheme not in {"http", "https"}:
        raise ValueError("VLLM_SCHEME phải là http hoặc https.")
    if "://" in host:
        raise ValueError("VLLM_HOST chỉ là IP hoặc tên miền, không kèm http://.")
    if ":" in host:
        raise ValueError("VLLM_HOST không kèm cổng. Cổng đặt ở VLLM_PORT.")
    if any(character.isspace() or character in "/@?#" for character in host):
        raise ValueError("VLLM_HOST chỉ là IP hoặc tên miền, ví dụ 203.0.113.10.")
    if not port:
        return f"{scheme}://{host}/v1"
    if not port.isdigit() or not 1 <= int(port) <= 65535:
        raise ValueError("VLLM_PORT phải là số nguyên trong 1–65535.")
    return f"{scheme}://{host}:{int(port)}/v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gọi thử API LLM, không chạy RAG.")
    parser.add_argument("--env-file", type=Path, default=MODULE_DIR / ".env",
                        help="File server .env hoặc RAG-module/.env trên host.")
    parser.add_argument("--base-url", help="Override URL, ví dụ http://127.0.0.1:8001/v1.")
    parser.add_argument("--timeout", type=int, help="Ưu tiên hơn VLLM_TIMEOUT; mặc định 120 giây.")
    parser.add_argument("--max-tokens", type=int, help="Giới hạn output của câu hỏi kiểm tra.")
    args = parser.parse_args(argv)
    try:
        env = read_environment(args.env_file)
        base_url = args.base_url or client_base_url(env)
        model = env.get("VLLM_MODEL") or env.get("LLM_SERVED_MODEL_NAME", DEFAULT_SERVED_MODEL_NAME)
        api_key = env.get("VLLM_API_KEY") or env.get("LLM_API_KEY", "")
        timeout = args.timeout if args.timeout is not None else positive_int(env, "VLLM_TIMEOUT", 120)
        token_setting = "VLLM_MAX_TOKENS" if "VLLM_MAX_TOKENS" in env else "LLM_CHECK_MAX_TOKENS"
        max_tokens = args.max_tokens if args.max_tokens is not None else positive_int(env, token_setting, 800)
        # Khi kiểm tra bằng .env RAG, gửi cùng lựa chọn thinking với RAG.
        raw_think = env.get("VLLM_THINK", "").strip().casefold()
        think = None
        if raw_think:
            if raw_think not in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
                raise ValueError("VLLM_THINK phải là true, false hoặc để trống.")
            think = raw_think in {"true", "1", "yes", "on"}
        answer = check_api(
            base_url, model.strip(), api_key.strip(), timeout=timeout,
            max_tokens=max_tokens, think=think,
        )
    except HTTPError as exc:
        # Không in response body hoặc header để tránh đưa secret vào log lỗi.
        if 300 <= exc.code < 400:
            print(f"HTTP {exc.code}: API yêu cầu redirect. Đặt URL đích chính xác, gồm https và /v1.", file=sys.stderr)
        else:
            print(f"HTTP {exc.code}: kiểm tra API key (401/403), URL/proxy (404), hoặc server LLM.", file=sys.stderr)
        return 1
    except (URLError, TimeoutError, ValueError, OSError) as exc:
        print(f"Không gọi được API LLM: {exc}", file=sys.stderr)
        return 1
    print(f"API hoạt động, model: {model}")
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
