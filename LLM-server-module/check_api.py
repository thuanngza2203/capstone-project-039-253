"""Kiểm tra model ID, API key và chat API; không cần embedding hay Chroma."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from serve import MODULE_DIR, read_environment


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


def check_api(base_url: str, model: str, api_key: str, *, timeout: int = 120) -> str:
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
    models = request_json(f"{base_url}/models", api_key, timeout=timeout)
    entries = models.get("data")
    if not isinstance(entries, list):
        raise ValueError("Models API thiếu danh sách data.")
    available = [item.get("id") for item in entries if isinstance(item, dict)]
    if model not in available:
        raise ValueError(f"Không tìm thấy model {model!r}. API đang có: {available}")
    response = request_json(f"{base_url}/chat/completions", api_key, timeout=timeout, payload={
        "model": model,
        "messages": [{"role": "user", "content": "Trả lời ngắn bằng tiếng Việt: bạn đã sẵn sàng chưa?"}],
        "max_tokens": 128,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Chat API thiếu choices[0].message.content.") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM trả content rỗng; kiểm tra thinking, model và giới hạn output.")
    return content.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gọi thử API LLM, không chạy RAG.")
    parser.add_argument("--env-file", type=Path, default=MODULE_DIR / ".env",
                        help="File server .env hoặc RAG-module/.env trên host.")
    parser.add_argument("--base-url", help="Override URL, ví dụ http://127.0.0.1:8001/v1.")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args(argv)
    try:
        env = read_environment(args.env_file)
        base_url = args.base_url or env.get("VLLM_BASE_URL") or f"http://127.0.0.1:{env.get('LLM_PORT', '8000')}/v1"
        model = env.get("VLLM_MODEL") or env.get("LLM_SERVED_MODEL_NAME", "qwen3.5-4b")
        api_key = env.get("VLLM_API_KEY") or env.get("LLM_API_KEY", "")
        answer = check_api(base_url, model.strip(), api_key.strip(), timeout=args.timeout)
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
