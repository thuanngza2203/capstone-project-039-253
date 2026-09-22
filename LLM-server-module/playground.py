"""Streamlit playground để thử LLM server trước khi ghép vào RAG.

Không import RAG, không nạp embedding/Chroma: chỉ gọi HTTP tới API OpenAI-
compatible của vLLM. Dùng chung .env và read_environment() với serve.py để
không sinh ra nguồn cấu hình thứ hai.

    streamlit run playground.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests
import streamlit as st

from serve import MODULE_DIR, read_environment

DEFAULT_SYSTEM = "Bạn là trợ lý hữu ích. Trả lời ngắn gọn bằng tiếng Việt."
# Prompt thật của RAG-module, để thử model trước khi ghép cả pipeline.
RAG_SYSTEM = """Bạn là trợ lý tra cứu kiến thức bệnh cây.

Quy tắc bắt buộc:
- Chỉ dùng thông tin trong phần NGỮ CẢNH được cung cấp.
- Nếu ngữ cảnh không đủ để trả lời, hãy nói rõ rằng kho tài liệu hiện tại chưa có đủ thông tin.
- Không tự tạo nguồn, tên thuốc, hoạt chất, liều lượng hoặc lịch phun.
- Khi dùng một thông tin, hãy dẫn nhãn [Nguồn n] tương ứng.
- Trả lời cùng ngôn ngữ với câu hỏi; nếu không xác định được thì trả lời bằng tiếng Việt."""

THINKING_DEFAULT = "mặc định của model"


def load_env(path: Path) -> dict[str, str]:
    """Thiếu .env thì vẫn mở được app; người dùng tự điền ở sidebar."""
    try:
        return read_environment(path)
    except (FileNotFoundError, OSError):
        return {}


def resolve_defaults(env: dict[str, str]) -> tuple[str, str, str]:
    """Ưu tiên biến phía client (VLLM_*) rồi mới tới biến phía server (LLM_*)."""
    base_url = env.get("VLLM_BASE_URL") or f"http://127.0.0.1:{env.get('LLM_PORT', '8000')}/v1"
    model = env.get("VLLM_MODEL") or env.get("LLM_SERVED_MODEL_NAME", "qwen3.8-27b-fp8")
    api_key = env.get("VLLM_API_KEY") or env.get("LLM_API_KEY", "thuanlocalmodel")
    return base_url.strip().rstrip("/"), model.strip(), api_key.strip()


def build_payload(messages, model, temperature, max_tokens, thinking, stream):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    # Để mặc định = không gửi kwarg, giống cách RAG xử lý VLLM_THINK để trống.
    if thinking != THINKING_DEFAULT:
        payload["chat_template_kwargs"] = {"enable_thinking": thinking == "bật"}
    if stream:
        payload["stream_options"] = {"include_usage": True}
    return payload


def headers_for(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def list_models(base_url: str, api_key: str, timeout: int) -> list[str]:
    response = requests.get(f"{base_url}/models", headers=headers_for(api_key), timeout=timeout)
    response.raise_for_status()
    return [item.get("id") for item in response.json().get("data", [])]


def send_blocking(base_url, api_key, payload, timeout):
    started = time.perf_counter()
    response = requests.post(
        f"{base_url}/chat/completions", headers=headers_for(api_key),
        json=payload, timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    message = body["choices"][0]["message"]
    return {
        "content": message.get("content") or "",
        # vLLM tách phần suy luận ra field riêng khi bật --reasoning-parser.
        "reasoning": message.get("reasoning_content") or "",
        "usage": body.get("usage") or {},
        "ttft": None,
        "elapsed": time.perf_counter() - started,
        "raw": body,
    }


def send_streaming(base_url, api_key, payload, timeout, sink):
    """Cùng cấu trúc kết quả với send_blocking, kèm TTFT đo được."""
    started = time.perf_counter()
    ttft = None
    content: list[str] = []
    reasoning: list[str] = []
    usage: dict = {}
    with requests.post(
        f"{base_url}/chat/completions", headers=headers_for(api_key),
        json=payload, timeout=timeout, stream=True,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            chunk = line[6:]
            if chunk == "[DONE]":
                break
            event = json.loads(chunk)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                piece = delta.get("content")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - started
                    content.append(piece)
                    sink("".join(content))
    return {
        "content": "".join(content),
        "reasoning": "".join(reasoning),
        "usage": usage,
        "ttft": ttft,
        "elapsed": time.perf_counter() - started,
        "raw": None,
    }


def main() -> None:
    st.set_page_config(page_title="LLM server playground", layout="wide")
    st.title("LLM server playground")
    st.caption("Thử riêng LLM. Không retrieval, không Chroma, không embedding.")

    env = load_env(MODULE_DIR / ".env")
    default_url, default_model, default_key = resolve_defaults(env)

    with st.sidebar:
        st.header("Kết nối")
        base_url = st.text_input("Base URL", value=default_url, help="Phải kết thúc bằng /v1")
        model = st.text_input("Model", value=default_model, help="Khớp LLM_SERVED_MODEL_NAME")
        api_key = st.text_input("API key", value=default_key, type="password")
        timeout = int(st.number_input("Timeout (giây)", 5, 600, 120))

        if st.button("Kiểm tra /v1/models", use_container_width=True):
            try:
                available = list_models(base_url.rstrip("/"), api_key, timeout)
            except Exception as exc:  # Playground: hiện lỗi thay vì dừng app.
                st.error(f"Không gọi được: {exc}")
            else:
                st.success(f"Model đang phục vụ: {available}")
                if model not in available:
                    st.warning(f"'{model}' không có trong danh sách trên.")

        st.header("Sinh câu trả lời")
        temperature = st.slider("Temperature", 0.0, 2.0, 0.0, 0.1)
        max_tokens = int(st.number_input("max_tokens", 16, 8192, 800))
        thinking = st.radio("enable_thinking", ["tắt", "bật", THINKING_DEFAULT], index=0,
                            help="Tương ứng VLLM_THINK bên RAG.")
        streaming = st.checkbox("Streaming (đo được TTFT)", value=True)

        st.header("System prompt")
        preset = st.radio("Chọn nhanh", ["Mặc định", "Prompt của RAG", "Tự viết"], index=0)

        if st.button("Xoá hội thoại", use_container_width=True):
            st.session_state.pop("history", None)
            st.rerun()

    system_default = {"Mặc định": DEFAULT_SYSTEM, "Prompt của RAG": RAG_SYSTEM}.get(preset, "")
    system_prompt = st.text_area("System prompt đang dùng", value=system_default, height=140)

    history = st.session_state.setdefault("history", [])
    for turn in history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    question = st.chat_input("Nhập câu hỏi để thử model...")
    if not question:
        return

    if not api_key:
        st.error("Chưa có API key. Điền ở sidebar hoặc đặt LLM_API_KEY trong .env.")
        return

    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    messages = [{"role": "system", "content": system_prompt}] if system_prompt.strip() else []
    messages += [{"role": turn["role"], "content": turn["content"]} for turn in history]
    payload = build_payload(messages, model, temperature, max_tokens, thinking, streaming)
    url = base_url.rstrip("/")

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            if streaming:
                result = send_streaming(url, api_key, payload, timeout,
                                        lambda text: placeholder.markdown(text + "|"))
            else:
                with st.spinner("Đang gọi model..."):
                    result = send_blocking(url, api_key, payload, timeout)
        except Exception as exc:  # Playground: báo lỗi rồi cho hỏi câu tiếp.
            placeholder.error(f"Lỗi gọi API: {exc}")
            history.pop()
            return

        answer = result["content"]
        if answer.strip():
            placeholder.markdown(answer)
        else:
            placeholder.warning(
                "Model trả content rỗng. Nếu mục Reasoning bên dưới có chữ thì reasoning "
                "parser đang nuốt câu trả lời: kiểm tra LLM_REASONING_PARSER."
            )

        usage = result["usage"]
        columns = st.columns(4)
        columns[0].metric("Tổng thời gian", f"{result['elapsed']:.2f}s")
        columns[1].metric("TTFT", "—" if result["ttft"] is None else f"{result['ttft']:.2f}s")
        columns[2].metric("Output token", usage.get("completion_tokens", "—"))
        produced = usage.get("completion_tokens")
        columns[3].metric(
            "Token/giây",
            f"{produced / result['elapsed']:.1f}" if produced and result["elapsed"] > 0 else "—",
        )

        if result["reasoning"]:
            with st.expander("Reasoning (vLLM tách riêng khỏi content)"):
                st.text(result["reasoning"])
        with st.expander("Payload đã gửi"):
            st.json(payload)
        if result["raw"] is not None:
            with st.expander("Response thô"):
                st.json(result["raw"])

    history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
