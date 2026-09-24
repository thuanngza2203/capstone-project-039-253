from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import config
import httpx
import langchain_openai
import pytest
import rag
from langchain_core.documents import Document


@pytest.mark.parametrize("provider", ["ollama", "gemini", "vllm"])
def test_env_file_selects_provider_on_startup(
    provider: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Đổi đúng một dòng .env sẽ chọn factory tương ứng ở lần chạy sau."""
    config_path = tmp_path / "config.py"
    config_path.write_text(Path(config.__file__).read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / ".env").write_text(f"LLM_PROVIDER= {provider.upper()} \n", encoding="utf-8")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("VLLM_THINK", "")

    for name, class_name in [
        ("langchain_ollama", "ChatOllama"),
        ("langchain_google_genai", "ChatGoogleGenerativeAI"),
        ("langchain_openai", "ChatOpenAI"),
    ]:
        def create_fake(*, _name=name, **kwargs):
            return _name

        monkeypatch.setitem(sys.modules, name, SimpleNamespace(**{class_name: create_fake}))

    loaded = runpy.run_path(str(config_path))
    expected = {
        "ollama": "langchain_ollama",
        "gemini": "langchain_google_genai",
        "vllm": "langchain_openai",
    }
    assert loaded["create_chat_model"]() == expected[provider]


@pytest.mark.parametrize(
    ("api_key", "think", "expected_think"),
    [("", "", None), ("docker-test-key", "false", False), ("docker-test-key", "true", True)],
)
@pytest.mark.parametrize("base_url", ["http://docker.test:8000/v1", "https://vast.test:31443/v1"])
def test_ask_sends_rag_prompt_to_vllm_chat_api(
    api_key: str,
    think: str,
    expected_think: bool | None,
    base_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real LangChain/SDK request with an offline HTTP transport."""
    monkeypatch.setenv("LLM_PROVIDER", "vllm")
    monkeypatch.setenv("VLLM_BASE_URL", base_url)
    monkeypatch.setenv("VLLM_MODEL", "plant-chat")
    monkeypatch.setenv("VLLM_API_KEY", api_key)
    monkeypatch.setenv("VLLM_MAX_TOKENS", "321")
    monkeypatch.setenv("VLLM_TIMEOUT", "25")
    monkeypatch.setenv("VLLM_THINK", think)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # An OpenAI account must not override the chosen local/remote endpoint/key.
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-used")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://must-not-be-used.invalid/v1")
    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "plant-chat",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Đốm xanh ô liu [Nguồn 1]."},
                    "finish_reason": "stop",
                }],
            },
        )

    class Store:
        def get(self, *, include):
            docs = self.similarity_search("", k=1)
            return {
                "documents": [doc.page_content for doc in docs],
                "metadatas": [doc.metadata for doc in docs],
            }

        def similarity_search(self, question, *, k):
            return [Document(
                page_content="Lá bệnh có đốm xanh ô liu.",
                metadata={"source": "apple/apple_scab.txt"},
            )]

    chat_class = langchain_openai.ChatOpenAI
    with httpx.Client(transport=httpx.MockTransport(handle)) as http_client:
        monkeypatch.setattr(
            langchain_openai,
            "ChatOpenAI",
            lambda **kwargs: chat_class(http_client=http_client, **kwargs),
        )
        answer, sources = rag.ask("Triệu chứng bệnh ghẻ táo?", vector_store=Store())

    assert answer == "Đốm xanh ô liu [Nguồn 1]."
    assert sources == ["apple/apple_scab.txt"]
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == f"{base_url}/chat/completions"
    assert request.headers["authorization"] == f"Bearer {api_key or 'EMPTY'}"
    assert request.extensions["timeout"]["read"] == 25
    body = json.loads(request.content)
    assert body["model"] == "plant-chat"
    assert body.get("max_completion_tokens", body.get("max_tokens")) == 321
    assert "Triệu chứng bệnh ghẻ táo?" in body["messages"][1]["content"]
    assert "[Nguồn 1: apple/apple_scab.txt]" in body["messages"][1]["content"]
    assert "Lá bệnh có đốm xanh ô liu." in body["messages"][1]["content"]
    if expected_think is None:
        assert "chat_template_kwargs" not in body
    else:
        assert body["chat_template_kwargs"] == {"enable_thinking": expected_think}


@pytest.mark.parametrize(
    ("setting", "value"),
    [("VLLM_MAX_TOKENS", "0"), ("VLLM_MAX_TOKENS", "abc"),
     ("VLLM_TIMEOUT", "-1"), ("VLLM_TIMEOUT", "abc"), ("VLLM_THINK", "invalid"),
     # URL sai phải báo tại chỗ, không để server remote trả 404/401 sau vòng mạng.
     ("VLLM_BASE_URL", "http://127.0.0.1:8000"),
     ("VLLM_BASE_URL", "http://127.0.0.1:8000/v1/chat/completions"),
     ("VLLM_BASE_URL", "ftp://llm.example.com/v1"),
     ("VLLM_BASE_URL", "/v1"),
     ("VLLM_BASE_URL", "https://user:pass@llm.example.com/v1")],
)
def test_invalid_vllm_config_fails_before_client_creation(
    setting: str, value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "vllm")
    monkeypatch.setenv("VLLM_MAX_TOKENS", "800")
    monkeypatch.setenv("VLLM_TIMEOUT", "120")
    monkeypatch.setenv("VLLM_THINK", "")
    monkeypatch.setenv(setting, value)
    monkeypatch.setattr(
        langchain_openai, "ChatOpenAI",
        lambda **kwargs: pytest.fail("Invalid config must not create an API client"),
    )
    with pytest.raises(RuntimeError, match=setting):
        config.create_chat_model()


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8001/v1",
    "https://llm.example.com:31443/v1",
    "https://llm.example.com/v1/",
])
def test_remote_vast_urls_are_accepted(url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """SSH tunnel và HTTPS public đều phải qua được validate."""
    monkeypatch.setenv("LLM_PROVIDER", "vllm")
    monkeypatch.setenv("VLLM_BASE_URL", url)
    monkeypatch.setenv("VLLM_THINK", "")
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", lambda **kwargs: kwargs)
    assert config.create_chat_model()["base_url"] == url.rstrip("/")


def _vllm_client(monkeypatch: pytest.MonkeyPatch, **env: str) -> dict:
    monkeypatch.setenv("LLM_PROVIDER", "vllm")
    monkeypatch.setenv("VLLM_THINK", "")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", lambda **kwargs: kwargs)
    return config.create_chat_model()


@pytest.mark.parametrize(("env", "expected"), [
    # Không điền gì: giữ nguyên mặc định cũ.
    ({}, "http://127.0.0.1:8000/v1"),
    # SSH tunnel: host là đầu tunnel trên máy mình.
    ({"VLLM_HOST": "127.0.0.1", "VLLM_PORT": "8001"}, "http://127.0.0.1:8001/v1"),
    # IP public Vast với cổng được map.
    ({"VLLM_HOST": "203.0.113.10", "VLLM_PORT": "41234"}, "http://203.0.113.10:41234/v1"),
    # HTTPS qua nginx, tên miền dùng cổng mặc định 443 thì không cần cổng.
    ({"VLLM_SCHEME": "HTTPS", "VLLM_HOST": "llm.example.com"}, "https://llm.example.com/v1"),
    # Cách cũ vẫn chạy: URL đầy đủ, không điền host/port.
    ({"VLLM_BASE_URL": "https://vast.test:31443/v1/"}, "https://vast.test:31443/v1"),
    # VLLM_SCHEME thường luôn có trong .env; không coi là xung đột với URL đầy đủ.
    ({"VLLM_SCHEME": "http", "VLLM_BASE_URL": "http://10.0.0.5:8000/v1"}, "http://10.0.0.5:8000/v1"),
])
def test_vllm_address_from_host_port_or_full_url(
    env: dict, expected: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _vllm_client(monkeypatch, **env)["base_url"] == expected


@pytest.mark.parametrize(("env", "message"), [
    ({"VLLM_BASE_URL": "http://a:8000/v1", "VLLM_HOST": "b"}, "Chỉ dùng một cách"),
    ({"VLLM_BASE_URL": "http://a:8000/v1", "VLLM_PORT": "8001"}, "Chỉ dùng một cách"),
    ({"VLLM_PORT": "8001"}, "phải điền cả VLLM_HOST"),
    ({"VLLM_HOST": "http://203.0.113.10"}, "không kèm http://"),
    ({"VLLM_HOST": "203.0.113.10:8001"}, "không kèm cổng"),
    ({"VLLM_HOST": "203.0.113.10/v1"}, "chỉ là IP hoặc tên miền"),
    ({"VLLM_HOST": "user@203.0.113.10"}, "chỉ là IP hoặc tên miền"),
    ({"VLLM_HOST": "203.0.113.10", "VLLM_PORT": "abc"}, "VLLM_PORT"),
    ({"VLLM_HOST": "203.0.113.10", "VLLM_PORT": "0"}, "VLLM_PORT"),
    ({"VLLM_HOST": "203.0.113.10", "VLLM_PORT": "70000"}, "VLLM_PORT"),
    ({"VLLM_HOST": "203.0.113.10", "VLLM_SCHEME": "ftp"}, "VLLM_SCHEME"),
])
def test_invalid_vllm_address_fails_before_network(
    env: dict, message: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _vllm_client(monkeypatch, **env)


def test_missing_vllm_dependency_reports_install_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "vllm")
    monkeypatch.setenv("VLLM_THINK", "")
    monkeypatch.setitem(sys.modules, "langchain_openai", None)
    with pytest.raises(RuntimeError, match="pip install -r requirements.txt"):
        config.create_chat_model()
