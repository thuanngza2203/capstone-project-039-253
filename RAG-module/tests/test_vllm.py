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
def test_ask_sends_rag_prompt_to_vllm_chat_api(
    api_key: str,
    think: str,
    expected_think: bool | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real LangChain/SDK request with an offline HTTP transport."""
    monkeypatch.setattr(config, "LLM_PROVIDER", "vllm")
    monkeypatch.setattr(config, "VLLM_BASE_URL", "http://docker.test:8000/v1")
    monkeypatch.setattr(config, "VLLM_MODEL", "plant-chat")
    monkeypatch.setenv("VLLM_API_KEY", api_key)
    monkeypatch.setenv("VLLM_MAX_TOKENS", "321")
    monkeypatch.setenv("VLLM_TIMEOUT", "25")
    monkeypatch.setenv("VLLM_THINK", think)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # A configured OpenAI account must not override the chosen Docker endpoint/key.
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
    assert str(request.url) == "http://docker.test:8000/v1/chat/completions"
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
     ("VLLM_TIMEOUT", "-1"), ("VLLM_TIMEOUT", "abc"), ("VLLM_THINK", "invalid")],
)
def test_invalid_vllm_config_fails_before_client_creation(
    setting: str, value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "LLM_PROVIDER", "vllm")
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


def test_missing_vllm_dependency_reports_install_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "LLM_PROVIDER", "vllm")
    monkeypatch.setenv("VLLM_THINK", "")
    monkeypatch.setitem(sys.modules, "langchain_openai", None)
    with pytest.raises(RuntimeError, match="pip install -r requirements.txt"):
        config.create_chat_model()
