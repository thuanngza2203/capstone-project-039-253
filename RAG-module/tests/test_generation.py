from __future__ import annotations

import pytest
import config
import rag
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda


@pytest.fixture(autouse=True)
def generation_only(monkeypatch: pytest.MonkeyPatch) -> None:
    # Các test này kiểm tra prompt/LLM, dùng thứ tự cố định của fake semantic store.
    monkeypatch.setenv("RETRIEVAL_MODE", "semantic")


class RecordingVectorStore:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents

    def similarity_search(self, question: str, *, k: int) -> list[Document]:
        return self.documents[:k]


def test_ask_builds_grounded_prompt_and_returns_unique_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = [
        Document(
            page_content="Vết bệnh ghẻ có màu xanh ô liu.",
            metadata={"source": "apple/apple_scab.txt"},
        ),
        Document(
            page_content="Lá bệnh thối đen có đốm mắt ếch.",
            metadata={"source": "apple/apple_black_rot.txt"},
        ),
        Document(
            page_content="Cần thu gom lá bệnh.",
            metadata={"source": "apple/apple_scab.txt"},
        ),
    ]
    captured: dict[str, str] = {}

    def fake_generate(prompt_value):
        captured["prompt"] = prompt_value.to_string()
        return AIMessage(
            content="Vết bệnh thường có màu xanh ô liu [Nguồn 1]."
        )

    monkeypatch.setattr(
        rag,
        "create_chat_model",
        lambda: pytest.fail("Không được gọi Gemini khi đã inject fake LLM"),
    )
    answer, sources = rag.ask(
        "  Triệu chứng bệnh ghẻ táo là gì?  ",
        k=3,
        llm=RunnableLambda(fake_generate),
        vector_store=RecordingVectorStore(documents),
    )

    assert answer == "Vết bệnh thường có màu xanh ô liu [Nguồn 1]."
    assert sources == [
        "apple/apple_scab.txt",
        "apple/apple_black_rot.txt",
    ]
    prompt = captured["prompt"]
    assert "Chỉ dùng thông tin trong phần NGỮ CẢNH" in prompt
    assert "Không tự tạo nguồn, tên thuốc, hoạt chất, liều lượng" in prompt
    assert "[Nguồn 1: apple/apple_scab.txt]" in prompt
    assert "[Nguồn 2: apple/apple_black_rot.txt]" in prompt
    assert "[Nguồn 3: apple/apple_scab.txt]" in prompt
    assert "Triệu chứng bệnh ghẻ táo là gì?" in prompt


def test_ask_does_not_create_llm_when_retrieval_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rag,
        "create_chat_model",
        lambda: pytest.fail("Không được gọi LLM khi retrieval rỗng"),
    )

    answer, sources = rag.ask(
        "Câu hỏi ngoài corpus",
        vector_store=RecordingVectorStore([]),
    )

    assert "chưa có đủ thông tin" in answer
    assert sources == []


def test_ask_requires_gemini_key_only_after_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    store = RecordingVectorStore(
        [
            Document(
                page_content="Thông tin bệnh ghẻ táo.",
                metadata={"source": "apple/apple_scab.txt"},
            )
        ]
    )

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        rag.ask("Bệnh ghẻ táo là gì?", vector_store=store)


def test_ask_rejects_empty_llm_response() -> None:
    llm = RunnableLambda(lambda _: AIMessage(content="   "))
    store = RecordingVectorStore(
        [
            Document(
                page_content="Thông tin bệnh ghẻ táo.",
                metadata={"source": "apple/apple_scab.txt"},
            )
        ]
    )

    with pytest.raises(RuntimeError, match="LLM trả về nội dung rỗng"):
        rag.ask("Bệnh ghẻ táo là gì?", llm=llm, vector_store=store)


def test_ask_propagates_provider_error_without_fake_answer() -> None:
    def raise_quota_error(_):
        raise RuntimeError("quota exceeded")

    llm = RunnableLambda(raise_quota_error)
    store = RecordingVectorStore(
        [
            Document(
                page_content="Thông tin bệnh ghẻ táo.",
                metadata={"source": "apple/apple_scab.txt"},
            )
        ]
    )

    with pytest.raises(RuntimeError, match="quota exceeded"):
        rag.ask("Bệnh ghẻ táo là gì?", llm=llm, vector_store=store)
