"""Kiểm tra vòng đời tài nguyên, không tải checkpoint hoặc gọi provider thật."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import config
import pytest
import rag
import retrieval
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda


@pytest.fixture(autouse=True)
def clear_model_caches():
    # Không để fake model trong cache ảnh hưởng các test chạy sau đó.
    config.create_embeddings.cache_clear()
    retrieval.load_reranker.cache_clear()
    yield
    config.create_embeddings.cache_clear()
    retrieval.load_reranker.cache_clear()


@pytest.mark.parametrize("rerank", [False, True])
def test_multiple_questions_reuse_models_store_and_client(
    monkeypatch, tmp_path, fake_embeddings, rerank,
):
    calls = {"embedding": 0, "reranker": 0, "predict": 0, "store": 0, "llm": 0}
    prompts = []
    doc = Document(page_content="Bệnh ghẻ táo có vết xanh ô liu.", metadata={"source": "apple.txt"})

    def embedding_factory(**kwargs):
        calls["embedding"] += 1
        return fake_embeddings

    class FakeReranker:
        def __init__(self, *args, **kwargs):
            calls["reranker"] += 1

        def predict(self, pairs, **kwargs):
            calls["predict"] += 1
            return [1.0] * len(pairs)

    class Store:
        def get(self, **kwargs):
            return {"ids": ["one"], "documents": [doc.page_content], "metadatas": [doc.metadata]}

        def similarity_search(self, question, *, k):
            fake_embeddings.embed_query(question)
            return [doc]

    def store_factory(embeddings, *args):
        calls["store"] += 1
        assert embeddings is fake_embeddings
        return Store()

    def generate(prompt):
        prompts.append(prompt.to_string())
        return AIMessage(content="Có vết xanh ô liu [Nguồn 1].")

    def llm_factory():
        calls["llm"] += 1
        return RunnableLambda(generate)

    monkeypatch.setitem(sys.modules, "langchain_huggingface", SimpleNamespace(HuggingFaceEmbeddings=embedding_factory))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=FakeReranker))
    monkeypatch.setattr(rag, "new_vector_store", store_factory)
    monkeypatch.setattr(rag, "create_chat_model", llm_factory)

    # Test vòng đời tài nguyên, tắt memory để không thêm lời gọi rewrite.
    session = rag.RAGSession(persist_directory=tmp_path, mode="hybrid", rerank=rerank, history_turns=0)
    assert all(value == 0 for value in calls.values())  # Constructor chưa nạp tài nguyên.
    session.warmup()
    session.warmup()
    assert calls["llm"] == 0  # Warmup không gửi câu hỏi đến LLM.
    assert session.search("bệnh ghẻ táo").documents == [doc]
    for question in ("Ghẻ táo có triệu chứng gì?", "Màu vết bệnh ghẻ táo?"):
        answer, sources = session.ask(question)
        assert answer == "Có vết xanh ô liu [Nguồn 1]."
        assert sources == ["apple.txt"]

    assert calls == {"embedding": 1, "reranker": int(rerank), "predict": 3 if rerank else 0, "store": 1, "llm": 1}
    assert fake_embeddings.query_calls == 3
    assert "Ghẻ táo có triệu chứng gì?" not in prompts[1]  # Không tự thêm chat history.


def test_bm25_session_reads_real_index_without_embedding_or_llm(
    monkeypatch, tmp_path, apple_data_dir, fake_embeddings,
):
    index_dir = tmp_path / "index"
    rag.build_index(apple_data_dir, index_dir, embeddings=fake_embeddings)
    monkeypatch.setattr(rag, "create_embeddings", lambda: pytest.fail("BM25 không cần embedding"))
    monkeypatch.setattr(rag, "create_chat_model", lambda: pytest.fail("Search không cần LLM"))
    session = rag.RAGSession(persist_directory=index_dir, mode="bm25", rerank=False)
    session.warmup()
    assert session.search("Venturia").documents
    assert session.search("Diplodia").documents
    assert fake_embeddings.query_calls == 0


def test_session_does_not_create_llm_for_empty_context(monkeypatch):
    store = SimpleNamespace(similarity_search=lambda *args, **kwargs: [])
    monkeypatch.setattr(rag, "create_chat_model", lambda: pytest.fail("Không được tạo LLM"))
    session = rag.RAGSession(vector_store=store, mode="semantic", rerank=False)
    answer, sources = session.ask("ngoài corpus")
    assert "chưa có đủ thông tin" in answer
    assert sources == []


def test_session_can_retry_failed_provider_initialization(monkeypatch):
    doc = Document(page_content="Thông tin ghẻ táo.", metadata={"source": "apple.txt"})
    store = SimpleNamespace(similarity_search=lambda *args, **kwargs: [doc])
    calls = []

    def llm_factory():
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("provider unavailable")
        return RunnableLambda(lambda _: AIMessage(content="answer"))

    monkeypatch.setattr(rag, "create_chat_model", llm_factory)
    session = rag.RAGSession(vector_store=store, mode="semantic", rerank=False, history_turns=0)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        session.ask("ghẻ táo")
    assert session.ask("ghẻ táo") == ("answer", ["apple.txt"])
    assert session.ask("triệu chứng ghẻ táo") == ("answer", ["apple.txt"])
    assert len(calls) == 2


def test_session_rejects_invalid_input_before_loading(monkeypatch):
    monkeypatch.setattr(rag, "_load_vector_store", lambda *args, **kwargs: pytest.fail("Không được mở index"))
    with pytest.raises(ValueError, match="k phải"):
        rag.RAGSession(k=0)
    session = rag.RAGSession()
    with pytest.raises(ValueError, match="để trống"):
        session.ask("  ")


def test_session_keeps_selected_settings_for_its_lifetime(monkeypatch):
    session = rag.RAGSession(mode="bm25", rerank=False)
    monkeypatch.setenv("RETRIEVAL_MODE", "semantic")
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    assert session.settings.mode == "bm25"
    assert session.settings.reranker_enabled is False
