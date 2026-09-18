from __future__ import annotations

from pathlib import Path

import pytest
import rag
from langchain_core.documents import Document


class RecordingVectorStore:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.calls = []

    def get(self, *, include):
        return {
            "documents": [doc.page_content for doc in self.documents],
            "metadatas": [doc.metadata for doc in self.documents],
        }

    # Cố ý không nhận filter: test sẽ fail nếu code tự khóa phạm vi theo alias.
    def similarity_search(self, question: str, *, k: int) -> list[Document]:
        self.calls.append((question, k))
        return self.documents[:k]


@pytest.mark.parametrize("mode", ["semantic", "hybrid"])
@pytest.mark.parametrize("question", [
    "Bệnh cháy lá sớm trên khoai tây có triệu chứng gì?",
    "Cách trị early blight trên potato?",
    "Không phải cháy lá dâu tây, có thể là bệnh gì?",
    "So sánh bệnh cháy lá dâu tây và cháy sớm khoai tây",
])
def test_alias_does_not_exclude_other_crops(question: str, mode: str) -> None:
    docs = [
        Document(page_content="Cháy lá dâu tây.", metadata={
            "source": "strawberry.txt", "disease_id": "strawberry-scorch",
            "disease_aliases": "cháy lá",
        }),
        Document(page_content="Cháy sớm khoai tây, early blight.", metadata={
            "source": "potato.txt", "disease_id": "potato-blight",
        }),
    ]
    store = RecordingVectorStore(docs)
    result = rag.retrieve(question, k=4, mode=mode, vector_store=store)
    assert {doc.metadata["source"] for doc in result} == {"strawberry.txt", "potato.txt"}
    assert store.calls[0][0] == question


def test_build_index_twice_replaces_collection_without_duplicates(
    apple_data_dir: Path, tmp_path: Path, fake_embeddings,
) -> None:
    options = dict(embeddings=fake_embeddings, collection_name="test_rebuild",
                   chunk_size=100_000, chunk_overlap=0)
    index_dir = tmp_path / "chroma"
    first = rag.build_index(apple_data_dir, index_dir, **options)
    second = rag.build_index(apple_data_dir, index_dir, **options)
    documents = rag.retrieve(
        "Bệnh ghẻ táo có triệu chứng gì?", k=10,
        embeddings=fake_embeddings, persist_directory=index_dir,
        collection_name="test_rebuild",
    )
    assert first == second == (2, 2)
    # Không ép mọi kết quả vào scab: cả hai tài liệu vẫn được xét.
    assert len(documents) == 2
    assert documents[0].metadata["source"] == "apple/apple_scab.txt"
    assert fake_embeddings.document_calls == 2


@pytest.mark.parametrize("mode", ["semantic", "bm25", "hybrid"])
def test_persisted_search_ranks_relevant_apple_document(
    mode: str, apple_data_dir: Path, tmp_path: Path, fake_embeddings,
) -> None:
    index_dir = tmp_path / "chroma"
    rag.build_index(
        apple_data_dir, index_dir, embeddings=fake_embeddings,
        collection_name="test_relevance", chunk_size=100_000, chunk_overlap=0,
    )
    result = rag.retrieve(
        "Triệu chứng bệnh ghẻ táo do Venturia là gì?", k=1, mode=mode,
        embeddings=fake_embeddings, persist_directory=index_dir,
        collection_name="test_relevance",
    )
    assert result[0].metadata["source"] == "apple/apple_scab.txt"


def test_bm25_reads_current_index_without_loading_embedding(
    apple_data_dir: Path, tmp_path: Path, fake_embeddings, monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_dir = tmp_path / "chroma"
    options = dict(embeddings=fake_embeddings, collection_name="test_snapshot")
    rag.build_index(apple_data_dir, index_dir, **options)
    monkeypatch.setattr(rag, "create_embeddings", lambda: pytest.fail("BM25 needs no model"))

    # Sửa TXT chưa index: BM25 phải tiếp tục tìm trên snapshot Chroma cũ.
    scab_file = apple_data_dir / "apple" / "apple_scab.txt"
    scab_file.write_text("newuniqueterm", encoding="utf-8")
    search_options = dict(mode="bm25", persist_directory=index_dir,
                          collection_name="test_snapshot")
    assert rag.retrieve("newuniqueterm", **search_options) == []
    assert rag.retrieve("Venturia", **search_options)

    # Rebuild cùng collection: BM25 phải thấy text mới, không dùng cache cũ.
    rag.build_index(apple_data_dir, index_dir, **options)
    result = rag.retrieve("newuniqueterm", **search_options)
    assert len(result) == 1
    assert "newuniqueterm" in result[0].page_content
    assert fake_embeddings.query_calls == 0


def test_retrieve_uses_injected_store_and_strips_question() -> None:
    expected = Document(page_content="Đốm mắt ếch.", metadata={"source": "apple.txt"})
    store = RecordingVectorStore([expected])
    result = rag.retrieve("  đốm mắt ếch  ", k=3, vector_store=store)
    assert result == [expected]
    assert store.calls[0][0] == "đốm mắt ếch"
    assert store.calls[0][1] >= 3


@pytest.mark.parametrize(("question", "k", "message"), [
    ("   ", 4, "Câu hỏi không được để trống"),
    ("bệnh ghẻ táo", 0, "k phải lớn hơn 0"),
])
def test_retrieve_validates_input(question: str, k: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        rag.retrieve(question, k=k, vector_store=RecordingVectorStore([]))


def test_retrieve_reports_missing_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rag, "create_embeddings", lambda: pytest.fail("Missing index"))
    with pytest.raises(RuntimeError, match="Index chưa được tạo"):
        rag.retrieve("ghẻ táo", persist_directory=tmp_path / "missing")


@pytest.mark.parametrize("mode", ["semantic", "bm25", "hybrid"])
def test_retrieve_reports_empty_index(mode: str, tmp_path: Path, fake_embeddings) -> None:
    with pytest.raises(RuntimeError, match="Index đang rỗng"):
        rag.retrieve(
            "ghẻ táo", mode=mode, persist_directory=tmp_path,
            embeddings=fake_embeddings, collection_name="empty",
        )
