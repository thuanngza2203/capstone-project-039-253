from __future__ import annotations

from pathlib import Path

import pytest
import rag
from langchain_core.documents import Document


class RecordingVectorStore:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.calls: list[tuple[str, int]] = []

    def similarity_search(self, question: str, *, k: int) -> list[Document]:
        self.calls.append((question, k))
        return self.documents[:k]


class CatalogVectorStore:
    """Fake store co catalog metadata va ho tro Chroma-style filter."""

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.get_calls: list[list[str] | None] = []
        self.search_calls: list[dict[str, object]] = []

    def get(self, *, include: list[str] | None = None):
        self.get_calls.append(include)
        return {
            "metadatas": [document.metadata for document in self.documents],
        }

    def similarity_search(
        self,
        question: str,
        *,
        k: int,
        filter: dict[str, str] | None = None,
    ) -> list[Document]:
        self.search_calls.append(
            {"question": question, "k": k, "filter": filter}
        )
        candidates = self.documents
        if filter:
            disease_id = filter["disease_id"]
            candidates = [
                document
                for document in candidates
                if document.metadata["disease_id"] == disease_id
            ]
        return candidates[:k]


@pytest.fixture
def catalog_store() -> CatalogVectorStore:
    return CatalogVectorStore(
        [
            Document(
                page_content="Thông tin bệnh ghẻ táo.",
                metadata={
                    "source": "apple/apple_scab.txt",
                    "disease": "Bệnh ghẻ táo",
                    "disease_aliases": (
                        "Bệnh ghẻ táo | Apple scab | bệnh sẹo táo | Scab"
                    ),
                    "disease_id": "apple_scab",
                },
            ),
            Document(
                page_content="Thông tin bệnh thối đen.",
                metadata={
                    "source": "apple/apple_black_rot.txt",
                    "disease": "Bệnh thối đen trên cây táo",
                    "disease_aliases": (
                        "Bệnh thối đen trên cây táo | Apple black rot | Black Rot"
                    ),
                    "disease_id": "apple_black_rot",
                },
            ),
        ]
    )


@pytest.mark.parametrize(
    "question",
    [
        "Bệnh ghẻ táo có triệu chứng gì?",
        "Bệnh ghẻ trên táo có triệu chứng gì?",
        "Apple scab gây hại như thế nào?",
        "Cách quản lý bệnh sẹo táo?",
    ],
)
def test_retrieve_filters_catalog_for_one_explicit_disease(
    question: str,
    catalog_store: CatalogVectorStore,
) -> None:
    documents = rag.retrieve(question, k=4, vector_store=catalog_store)

    assert [document.metadata["source"] for document in documents] == [
        "apple/apple_scab.txt"
    ]
    assert catalog_store.get_calls == [["metadatas"]]
    assert catalog_store.search_calls == [
        {
            "question": question,
            "k": 4,
            "filter": {"disease_id": "apple_scab"},
        }
    ]


@pytest.mark.parametrize(
    "question",
    [
        "Tài liệu có những triệu chứng nào trên lá?",
        "Quả táo bị thối và chuyển màu đen.",
        "Không phải bệnh ghẻ táo, có thể là gì khác?",
    ],
)
def test_retrieve_without_disease_name_searches_entire_corpus(
    question: str,
    catalog_store: CatalogVectorStore,
) -> None:
    documents = rag.retrieve(question, k=10, vector_store=catalog_store)

    assert {document.metadata["source"] for document in documents} == {
        "apple/apple_scab.txt",
        "apple/apple_black_rot.txt",
    }
    assert catalog_store.get_calls == [["metadatas"]]
    assert catalog_store.search_calls == [
        {"question": question, "k": 10, "filter": None}
    ]


@pytest.mark.parametrize(
    "question",
    [
        "So sánh bệnh ghẻ táo và bệnh thối đen trên cây táo.",
        "So sánh scab và black rot.",
    ],
)
def test_retrieve_for_two_named_diseases_does_not_narrow_to_one(
    question: str,
    catalog_store: CatalogVectorStore,
) -> None:
    documents = rag.retrieve(question, k=10, vector_store=catalog_store)

    assert {document.metadata["source"] for document in documents} == {
        "apple/apple_scab.txt",
        "apple/apple_black_rot.txt",
    }
    assert catalog_store.search_calls == [
        {"question": question, "k": 10, "filter": None}
    ]


def test_retrieve_filters_black_rot_from_explicit_vietnamese_name(
    catalog_store: CatalogVectorStore,
) -> None:
    question = "Bệnh thối đen trên táo có triệu chứng gì?"

    documents = rag.retrieve(question, k=4, vector_store=catalog_store)

    assert [document.metadata["source"] for document in documents] == [
        "apple/apple_black_rot.txt"
    ]
    assert catalog_store.search_calls == [
        {
            "question": question,
            "k": 4,
            "filter": {"disease_id": "apple_black_rot"},
        }
    ]


def test_build_index_twice_replaces_collection_without_duplicates(
    apple_data_dir: Path,
    tmp_path: Path,
    fake_embeddings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_dir = tmp_path / "chroma"
    collection_name = "test_rebuild"
    monkeypatch.setattr(
        rag,
        "create_embeddings",
        lambda: pytest.fail("Không được tải embedding model trong test"),
    )

    first_counts = rag.build_index(
        apple_data_dir,
        index_dir,
        embeddings=fake_embeddings,
        collection_name=collection_name,
        chunk_size=100_000,
        chunk_overlap=0,
    )
    second_counts = rag.build_index(
        apple_data_dir,
        index_dir,
        embeddings=fake_embeddings,
        collection_name=collection_name,
        chunk_size=100_000,
        chunk_overlap=0,
    )
    documents = rag.retrieve(
        "bệnh trên cây táo",
        k=10,
        embeddings=fake_embeddings,
        persist_directory=index_dir,
        collection_name=collection_name,
    )
    scab_documents = rag.retrieve(
        "Bệnh ghẻ táo có triệu chứng gì?",
        k=10,
        embeddings=fake_embeddings,
        persist_directory=index_dir,
        collection_name=collection_name,
    )

    assert first_counts == (2, 2)
    assert second_counts == (2, 2)
    assert len(documents) == 2
    assert {document.metadata["source"] for document in documents} == {
        "apple/apple_black_rot.txt",
        "apple/apple_scab.txt",
    }
    assert {document.metadata["source"] for document in scab_documents} == {
        "apple/apple_scab.txt"
    }
    assert fake_embeddings.document_calls == 2
    assert fake_embeddings.query_calls == 2


def test_real_chroma_rebuild_then_top_four_scab_query_stays_in_scab(
    apple_data_dir: Path,
    tmp_path: Path,
    fake_embeddings,
) -> None:
    index_dir = tmp_path / "chroma-filter"
    build_options = {
        "embeddings": fake_embeddings,
        "collection_name": "test_scab_filter",
        "chunk_size": 1_000,
        "chunk_overlap": 150,
    }
    first_counts = rag.build_index(
        apple_data_dir,
        index_dir,
        **build_options,
    )
    second_counts = rag.build_index(
        apple_data_dir,
        index_dir,
        **build_options,
    )

    documents = rag.retrieve(
        "Bệnh ghẻ táo có triệu chứng gì?",
        k=4,
        embeddings=fake_embeddings,
        persist_directory=index_dir,
        collection_name="test_scab_filter",
    )

    assert first_counts == second_counts
    assert first_counts[0] == 2
    assert len(documents) == 4
    assert {document.metadata["source"] for document in documents} == {
        "apple/apple_scab.txt"
    }


def test_persisted_retrieval_returns_relevant_apple_document(
    apple_data_dir: Path,
    tmp_path: Path,
    fake_embeddings,
) -> None:
    index_dir = tmp_path / "chroma"
    rag.build_index(
        apple_data_dir,
        index_dir,
        embeddings=fake_embeddings,
        collection_name="test_relevance",
        chunk_size=100_000,
        chunk_overlap=0,
    )

    result = rag.retrieve(
        "Triệu chứng bệnh ghẻ táo do Venturia là gì?",
        k=1,
        embeddings=fake_embeddings,
        persist_directory=index_dir,
        collection_name="test_relevance",
    )

    assert len(result) == 1
    assert result[0].metadata["source"] == "apple/apple_scab.txt"


def test_retrieve_uses_injected_store_and_strips_question() -> None:
    expected = Document(
        page_content="Đốm mắt ếch.",
        metadata={"source": "apple/apple_black_rot.txt"},
    )
    store = RecordingVectorStore([expected])

    result = rag.retrieve("  đốm mắt ếch  ", k=3, vector_store=store)

    assert result == [expected]
    assert store.calls == [("đốm mắt ếch", 3)]


@pytest.mark.parametrize(
    ("question", "k", "message"),
    [
        ("   ", 4, "Câu hỏi không được để trống"),
        ("bệnh ghẻ táo", 0, "k phải lớn hơn 0"),
    ],
)
def test_retrieve_validates_input(
    question: str,
    k: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        rag.retrieve(question, k=k, vector_store=RecordingVectorStore([]))


def test_retrieve_reports_missing_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_index = tmp_path / "not-built"
    monkeypatch.setattr(
        rag,
        "create_embeddings",
        lambda: pytest.fail("Không được tải embedding khi index chưa tồn tại"),
    )

    with pytest.raises(RuntimeError, match="Index chưa được tạo"):
        rag.retrieve(
            "bệnh ghẻ táo",
            persist_directory=missing_index,
            collection_name="missing",
        )


def test_retrieve_reports_empty_index(
    tmp_path: Path,
    fake_embeddings,
) -> None:
    empty_index = tmp_path / "empty-index"
    empty_index.mkdir()

    with pytest.raises(RuntimeError, match="Index đang rỗng"):
        rag.retrieve(
            "bệnh ghẻ táo",
            embeddings=fake_embeddings,
            persist_directory=empty_index,
            collection_name="empty",
        )
