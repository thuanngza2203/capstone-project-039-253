"""Tìm có phạm vi: không phạm vi thì y như cũ; có phạm vi thì mọi nhánh đều bị giới hạn."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

import rag
from config import COLLECTION_NAME, get_retrieval_settings
from retrieval import MetadataScope, bm25_search_index, build_corpus_index, search_store


def document(text: str, source: str, crop: str) -> Document:
    return Document(page_content=text, metadata={"source": source, "crop": crop})


def test_chroma_filter_shapes() -> None:
    assert MetadataScope().chroma_filter() is None
    assert not MetadataScope().active
    assert MetadataScope(sources=("a.txt",)).chroma_filter() == {"source": "a.txt"}
    assert MetadataScope(sources=("a.txt", "b.txt")).chroma_filter() == {
        "source": {"$in": ["a.txt", "b.txt"]}}
    assert MetadataScope(crop="apple").chroma_filter() == {"crop": "apple"}
    assert MetadataScope(sources=("a.txt",), crop="apple").chroma_filter() == {
        "$and": [{"source": "a.txt"}, {"crop": "apple"}]}


def test_bm25_filters_before_truncating_top_k() -> None:
    """Top của toàn corpus nằm ngoài phạm vi thì phạm vi vẫn phải có kết quả."""
    corpus = [document("xử lý xử lý xử lý gỉ sắt", f"rust{i}.txt", "apple") for i in range(3)]
    corpus.append(document("xử lý ghẻ táo", "scab.txt", "apple"))
    index = build_corpus_index(corpus)
    scope = MetadataScope(sources=("scab.txt",))
    hits = bm25_search_index("xử lý", index, limit=1, keep=scope.matches)
    assert [hit.document.metadata["source"] for hit in hits] == ["scab.txt"]
    assert hits[0].bm25_rank == 1
    assert bm25_search_index("xử lý", index, limit=1)[0].document.metadata["source"] != "scab.txt"


@pytest.fixture
def real_store(tmp_path: Path, apple_data_dir: Path, fake_embeddings):
    """Chroma thật trong thư mục tạm: kiểm tra cú pháp filter với chromadb thật."""
    rag.build_index(apple_data_dir, tmp_path / "index", embeddings=fake_embeddings)
    return rag.new_vector_store(fake_embeddings, tmp_path / "index", COLLECTION_NAME)


@pytest.mark.parametrize("mode", ["semantic", "bm25", "hybrid"])
def test_real_chroma_scope_limits_every_branch(real_store, mode: str) -> None:
    settings = get_retrieval_settings(mode=mode)
    query = "thối đen mắt ếch Diplodia ghẻ Venturia"
    # Không phạm vi: tài liệu ngoài phạm vi lọt vào top (thối đen áp đảo query này).
    unscoped = search_store(query, real_store, k=6, settings=settings)
    assert "apple/apple_black_rot.txt" in {d.metadata["source"] for d in unscoped.documents}

    scoped = search_store(query, real_store, k=6, settings=settings,
                          scope=MetadataScope(sources=("apple/apple_scab.txt",)))
    assert scoped.documents
    assert {d.metadata["source"] for d in scoped.documents} == {"apple/apple_scab.txt"}
    assert scoped.to_debug_dict()["scope"] == {"sources": ["apple/apple_scab.txt"], "crop": None}


def test_real_chroma_crop_and_multi_source_filters(real_store) -> None:
    settings = get_retrieval_settings(mode="hybrid")
    both = MetadataScope(sources=("apple/apple_scab.txt", "apple/apple_black_rot.txt"))
    assert search_store("bệnh", real_store, k=4, settings=settings, scope=both).documents
    assert search_store("bệnh", real_store, k=4, settings=settings,
                        scope=MetadataScope(crop="apple")).documents
    # Cây không có trong index: rỗng, không lấy bù tài liệu cây khác.
    empty = search_store("bệnh", real_store, k=4, settings=settings,
                         scope=MetadataScope(crop="tomato"))
    assert empty.documents == []


def test_retrieve_passes_scope_through(real_store) -> None:
    found = rag.retrieve("thối đen", k=4, vector_store=real_store,
                         scope=MetadataScope(sources=("apple/apple_scab.txt",)))
    assert {d.metadata["source"] for d in found} == {"apple/apple_scab.txt"}
