from __future__ import annotations

import pytest
import rag
import retrieval
from config import RetrievalSettings, get_retrieval_settings
from langchain_core.documents import Document
from retrieval import SearchHit, bm25_search, chunk_key, reciprocal_rank_fusion


def document(text: str, source: str = "test.txt", start: int = 0) -> Document:
    return Document(page_content=text, metadata={"source": source, "start_index": start})


class Store:
    def __init__(self, corpus, semantic_results=None):
        self.corpus = corpus
        self.semantic_results = semantic_results if semantic_results is not None else corpus
        self.semantic_calls = []

    def get(self, *, include):
        return {
            "documents": [doc.page_content for doc in self.corpus],
            "metadatas": [doc.metadata for doc in self.corpus],
        }

    def similarity_search(self, question, *, k):
        self.semantic_calls.append((question, k))
        return self.semantic_results[:k]


class ChromaLikeStore(Store):
    """Giong Chroma: ``include=[]`` chi tra id, nen cache corpus co hieu luc."""

    def __init__(self, corpus, semantic_results=None):
        super().__init__(corpus, semantic_results)
        self.content_reads = 0

    def get(self, *, include):
        if not include:
            return {"ids": [f"id-{index}" for index in range(len(self.corpus))]}
        self.content_reads += 1
        return super().get(include=include)


def test_corpus_is_read_once_while_collection_is_unchanged() -> None:
    docs = [document("Venturia inaequalis", "scab.txt"),
            document("Diplodia seriata", "rot.txt")]
    store = ChromaLikeStore(docs)
    assert rag.retrieve("Venturia", mode="bm25", vector_store=store)
    assert rag.retrieve("Diplodia", mode="bm25", vector_store=store)
    assert store.content_reads == 1


def test_corpus_is_read_again_after_collection_changes() -> None:
    docs = [document("Venturia inaequalis", "scab.txt")]
    store = ChromaLikeStore(docs)
    assert rag.retrieve("Venturia", mode="bm25", vector_store=store)
    store.corpus = docs + [document("Diplodia seriata", "rot.txt")]
    found = rag.retrieve("Diplodia", mode="bm25", vector_store=store)
    assert [doc.metadata["source"] for doc in found] == ["rot.txt"]
    assert store.content_reads == 2


def test_store_without_ids_still_returns_correct_results() -> None:
    """Store khong tra 'ids' thi bo cache; ket qua van phai dung."""
    store = Store([document("Venturia inaequalis", "scab.txt"),
                   document("Diplodia seriata", "rot.txt")])
    for _ in range(2):
        found = rag.retrieve("Diplodia", mode="bm25", vector_store=store)
        assert [doc.metadata["source"] for doc in found] == ["rot.txt"]


@pytest.mark.parametrize("question", ["ĐỐM MẮT ẾCH", "dom mat ech", "đốm mắt ếch"])
def test_bm25_handles_case_accents_and_unicode_forms(question: str) -> None:
    correct = document("Lá có đốm mắt ếch.", "apple.txt")
    other = document("Lá có phấn trắng.", "cherry.txt")
    hits = bm25_search(question, [correct, other], limit=4)
    assert hits[0].document == correct


@pytest.mark.parametrize("corpus", [[], [document("  ")], [document("!!!")]])
def test_bm25_handles_empty_text(corpus) -> None:
    assert bm25_search("Venturia", corpus, limit=4) == []


def test_bm25_does_not_fill_results_with_zero_overlap() -> None:
    docs = [document("Venturia inaequalis"), document("Alternaria solani")]
    assert bm25_search("unmatchedword", docs, limit=4) == []
    assert bm25_search("???", docs, limit=4) == []
    assert len(bm25_search("Venturia", docs, limit=4)) == 1


def test_rrf_rewards_agreement_and_ignores_raw_score_scale() -> None:
    a, b, c = [document(name, f"{name}.txt") for name in "abc"]
    dense = [SearchHit(a, chunk_key(a), semantic_rank=1),
             SearchHit(b, chunk_key(b), semantic_rank=2)]
    sparse = [SearchHit(c, chunk_key(c), bm25_rank=1, bm25_score=1_000_000),
              SearchHit(b, chunk_key(b), bm25_rank=2, bm25_score=1)]
    hits = reciprocal_rank_fusion(dense, sparse, rrf_k=60)
    assert hits[0].document == b
    assert hits[0].rrf_score == pytest.approx(2 / 62)
    assert hits[0].semantic_rank == hits[0].bm25_rank == 2


def test_rrf_deduplicates_per_branch_but_keeps_different_chunks() -> None:
    first = document("Đoạn thứ nhất", start=0)
    second = document("Đoạn thứ hai", start=100)
    dense_hit = SearchHit(first, chunk_key(first), semantic_rank=1)
    sparse_hit = SearchHit(first, chunk_key(first), bm25_rank=1, bm25_score=10)
    other = SearchHit(second, chunk_key(second), bm25_rank=2, bm25_score=5)
    hits = reciprocal_rank_fusion([dense_hit, dense_hit], [sparse_hit, other], 60)
    assert len(hits) == 2
    assert hits[0].rrf_score == pytest.approx(2 / 61)
    assert chunk_key(first) != chunk_key(second)


def test_hybrid_recovers_candidate_missing_from_semantic_results() -> None:
    correct = document("Tác nhân Venturia inaequalis gây bệnh ghẻ táo.", "scab.txt")
    wrong = document("Bệnh phấn trắng trên cherry.", "mildew.txt")
    store = Store([correct, wrong], semantic_results=[wrong])
    result = rag.retrieve_with_debug("Venturia inaequalis", k=2, vector_store=store)
    assert correct in result.documents
    rescued = next(hit for hit in result.hits if hit.document == correct)
    assert rescued.bm25_rank == 1
    assert rescued.semantic_rank is None
    assert result.semantic_count == result.bm25_count == 1
    assert result.merged_count == 2


@pytest.mark.parametrize("mode", ["semantic", "bm25", "hybrid"])
def test_mode_uses_only_requested_branches(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    doc = document("Venturia")
    store = Store([doc])
    if mode == "semantic":
        monkeypatch.setattr(store, "get", lambda **kwargs: pytest.fail("No BM25 read"))
    if mode == "bm25":
        monkeypatch.setattr(store, "similarity_search", lambda *a, **kw: pytest.fail("No embedding"))
    assert rag.retrieve("Venturia", mode=mode, vector_store=store) == [doc]


def test_reranker_receives_bounded_candidates_and_can_change_top_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = [document(f"text {i}", f"{i}.txt") for i in range(8)]
    settings = RetrievalSettings(mode="semantic", candidate_k=3, reranker_enabled=True)
    received = []

    class Reranker:
        def predict(self, pairs, **kwargs):
            received.extend(pairs)
            return [0.2, 0.9, 0.1]

    monkeypatch.setattr(retrieval, "load_reranker", lambda *args: Reranker())
    result = retrieval.search_store("question", Store(docs), k=1, settings=settings)
    assert received == [("question", doc.page_content) for doc in docs[:3]]
    assert result.documents == [docs[1]]
    assert result.hits[0].semantic_rank == 2
    assert result.hits[0].rerank_score == 0.9
    assert result.reranked
    # Debug vẫn còn ứng viên không được chọn; Document/metadata gốc không bị sửa.
    debug = result.to_debug_dict()
    assert len(debug["candidates"]) == 3
    assert sum(row["selected"] for row in debug["candidates"]) == 1
    assert "rerank_score" not in docs[1].metadata


@pytest.mark.parametrize("scores", [[], [float("nan")], [float("inf")]])
def test_bad_reranker_output_is_reported(scores, monkeypatch: pytest.MonkeyPatch) -> None:
    class Reranker:
        def predict(self, pairs, **kwargs):
            return scores

    monkeypatch.setattr(retrieval, "load_reranker", lambda *args: Reranker())
    with pytest.raises(RuntimeError, match="Reranker thất bại"):
        rag.retrieve("Venturia", vector_store=Store([document("Venturia")]), rerank=True)


def test_reranker_disabled_or_empty_results_does_not_load_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(retrieval, "load_reranker", lambda *a: pytest.fail("No model download"))
    assert rag.retrieve("Venturia", vector_store=Store([]), rerank=True) == []
    assert rag.retrieve("Venturia", vector_store=Store([document("Venturia")]), rerank=False)


def test_requested_k_can_exceed_default_candidate_limit() -> None:
    docs = [document(str(i), start=i) for i in range(25)]
    result = rag.retrieve("query", k=25, mode="semantic", vector_store=Store(docs))
    assert len(result) == 25


@pytest.mark.parametrize(("name", "value"), [
    ("RETRIEVAL_MODE", "unknown"), ("RETRIEVAL_CANDIDATE_K", "0"),
    ("RETRIEVAL_CANDIDATE_K", "abc"), ("RETRIEVAL_RRF_K", "-1"),
    ("RERANKER_ENABLED", "maybe"),
])
def test_invalid_settings_are_explicit(name, value, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        get_retrieval_settings()


def test_explicit_options_override_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVAL_MODE", "bm25")
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    settings = get_retrieval_settings(mode="SEMANTIC", rerank=False)
    assert settings.mode == "semantic"
    assert not settings.reranker_enabled


@pytest.mark.parametrize(("question", "source"), [
    ("Bệnh cháy lá sớm trên khoai tây có triệu chứng gì?", "potato/potato_early_blight.txt"),
    ("Cách trị early blight trên potato?", "potato/potato_early_blight.txt"),
    ("Corynespora cassiicola", "tomato/tomato_target_spot.txt"),
])
def test_real_corpus_bm25_keeps_expected_source_in_top_four(question, source) -> None:
    chunks = rag.split_documents(rag.load_documents())
    result = rag.retrieve(question, k=4, mode="bm25", vector_store=Store(chunks))
    assert source in {doc.metadata["source"] for doc in result}
