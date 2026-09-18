"""Kiểm tra phép chấm evidence độc lập với retrieval/model."""
from langchain_core.documents import Document

from scripts.benchmark_chunking import coverage, resolve_evidence, score_documents


def span(start, end, source="a.txt"):
    return Document(page_content="irrelevant header", metadata={"source": source, "start_index": start, "end_index": end})


def labels(quote="ABC\n\nDEF"):
    rows = [{"id": "q1", "expected_evidence": [{"source": "a.txt", "quote": quote}], "required_context": []}]
    documents = [Document(page_content=quote, metadata={"source": "a.txt"})]
    return resolve_evidence(rows, documents)["q1"]


def test_coverage_ignores_whitespace_between_chunks():
    evidence = labels()["expected_evidence"][0]
    assert coverage(evidence, [span(0, 3), span(5, 8)]) == 1


def test_missing_actual_characters_cannot_count_as_complete():
    evidence = labels()["expected_evidence"][0]
    assert coverage(evidence, [span(0, 2), span(5, 8)]) == 5 / 6


def test_overlapping_hits_are_not_double_counted():
    evidence = labels("ABCDEF")["expected_evidence"][0]
    assert coverage(evidence, [span(0, 4), span(2, 6), span(0, 6)]) == 1
    assert coverage(evidence, [span(0, 4), span(0, 4)]) == 4 / 6


def test_score_does_not_accept_wrong_source_or_header_only_match():
    expected = labels("ABCDEF")
    score = score_documents([span(0, 6, "b.txt"), span(0, 3), span(3, 6)], expected)
    assert score["hit_at_1"] == 0
    assert score["recall_at_k"] == 1
    assert score["mrr_at_k"] == 0  # Hai đoạn cùng phủ đủ; không chunk nào tự có đủ bằng chứng.
