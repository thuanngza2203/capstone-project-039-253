"""Kiem tra phan deterministic cua hop dong trong SYSTEM_PROMPT.

Cac test nay khong cham chat luong cau chu. Chung chi bat nhung vi pham co the
kiem tra bang may: nhan [Nguon n] tro ra ngoai NGU CANH, danh so nguon khong
lien tuc, va cau tu choi lai di kem trich dan.
"""

from __future__ import annotations

import re

import pytest
import rag
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda


@pytest.fixture(autouse=True)
def semantic_only(monkeypatch: pytest.MonkeyPatch) -> None:
    # Giu thu tu co dinh cua store gia de so sanh so nguon.
    monkeypatch.setenv("RETRIEVAL_MODE", "semantic")


class Store:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents

    def similarity_search(self, question: str, *, k: int) -> list[Document]:
        return self.documents[:k]


def store_with(count: int) -> Store:
    return Store([
        Document(page_content=f"Noi dung {index}.",
                 metadata={"source": f"apple/doc_{index}.txt"})
        for index in range(1, count + 1)
    ])


def answering(text: str) -> RunnableLambda:
    return RunnableLambda(lambda prompt_value: AIMessage(content=text))


def capturing(text: str, captured: dict) -> RunnableLambda:
    def generate(prompt_value):
        captured["prompt"] = prompt_value.to_string()
        return AIMessage(content=text)

    return RunnableLambda(generate)


@pytest.mark.parametrize("answer", ["Xem [Nguồn 3].", "Theo [Nguồn 0] thì không."])
def test_citation_outside_context_is_reported(answer: str) -> None:
    with pytest.warns(UserWarning, match="dẫn nguồn không có trong NGỮ CẢNH"):
        rag.ask("Câu hỏi?", k=2, llm=answering(answer), vector_store=store_with(2))


def test_citations_inside_context_do_not_warn(recwarn: pytest.WarningsRecorder) -> None:
    answer, sources = rag.ask(
        "Câu hỏi?", k=2,
        llm=answering("Ý A [Nguồn 1] và ý B [Nguồn 2]."),
        vector_store=store_with(2),
    )
    assert answer.startswith("Ý A")
    assert len(sources) == 2
    assert [w for w in recwarn.list if "NGỮ CẢNH" in str(w.message)] == []


def test_context_sources_are_numbered_from_one_without_gaps() -> None:
    """Nhan trong prompt phai la 1..n de so nguon LLM dan co nghia."""
    captured: dict[str, str] = {}
    rag.ask(
        "Câu hỏi?", k=3,
        llm=capturing("Không dẫn nguồn.", captured),
        vector_store=store_with(3),
    )
    numbers = [int(value) for value in re.findall(r"\[Nguồn (\d+):", captured["prompt"])]
    assert numbers == [1, 2, 3]


def test_refusal_without_context_carries_no_citation() -> None:
    answer, sources = rag.ask("Câu hỏi ngoài corpus?", vector_store=store_with(0))
    assert "chưa có đủ thông tin" in answer
    assert sources == []
    assert rag.cited_source_numbers(answer) == []


def test_citation_parser_reads_numbers_in_order() -> None:
    answer = "Ý A [Nguồn 2], ý B [Nguồn 1], nhắc lại [ Nguồn 2 ]."
    assert rag.cited_source_numbers(answer) == [2, 1, 2]
    assert rag.invalid_citations(answer, document_count=2) == []
    assert rag.invalid_citations(answer, document_count=1) == [2]
