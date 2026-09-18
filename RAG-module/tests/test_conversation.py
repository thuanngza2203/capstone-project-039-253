"""Kiểm tra memory và query rewrite bằng fake LLM, không tải model."""

from __future__ import annotations

import json
import pytest
import rag
from config import ChatSettings, get_chat_settings
from conversation import ConversationMemory
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda


class RecordingStore:
    def __init__(self):
        self.queries = []

    def similarity_search(self, question, *, k):
        self.queries.append(question)
        if "khoai tây" in question:
            return [Document(page_content="Thông tin khoai tây lượt mới.", metadata={"source": "potato.txt"})]
        return [Document(page_content="Thông tin ghẻ táo lượt mới.", metadata={"source": "apple.txt"})]


def make_session(responses, **kwargs):
    prompts = []
    replies = iter(responses)
    store = RecordingStore()

    def generate(prompt):
        prompts.append(prompt.to_messages())
        response = next(replies)
        if isinstance(response, Exception):
            raise response
        return AIMessage(content=response)

    session = rag.RAGSession(
        vector_store=store, llm=RunnableLambda(generate), mode="semantic", rerank=False, **kwargs,
    )
    return session, store, prompts


def test_followup_is_rewritten_before_retrieval_and_history_reaches_generation():
    session, store, prompts = make_session([
        "Ghẻ táo gây vết xanh ô liu [Nguồn 1].",
        json.dumps({"query": "Tác nhân gây bệnh ghẻ táo là gì?"}),
        "Tác nhân là Venturia inaequalis [Nguồn 1].",
    ])
    first = session.ask_with_debug("Bệnh ghẻ táo có triệu chứng gì?")
    assert len(prompts) == 1  # Câu đầu không gọi rewrite.
    assert first.history_turns_used == 0
    second = session.ask_with_debug("Bệnh đó do tác nhân nào?")
    assert store.queries == ["Bệnh ghẻ táo có triệu chứng gì?", "Tác nhân gây bệnh ghẻ táo là gì?"]
    assert second.retrieval_query == store.queries[-1]
    assert second.history_turns_used == 1
    assert second.sources == ["apple.txt"]
    assert len(prompts) == 3  # Câu sau thêm đúng một lời gọi rewrite.
    rewrite_prompt, answer_prompt = prompts[1:]
    assert [message.type for message in rewrite_prompt] == ["system", "human", "ai", "human"]
    assert "xanh ô liu" in rewrite_prompt[2].content
    assert answer_prompt[1].content == "Bệnh ghẻ táo có triệu chứng gì?"
    assert "Bệnh đó do tác nhân nào?" in answer_prompt[-1].content
    assert "CÂU HỎI ĐÃ LÀM RÕ:\nTác nhân gây bệnh ghẻ táo là gì?" in answer_prompt[-1].content
    assert "Thông tin ghẻ táo lượt mới" in answer_prompt[-1].content
    assert "không phải bằng chứng" in answer_prompt[0].content
    assert "không tái sử dụng" in answer_prompt[0].content
    assert session.history[-1].question == "Bệnh đó do tác nhân nào?"
    assert session.history[-1].retrieval_query == second.retrieval_query


def test_topic_switch_uses_new_query_and_current_sources():
    session, store, prompts = make_session([
        "Thông tin về táo [Nguồn 9].",
        json.dumps({"query": "Cháy lá sớm trên khoai tây có triệu chứng gì?"}),
        "Thông tin về khoai tây [Nguồn 1].",
    ])
    session.ask("Bệnh ghẻ táo?")
    result = session.ask_with_debug("Cháy lá sớm trên khoai tây có triệu chứng gì?")
    assert "khoai tây" in store.queries[-1]
    assert result.sources == ["potato.txt"]
    current_context = prompts[-1][-1].content
    assert "[Nguồn 1: potato.txt]" in current_context
    assert "apple.txt" not in current_context
    assert result.to_debug_dict()["retrieval_query"] == store.queries[-1]


def test_reset_and_separate_sessions_do_not_leak_history():
    session, _, prompts = make_session(["answer one", "answer after reset"])
    other, _, other_prompts = make_session(["other answer"])
    session.ask("ghẻ táo")
    other.ask("khoai tây")
    assert len(other_prompts) == 1
    assert len(other_prompts[0]) == 2  # system + câu hiện tại, không có lượt của session kia.
    session.clear_history()
    assert session.history == ()
    session.ask("câu hỏi mới")
    assert len(prompts) == 2  # Không rewrite sau reset.
    assert len(prompts[-1]) == 2
    assert other.history[0].question == "khoai tây"


def test_zero_turns_disables_memory_and_rewriting():
    session, store, prompts = make_session(["one", "two"], history_turns=0)
    session.ask("ghẻ táo")
    session.ask("bệnh đó là gì?")
    assert store.queries == ["ghẻ táo", "bệnh đó là gì?"]
    assert len(prompts) == 2
    assert session.history == ()


@pytest.mark.parametrize("bad_rewrite", [
    "   ", "Tác nhân là nấm X.", RuntimeError("rewrite unavailable"),
    json.dumps({"query": ""}), json.dumps({"query": "x" * 2001 + "?"}),
    json.dumps({"query": "Tác nhân là nấm X."}), json.dumps({"query": 123}),
    json.dumps({"answer": "Tác nhân là nấm X."}), json.dumps(["query?"]),
])
def test_failed_rewrite_does_not_retrieve_or_append_history(bad_rewrite):
    session, store, _ = make_session(["first answer", bad_rewrite])
    session.ask("ghẻ táo")
    before = session.history
    with pytest.raises(RuntimeError):
        session.ask("bệnh đó?")
    assert session.history == before
    assert store.queries == ["ghẻ táo"]


def test_failed_generation_does_not_commit_turn_and_retry_uses_previous_history():
    session, store, prompts = make_session([
        "first answer", json.dumps({"query": "Triệu chứng ghẻ táo?"}), RuntimeError("generation failed"),
        json.dumps({"query": "Triệu chứng ghẻ táo?"}), "successful retry",
    ])
    session.ask("ghẻ táo")
    with pytest.raises(RuntimeError, match="generation failed"):
        session.ask("triệu chứng bệnh đó?")
    assert len(session.history) == 1
    session.ask("triệu chứng bệnh đó?")
    assert len(session.history) == 2
    assert len(prompts[3]) == 4  # Lịch sử rewrite retry vẫn chỉ có một lượt hoàn chỉnh.


def test_search_does_not_use_or_change_history_or_call_llm():
    session, store, prompts = make_session(["answer"])
    session.ask("ghẻ táo")
    before = session.history
    session.search("bệnh đó?")
    assert len(prompts) == 1
    assert store.queries[-1] == "bệnh đó?"
    assert session.history == before


def test_memory_keeps_complete_turns_and_resolved_subject_after_eviction():
    memory = ConversationMemory(ChatSettings(history_turns=1))
    memory.add("ghẻ táo", "ghẻ táo", "first answer")
    memory.add("bệnh đó?", "Tác nhân gây bệnh ghẻ táo?", "second answer")
    assert len(memory.turns) == 1
    messages = memory.messages()
    assert [message.type for message in messages] == ["human", "ai"]
    assert "Tác nhân gây bệnh ghẻ táo?" in messages[0].content
    assert "first answer" not in messages[1].content


def test_memory_bounds_chars_even_for_a_single_large_turn():
    memory = ConversationMemory(ChatSettings(history_turns=4, history_max_chars=300))
    memory.add("old question", "old query", "old answer")
    memory.add("q" * 1000, "r" * 1000, "a" * 1000)
    assert len(memory.turns) == 1
    assert sum(turn.char_count for turn in memory.turns) <= 300
    assert memory.turns[0].answer.endswith("…")
    assert len(memory.messages()) == 2


@pytest.mark.parametrize("name,value", [
    ("CHAT_HISTORY_TURNS", "-1"), ("CHAT_HISTORY_TURNS", "many"),
    ("CHAT_HISTORY_MAX_CHARS", "0"), ("CHAT_HISTORY_MAX_CHARS", "299"),
    ("CHAT_HISTORY_MAX_CHARS", "invalid"),
])
def test_memory_config_rejects_invalid_values(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        get_chat_settings()


def test_memory_config_allows_zero_and_explicit_override(monkeypatch):
    monkeypatch.setenv("CHAT_HISTORY_TURNS", "invalid")
    assert get_chat_settings(history_turns=0).history_turns == 0
