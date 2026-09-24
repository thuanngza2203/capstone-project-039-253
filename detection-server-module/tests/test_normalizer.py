"""GroqQueryNormalizer với client giả: tham số gửi Groq và cách đọc JSON trả về."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from app.llm.groq import GroqQueryNormalizer

ANSWER = {
    "normalized_query": "Bệnh ghẻ táo chữa thế nào?", "plant": "apple", "disease": "apple_scab",
    "disease_named": True, "symptoms": [], "intent": "treatment", "focus": None,
    "refers_to_previous_context": False, "is_plant_related": True,
}


class FakeCompletions:
    def __init__(self, content: str):
        self.content = content
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def normalizer(content: str, **kwargs) -> tuple[GroqQueryNormalizer, FakeCompletions]:
    instance = GroqQueryNormalizer(api_key="test", model="openai/gpt-oss-120b", **kwargs)
    completions = FakeCompletions(content)
    instance.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return instance, completions


def test_default_reasoning_effort_is_accepted_by_gpt_oss():
    """gpt-oss trên Groq chỉ nhận low/medium/high; "none" làm mọi lượt chat lỗi 400."""
    instance, completions = normalizer(json.dumps(ANSWER))
    analysis = asyncio.run(instance.analyze(raw_query="ghe tao chua sao", session_context=""))
    assert completions.calls[0]["reasoning_effort"] == "low"
    assert analysis.disease == "apple_scab" and analysis.disease_named is True


def test_reasoning_effort_can_be_omitted_for_models_without_it():
    instance, completions = normalizer(json.dumps(ANSWER), reasoning_effort=None)
    asyncio.run(instance.analyze(raw_query="x", session_context=""))
    assert "reasoning_effort" not in completions.calls[0]


def test_answer_missing_disease_named_is_rejected():
    """Mọi trường đều bắt buộc: thiếu trường thì pipeline chuyển sang câu gốc (fallback)."""
    incomplete = {key: value for key, value in ANSWER.items() if key != "disease_named"}
    instance, _ = normalizer("```json\n" + json.dumps(incomplete) + "\n```")
    with pytest.raises(RuntimeError, match="QueryAnalysis"):
        asyncio.run(instance.analyze(raw_query="x", session_context=""))
