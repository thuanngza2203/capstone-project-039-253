"""ChatService với normalizer/detector/backend giả: kiểm tra payload gửi sang RAG."""

import asyncio
import json
from pathlib import Path

import pytest

from app.answer.base import AnswerBackend, AnswerBackendError
from app.chat.context import ContextResolver
from app.chat.pipeline import ChatService
from app.chat.query import RetrievalQueryBuilder
from app.chat.routing import QueryRouter
from app.chat.session import InMemorySessionStore
from app.llm.base import QueryNormalizerLLM
from app.schemas import Action, ChatTurn, DetectionResult, Intent, QueryAnalysis, RagAnswer

OPENAPI_FILE = Path(__file__).resolve().parents[2] / "2026-09-23-rag-api-openapi.json"

APPLE_SCAB = DetectionResult(plant="Apple", disease="Apple___Apple_scab", confidence=0.93)
APPLE_SUBJECT = "plant=Apple; disease=Apple___Apple_scab; confidence=0.93; source=plant_ai_pipeline"


def analysis(**values) -> QueryAnalysis:
    base = dict(
        normalized_query=values.get("normalized_query", "q"),
        plant=None, disease=None, symptoms=[], intent=Intent.TREATMENT, focus=None,
        refers_to_previous_context=False, is_plant_related=True,
    )
    base.update(values)
    return QueryAnalysis(**base)


class FakeNormalizer(QueryNormalizerLLM):
    def __init__(self):
        self.next: QueryAnalysis | None = None

    async def analyze(self, raw_query, session_context):
        return self.next


class FakeDetector:
    def __init__(self, result: DetectionResult):
        self.result = result

    async def detect(self, image_bytes, filename=None):
        return self.result


class RecordingBackend(AnswerBackend):
    name = "fake"

    def __init__(self, error: Exception | None = None):
        self.requests = []
        self.error = error

    async def answer(self, request, context=None):
        self.requests.append(request)
        if self.error:
            raise self.error
        return RagAnswer(answer="Trả lời.", sources=["apple/apple_scab.txt"], grounded=True,
                         scope_status="document")


class Harness:
    def __init__(self, detection=APPLE_SCAB, error=None):
        self.normalizer = FakeNormalizer()
        self.backend = RecordingBackend(error)
        self.sessions = InMemorySessionStore()
        self.feedback = []

        async def record(**kwargs):
            self.feedback.append(kwargs)
            return f"fb{len(self.feedback)}"

        self.service = ChatService(
            normalizer=self.normalizer,
            answer_backend=self.backend,
            detector=FakeDetector(detection),
            sessions=self.sessions,
            resolver=ContextResolver(),
            router=QueryRouter(),
            query_builder=RetrievalQueryBuilder(),
            feedback_recorder=record,
        )

    def chat(self, message, result: QueryAnalysis, image: bytes | None = None):
        self.normalizer.next = result
        return asyncio.run(self.service.chat(
            session_id="s1", raw_query=message, image_bytes=image,
            image_filename="leaf.jpg" if image else None,
        ))

    def payload(self, index=-1) -> dict:
        return self.backend.requests[index].model_dump(mode="json", exclude_none=True)


def test_text_only_payload():
    h = Harness()
    response = h.chat(
        "Bệnh ghẻ táo xử lý thế nào?",
        analysis(plant="apple", disease="apple_scab", intent=Intent.TREATMENT),
    )
    assert h.payload() == {
        "query": "Bệnh ghẻ táo xử lý thế nào?",
        "retrieval_query": "Cách điều trị bệnh apple_scab trên cây apple",
        "plant_type": "apple",
        "disease": "apple_scab",
        "history": [],
    }
    assert response.answer == "Trả lời."
    assert response.sources == ["apple/apple_scab.txt"]
    assert h.feedback[0]["metadata"]["sources"] == ["apple/apple_scab.txt"]
    assert h.feedback[0]["metadata"]["scope_status"] == "document"


def test_image_payload_sends_raw_labels_and_canonical_retrieval_query():
    h = Harness()
    h.chat("Lá táo nhà tôi bị vậy có cần nhổ cây không?", analysis(intent=Intent.TREATMENT), image=b"img")
    assert h.payload() == {
        "query": "Lá táo nhà tôi bị vậy có cần nhổ cây không?",
        "retrieval_query": "Cách điều trị bệnh apple_scab trên cây apple",
        "plant_type": "Apple",
        "disease": "Apple___Apple_scab",
        "subject_context": APPLE_SUBJECT,
        "history": [],
    }


def test_image_only_uses_default_question():
    h = Harness()
    h.chat("", analysis(intent=Intent.DIAGNOSIS), image=b"img")
    assert h.payload()["query"] == "Ảnh này đang bị bệnh gì?"


def test_follow_up_carries_session_detection_and_history():
    h = Harness()
    h.chat("Lá táo nhà tôi bị vậy có cần nhổ cây không?", analysis(intent=Intent.TREATMENT), image=b"img")
    h.chat("Vậy phòng bệnh này thế nào?",
           analysis(intent=Intent.PREVENTION, refers_to_previous_context=True))
    payload = h.payload()
    assert payload["retrieval_query"] == "Cách phòng ngừa bệnh apple_scab trên cây apple"
    assert payload["plant_type"] == "Apple"
    assert payload["disease"] == "Apple___Apple_scab"
    assert payload["subject_context"] == APPLE_SUBJECT
    assert payload["history"] == [
        {"role": "user", "content": "Lá táo nhà tôi bị vậy có cần nhổ cây không?"},
        {"role": "assistant", "content": "Trả lời."},
    ]
    assert "rewrite_query" not in payload


def test_request_image_does_not_call_backend():
    h = Harness()
    response = h.chat("Lá cà chua có đốm, bệnh gì?",
                      analysis(plant="tomato", intent=Intent.DIAGNOSIS, symptoms=["đốm"]))
    assert response.action == Action.REQUEST_IMAGE
    assert h.backend.requests == []
    assert h.feedback[0]["metadata"]["sources"] == []
    assert response.debug.rag_request is None


def test_backend_error_saves_nothing():
    h = Harness(error=AnswerBackendError(503, "down"))
    with pytest.raises(AnswerBackendError):
        h.chat("Bệnh ghẻ táo xử lý thế nào?", analysis(plant="apple", disease="apple_scab"))
    snapshot = asyncio.run(h.sessions.snapshot("s1"))
    assert snapshot.turns == []
    assert h.feedback == []


def test_plant_without_disease_model_still_completes_the_turn():
    h = Harness(detection=DetectionResult(plant="Orange", disease=None, confidence=0.8))
    response = h.chat("", analysis(intent=Intent.DIAGNOSIS), image=b"img")
    assert response.memory.plant == "Orange"
    assert response.memory.disease is None
    assert len(asyncio.run(h.sessions.snapshot("s1")).turns) == 2


def test_long_history_is_clipped_to_contract_limits():
    h = Harness()
    for index in range(8):
        role = "user" if index % 2 == 0 else "assistant"
        asyncio.run(h.sessions.add_turn("s1", ChatTurn(role=role, content="x" * 9000)))
    h.chat("Ghẻ táo trị sao?", analysis(plant="apple", disease="apple_scab"))
    history = h.payload()["history"]
    assert len(history) <= 12
    assert all(len(message["content"]) == 8000 for message in history)


@pytest.mark.skipif(not OPENAPI_FILE.is_file(), reason="Thiếu file OpenAPI của RAG")
def test_payload_fields_are_allowed_by_rag_contract():
    schema = json.loads(OPENAPI_FILE.read_text(encoding="utf-8"))["components"]["schemas"]["AnswerRequest"]
    h = Harness()
    h.chat("Lá táo nhà tôi bị vậy có cần nhổ cây không?", analysis(intent=Intent.TREATMENT), image=b"img")
    h.chat("Vậy phòng bệnh này thế nào?",
           analysis(intent=Intent.PREVENTION, refers_to_previous_context=True))
    for request in h.backend.requests:
        payload = request.model_dump(mode="json", exclude_none=True)
        assert set(payload) <= set(schema["properties"])
        assert set(schema["required"]) <= set(payload)


def test_invalid_image_stops_before_normalizer_and_saves_nothing():
    from app.plant_ai.pipeline.prediction import InvalidImageError

    h = Harness()

    class Rejecting:
        async def detect(self, image_bytes, filename=None):
            raise InvalidImageError("Ảnh không hợp lệ: mình không thấy lá cây nào trong ảnh.")

    async def must_not_normalize(raw_query, session_context):
        raise AssertionError("Không được gọi Groq khi ảnh không hợp lệ")

    h.service.detector = Rejecting()
    h.normalizer.analyze = must_not_normalize
    with pytest.raises(InvalidImageError):
        asyncio.run(h.service.chat(session_id="s1", raw_query="", image_bytes=b"img"))
    snapshot = asyncio.run(h.sessions.snapshot("s1"))
    assert snapshot.turns == [] and snapshot.last_detection is None
    assert h.feedback == [] and h.backend.requests == []


def test_backend_error_after_image_saves_no_detection():
    h = Harness(error=AnswerBackendError(503, "down"))
    with pytest.raises(AnswerBackendError):
        h.chat("Lá táo bị gì?", analysis(intent=Intent.TREATMENT), image=b"img")
    snapshot = asyncio.run(h.sessions.snapshot("s1"))
    assert snapshot.turns == [] and snapshot.last_detection is None


def test_normalizer_error_after_image_saves_no_detection():
    h = Harness()

    async def broken(raw_query, session_context):
        raise RuntimeError("Groq down")

    h.normalizer.analyze = broken
    with pytest.raises(RuntimeError):
        asyncio.run(h.service.chat(session_id="s1", raw_query="", image_bytes=b"img"))
    assert asyncio.run(h.sessions.get_last_detection("s1")) is None
