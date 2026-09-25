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
from app.schemas import (
    Action,
    ChatTurn,
    DetectionResult,
    Intent,
    QueryAnalysis,
    RagAnswer,
    SourceDocument,
    SourceLink,
)

OPENAPI_FILE = Path(__file__).resolve().parents[2] / "2026-09-23-rag-api-openapi.json"

APPLE_SCAB = DetectionResult(plant="Apple", disease="Apple___Apple_scab", confidence=0.93)
APPLE_SUBJECT = "plant=Apple; disease=Apple___Apple_scab; confidence=0.93; source=plant_ai_pipeline"


def analysis(**values) -> QueryAnalysis:
    base = dict(
        normalized_query=values.get("normalized_query", "q"),
        plant=None, disease=None, disease_named=values.get("disease") is not None,
        symptoms=[], intent=Intent.TREATMENT, focus=None,
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
    def __init__(self, detection=APPLE_SCAB, error=None, search_original_query=False):
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
            query_builder=RetrievalQueryBuilder(search_original_query=search_original_query),
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


def test_text_only_payload_searches_with_normalized_query():
    h = Harness()
    response = h.chat(
        "Bệnh ghẻ táo xử lý thế nào?",
        analysis(normalized_query="Bệnh ghẻ táo xử lý thế nào?", plant="apple", disease="apple_scab"),
    )
    # Câu gốc trùng câu chuẩn hóa: không gửi extra_queries.
    assert h.payload() == {
        "query": "Bệnh ghẻ táo xử lý thế nào?",
        "retrieval_query": "Bệnh ghẻ táo xử lý thế nào?",
        "plant_type": "apple",
        "disease": "apple_scab",
        "history": [],
    }
    assert response.answer == "Trả lời."
    assert response.sources == ["apple/apple_scab.txt"]
    assert h.feedback[0]["metadata"]["sources"] == ["apple/apple_scab.txt"]
    assert h.feedback[0]["metadata"]["scope_status"] == "document"


def test_chosen_model_and_source_documents_are_kept_with_the_turn():
    h = Harness()
    documents = [SourceDocument(source="apple/apple_scab.txt", title="Bệnh ghẻ táo",
                                links=[SourceLink(label="UMN", url="https://extension.umn.edu/x")])]

    async def answer(request, context=None):
        h.backend.requests.append(request)
        return RagAnswer(answer="Trả lời.", sources=["apple/apple_scab.txt"], documents=documents,
                         grounded=True, scope_status="document",
                         llm_provider="gemini", llm_model="gemini-2.5-flash")

    h.backend.answer = answer
    h.normalizer.next = analysis(plant="apple", disease="apple_scab")
    response = asyncio.run(h.service.chat(session_id="s1", raw_query="ghẻ táo?", llm_provider="gemini"))
    assert h.payload()["llm_provider"] == "gemini"
    assert response.source_documents == documents
    assert response.debug.llm_model == "gemini-2.5-flash"
    assert h.feedback[0]["metadata"]["llm_provider"] == "gemini"
    # Mở lại hội thoại: lượt đã lưu vẫn có tài liệu + link để web hiển thị.
    stored = asyncio.run(h.sessions.snapshot("s1")).turns[-1].debug
    assert stored["source_documents"][0]["links"][0]["url"] == "https://extension.umn.edu/x"


def test_raw_query_is_not_an_extra_search_query_by_default():
    """Đo 24/09: gửi kèm câu gốc không giúp retrieval, nên tắt mặc định."""
    h = Harness()
    h.chat("ghe tao tri sao", analysis(normalized_query="Ghẻ táo trị thế nào?",
                                        plant="apple", disease="apple_scab"))
    assert "extra_queries" not in h.payload()


def test_raw_query_is_sent_as_extra_search_query_when_enabled_and_different():
    h = Harness(search_original_query=True)
    h.chat(
        "benh ghe tao xu ly sao, co can nho cay ko",
        analysis(normalized_query="Bệnh ghẻ táo xử lý thế nào, có cần nhổ cây không?",
                 plant="apple", disease="apple_scab"),
    )
    payload = h.payload()
    assert payload["query"] == "benh ghe tao xu ly sao, co can nho cay ko"
    assert payload["retrieval_query"] == "Bệnh ghẻ táo xử lý thế nào, có cần nhổ cây không?"
    assert payload["extra_queries"] == ["benh ghe tao xu ly sao, co can nho cay ko"]

    h.chat("Ghẻ táo trị thế nào?", analysis(normalized_query="ghẻ táo  trị thế nào?",
                                              plant="apple", disease="apple_scab"))
    assert "extra_queries" not in h.payload()


def test_image_payload_sends_raw_labels_and_normalized_search_query():
    h = Harness()
    h.chat("Lá nhà tôi bị vậy có cần nhổ cây không?",
           analysis(normalized_query="Lá nhà tôi bị như vậy có cần nhổ cây không?"), image=b"img")
    # Cây/bệnh của ảnh chỉ đi trong plant_type/disease (RAG khoanh đúng tài liệu),
    # không gắn vào câu tìm: đo 24/09 thấy gắn tên làm thứ hạng trong tài liệu kém đi.
    assert h.payload() == {
        "query": "Lá nhà tôi bị vậy có cần nhổ cây không?",
        "retrieval_query": "Lá nhà tôi bị như vậy có cần nhổ cây không?",
        "plant_type": "Apple",
        "disease": "Apple___Apple_scab",
        "subject_context": APPLE_SUBJECT,
        "history": [],
    }


def test_image_only_uses_default_question():
    h = Harness()
    h.chat("", analysis(normalized_query="Ảnh này đang bị bệnh gì?", intent=Intent.DIAGNOSIS),
           image=b"img")
    payload = h.payload()
    assert payload["query"] == "Ảnh này đang bị bệnh gì?"
    assert "extra_queries" not in payload


def test_follow_up_carries_session_detection_and_history():
    h = Harness()
    h.chat("Lá táo nhà tôi bị vậy có cần nhổ cây không?", analysis(intent=Intent.TREATMENT), image=b"img")
    h.chat("Vậy phòng bệnh này thế nào?",
           analysis(normalized_query="Vậy phòng bệnh này thế nào?",
                    intent=Intent.PREVENTION, refers_to_previous_context=True))
    payload = h.payload()
    assert payload["retrieval_query"] == "Vậy phòng bệnh này thế nào?"
    assert payload["plant_type"] == "Apple"
    assert payload["disease"] == "Apple___Apple_scab"
    assert payload["subject_context"] == APPLE_SUBJECT
    assert payload["history"] == [
        {"role": "user", "content": "Lá táo nhà tôi bị vậy có cần nhổ cây không?"},
        {"role": "assistant", "content": "Trả lời."},
    ]
    assert "rewrite_query" not in payload


def test_guessed_disease_does_not_narrow_rag_scope():
    """"Bệnh đốm trên cây táo" → black_rot chỉ là quy ước: RAG tìm trong mọi tài liệu táo."""
    h = Harness()
    response = h.chat(
        "cach chua benh dom tren cay tao",
        analysis(normalized_query="Cách chữa bệnh đốm trên cây táo", plant="apple",
                 disease="black_rot", disease_named=False),
    )
    assert response.action == Action.ACCEPT_QUERY
    payload = h.payload()
    assert payload["plant_type"] == "apple" and "disease" not in payload
    assert payload["retrieval_query"] == "Cách chữa bệnh đốm trên cây táo"
    assert response.debug.suspected_disease == "black_rot"
    assert response.debug.resolved_disease is None


def test_image_disease_replaces_guessed_disease():
    h = Harness()
    response = h.chat(
        "bệnh đốm trên lá táo này chữa sao",
        analysis(normalized_query="Bệnh đốm trên lá táo này chữa thế nào?", plant="apple",
                 disease="black_rot", disease_named=False),
        image=b"img",
    )
    payload = h.payload()
    assert payload["disease"] == "Apple___Apple_scab"
    assert payload["retrieval_query"] == "Bệnh đốm trên lá táo này chữa thế nào?"
    assert response.debug.suspected_disease is None


def test_guessed_disease_without_plant_still_asks_for_clarification():
    h = Harness()
    response = h.chat("bệnh đốm chữa sao",
                      analysis(disease="black_rot", disease_named=False))
    assert response.action == Action.ASK_CLARIFICATION
    assert h.backend.requests == []


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


TOMATO_HEALTHY = DetectionResult(plant="Tomato", disease="healthy", confidence=0.97)


@pytest.mark.parametrize("message", ["Lá cây này có đang bị gì không?", ""])
def test_healthy_leaf_is_answered_without_rag(message):
    """Ảnh lá khỏe + hỏi có bệnh không (hoặc chỉ gửi ảnh): RAG chỉ có tài liệu bệnh nên không gọi."""
    h = Harness(detection=TOMATO_HEALTHY)
    response = h.chat(message, analysis(intent=Intent.DIAGNOSIS), image=b"img")
    assert h.backend.requests == []
    assert response.action == Action.HEALTHY_PLANT
    assert "lá cà chua khỏe mạnh" in response.answer and "97%" in response.answer
    assert h.feedback[0]["metadata"]["action"] == "HEALTHY_PLANT"
    assert response.memory == TOMATO_HEALTHY


def test_unsure_healthy_result_asks_for_a_better_photo():
    h = Harness(detection=DetectionResult(plant="Apple", disease="Apple___healthy", confidence=0.62))
    response = h.chat("", analysis(intent=Intent.DIAGNOSIS), image=b"img")
    assert response.action == Action.HEALTHY_PLANT
    assert "chưa chắc chắn" in response.answer and "lá táo" in response.answer


def test_healthy_leaf_prevention_question_still_uses_rag():
    h = Harness(detection=TOMATO_HEALTHY)
    response = h.chat("Cây này phòng bệnh thế nào?", analysis(intent=Intent.PREVENTION), image=b"img")
    assert response.action == Action.ACCEPT_QUERY
    assert h.payload()["disease"] == "healthy"


def test_disease_named_by_user_beats_healthy_image():
    h = Harness(detection=TOMATO_HEALTHY)
    response = h.chat("Lá này có bị mốc sương không?",
                      analysis(plant="tomato", disease="late_blight", intent=Intent.DIAGNOSIS), image=b"img")
    assert response.action == Action.ACCEPT_QUERY
    assert h.payload()["disease"] == "late_blight"


def test_follow_up_about_healthy_leaf_is_answered_without_rag():
    h = Harness(detection=TOMATO_HEALTHY)
    h.chat("", analysis(intent=Intent.DIAGNOSIS), image=b"img")
    response = h.chat("Vậy cây đó có bị bệnh gì không?",
                      analysis(intent=Intent.DIAGNOSIS, refers_to_previous_context=True))
    assert response.action == Action.HEALTHY_PLANT and h.backend.requests == []


def test_healthy_image_only_is_answered_even_when_groq_fails():
    h = Harness(detection=TOMATO_HEALTHY)
    h.normalizer.analyze = broken_normalizer
    response = asyncio.run(h.service.chat(session_id="s1", raw_query="", image_bytes=b"img"))
    assert response.action == Action.HEALTHY_PLANT and h.backend.requests == []


def test_leaf_image_is_never_out_of_scope():
    """Đo 26/09: chỉ gửi ảnh, Groq thấy "Ảnh này đang bị bệnh gì?" và báo is_plant_related=false."""
    h = Harness()
    response = h.chat("", analysis(intent=Intent.DIAGNOSIS, is_plant_related=False), image=b"img")
    assert response.action == Action.ACCEPT_QUERY
    assert h.payload()["disease"] == "Apple___Apple_scab"
    # Không có ảnh thì vẫn từ chối câu ngoài phạm vi như cũ.
    assert h.chat("Giá vàng hôm nay?", analysis(is_plant_related=False)).action == Action.OUT_OF_SCOPE


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


async def broken_normalizer(raw_query, session_context):
    raise RuntimeError("Groq down")


def test_normalizer_error_sends_raw_query_to_rag():
    """Groq lỗi: không trả 500; RAG tìm bằng câu gốc và tự viết lại câu nối tiếp."""
    h = Harness()
    h.normalizer.analyze = broken_normalizer
    response = asyncio.run(h.service.chat(session_id="s1", raw_query="ghe tao chua sao"))
    assert response.action == Action.ACCEPT_QUERY
    assert response.debug.normalizer_failed is True
    assert h.payload() == {"query": "ghe tao chua sao", "rewrite_query": True, "history": []}


def test_normalizer_error_with_image_uses_detection_and_completes_the_turn():
    h = Harness()
    h.normalizer.analyze = broken_normalizer
    asyncio.run(h.service.chat(session_id="s1", raw_query="", image_bytes=b"img"))
    payload = h.payload()
    assert payload["query"] == "Ảnh này đang bị bệnh gì?" and payload["rewrite_query"] is True
    assert payload["plant_type"] == "Apple" and payload["disease"] == "Apple___Apple_scab"
    assert "retrieval_query" not in payload
    assert asyncio.run(h.sessions.get_last_detection("s1")) == APPLE_SCAB
