from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Intent(str, Enum):
    DIAGNOSIS = "diagnosis"
    TREATMENT = "treatment"
    CAUSE = "cause"
    PREVENTION = "prevention"
    GENERAL_INFO = "general_info"
    OTHER = "other"


class Action(str, Enum):
    ACCEPT_QUERY = "ACCEPT_QUERY"
    REQUEST_IMAGE = "REQUEST_IMAGE"
    ASK_CLARIFICATION = "ASK_CLARIFICATION"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class QueryAnalysis(BaseModel):
    """
    Output bắt buộc của Query Normalizer LLM.

    Class này dùng strict JSON schema với Groq,
    vì vậy tất cả field phải xuất hiện trong JSON.
    Các field có thể không có dữ liệu sẽ nhận giá trị null.
    """

    model_config = ConfigDict(extra="forbid")

    normalized_query: str = Field(
        description=(
            "Câu user đã được sửa typo, viết tắt và chuẩn hóa. "
            "Không tự lấy cây hoặc bệnh từ ảnh/session."
        )
    )

    plant: str | None = Field(
        description=(
            "Tên cây được nêu trực tiếp hoặc được chuẩn hóa "
            "từ query hiện tại. Ví dụ: cây táo -> apple."
        )
    )

    disease: str | None = Field(
        description=(
            "Tên bệnh được nêu trực tiếp hoặc được ánh xạ từ "
            "cách gọi phổ thông theo taxonomy của hệ thống. "
            "Không suy bệnh từ triệu chứng mơ hồ."
        )
    )

    symptoms: list[str] = Field(
        description=(
            "Danh sách triệu chứng được user mô tả. "
            "Nếu không có triệu chứng thì trả về []."
        )
    )

    intent: Intent = Field(
        description="Ý định của câu hỏi hiện tại."
    )

    focus: str | None = Field(
        description=(
            "Chi tiết user đặc biệt quan tâm, ví dụ: "
            "lây sang cây bên cạnh, có cần nhổ cây, "
            "phòng tái phát. Nếu không có thì null."
        )
    )

    refers_to_previous_context: bool = Field(
        description=(
            "True nếu query cần context lượt trước để hiểu, "
            "ví dụ: bệnh này, nó, vậy chữa sao, "
            "còn phòng thế nào."
        )
    )

    is_plant_related: bool = Field(
        description=(
            "True nếu câu hỏi liên quan tới cây trồng, "
            "bệnh cây, chăm sóc cây hoặc triệu chứng trên cây."
        )
    )


class DetectionResult(BaseModel):
    plant: str
    # None khi cây không có model bệnh (Orange, Squash): chỉ nhận diện được cây.
    disease: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    source: str = "plant_ai_pipeline"


class ResolvedQuery(BaseModel):
    plant: Optional[str] = None
    disease: Optional[str] = None
    symptoms: list[str] = Field(default_factory=list)
    intent: Intent
    focus: Optional[str] = None
    context_source: str = "query"


class RouteDecision(BaseModel):
    action: Action
    requires_image: bool = False
    message: Optional[str] = None


class RagDocument(BaseModel):
    id: str
    title: str
    content: str
    source: str
    score: float | None = None


# ======================================================
# HỢP ĐỒNG VỚI RAG SERVER
# Khớp `AnswerRequest` trong 2026-09-23-rag-api-openapi.json. RAG từ chối trường
# lạ (HTTP 422), nên extra="forbid" để lỗi lộ ra ở phía detection trước.
# ======================================================

RAG_QUERY_MAX_CHARS = 2000
RAG_HISTORY_MAX_MESSAGES = 12
RAG_HISTORY_CONTENT_MAX_CHARS = 8000
RAG_SUBJECT_CONTEXT_MAX_CHARS = 1000


class RagChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=RAG_HISTORY_CONTENT_MAX_CHARS)


class RagAnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Câu LLM nhìn thấy và trả lời.
    query: str = Field(min_length=1, max_length=RAG_QUERY_MAX_CHARS)
    # Câu dùng để tìm tài liệu.
    retrieval_query: Optional[str] = Field(
        default=None, min_length=1, max_length=RAG_QUERY_MAX_CHARS
    )
    plant_type: Optional[str] = Field(default=None, max_length=100)
    disease: Optional[str] = Field(default=None, max_length=100)
    history: list[RagChatMessage] = Field(
        default_factory=list, max_length=RAG_HISTORY_MAX_MESSAGES
    )
    subject_context: Optional[str] = Field(
        default=None, max_length=RAG_SUBJECT_CONTEXT_MAX_CHARS
    )


class RagAnswer(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    grounded: bool
    scope_status: str
    # Chỉ GroqAnswerBackend điền: để debug như trước refactor.
    rag_documents: list[RagDocument] = Field(default_factory=list)
    feedback_examples: list[dict] = Field(default_factory=list)


class ChatTurn(BaseModel):
    role: str
    content: str
    image_uploaded: bool = False
    image_filename: Optional[str] = None
    message_id: Optional[str] = None
    feedback_id: Optional[str] = None
    action: Optional[Action] = None
    # Debug payload for this assistant turn. Stored only for development/debug UI.
    debug: dict | None = None
    created_at: Optional[datetime] = None


class SessionSnapshot(BaseModel):
    session_id: str
    last_detection: Optional[DetectionResult] = None
    turns: list[ChatTurn] = Field(default_factory=list)


class PipelineDebug(BaseModel):
    raw_query: str
    normalized_query: str
    intent: Intent

    explicit_plant: Optional[str] = None
    explicit_disease: Optional[str] = None

    symptoms: list[str] = Field(default_factory=list)

    focus: Optional[str] = None

    refers_to_previous_context: bool = False

    detection: Optional[DetectionResult] = None

    resolved_plant: Optional[str] = None
    resolved_disease: Optional[str] = None

    context_source: str = "query"

    action: Action

    retrieval_query: Optional[str] = None

    rag_documents: list[RagDocument] = Field(
        default_factory=list
    )

    # Exact payload sent to Feedback RAG. Keep this contract intentionally small.
    feedback_rag_payload: dict = Field(default_factory=dict)

    # Only the final reranked examples that are allowed into Answer LLM context.
    feedback_rag_examples: list[dict] = Field(default_factory=list)

    # groq = rag/ nội bộ + Groq; rag = RAG-module server.
    answer_backend: Optional[str] = None

    # Payload đã gửi sang AnswerBackend (None với REQUEST_IMAGE/ASK_CLARIFICATION...).
    rag_request: Optional[dict] = None

    rag_sources: list[str] = Field(default_factory=list)
    rag_grounded: Optional[bool] = None
    rag_scope_status: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    action: Action
    memory: Optional[DetectionResult] = None
    debug: PipelineDebug
    feedback_id: str | None = None
    # Trường mới; frontend hiện chỉ đọc 5 trường ở trên và bỏ qua trường này.
    sources: list[str] = Field(default_factory=list)


# ======================================================
# RESPONSE MODELS CHO SWAGGER
# Các document Mongo có thể có thêm trường: extra="allow" để không làm rơi dữ liệu
# frontend đang đọc (feedback_rating, image_filename...).
# ======================================================


class HealthResponse(BaseModel):
    status: str = "ok"


class OkResponse(BaseModel):
    ok: bool = True


class ConversationSummary(BaseModel):
    session_id: str
    title: str
    last_message: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ConversationDetail(BaseModel):
    model_config = ConfigDict(extra="allow")

    session_id: str
    title: Optional[str] = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    last_detection: Optional[DetectionResult] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class FeedbackUpdateResponse(BaseModel):
    success: bool
    feedback_id: str
    rating: str
    reasons: list[str] = Field(default_factory=list)


class AdminStatus(BaseModel):
    answer_backend: str
    feedback_examples_used: bool


class ReviewItem(BaseModel):
    """Một bản ghi chat_feedback; `_id` giữ nguyên tên vì trang admin đọc trường này."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str = Field(alias="_id")
    session_id: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    action: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    user_feedback: dict[str, Any] = Field(default_factory=dict)
    admin_feedback: dict[str, Any] = Field(default_factory=dict)
    training: dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class DeleteReviewResponse(BaseModel):
    success: bool
    deleted_id: str


class TrainingResponse(BaseModel):
    success: bool
    trained_count: int
    indexed_count: int
    mode: str


class TrainingExample(BaseModel):
    """Một ví dụ trong feedback_training_vectors (không kèm embedding)."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str = Field(alias="_id")
    feedback_id: str
    question: str
    preferred_answer: str
    planttype: Optional[str] = None
    disease: Optional[str] = None
    rating: Optional[str] = None
    reasons: list[str] = Field(default_factory=list)
    active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
