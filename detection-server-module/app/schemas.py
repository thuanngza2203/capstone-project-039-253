from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal, Optional

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
    # Người dùng bật "Tìm trên web": Groq tìm web và trả lời, không qua normalizer/router/RAG.
    WEB_SEARCH = "WEB_SEARCH"
    # Ảnh được nhận diện là lá khỏe và người dùng hỏi cây có bệnh không: trả lời luôn, không gọi RAG.
    HEALTHY_PLANT = "HEALTHY_PLANT"


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
            "Câu user đã được sửa dấu, typo, viết tắt; giữ đủ chi tiết user hỏi. "
            "Dùng làm câu tìm tài liệu. Không tự lấy cây hoặc bệnh từ ảnh/session."
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

    disease_named: bool = Field(
        description=(
            "True nếu user gọi đúng tên bệnh (tiếng Việt, tiếng Anh, tên khoa học). "
            "False nếu disease là null hoặc chỉ suy từ cách gọi chung chung theo "
            "quy ước, ví dụ 'bệnh đốm trên cây táo' -> black_rot."
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
    # Bệnh Groq đoán từ cách gọi chung chung (disease_named=false) khi không có bệnh
    # nào khác. Chỉ để log: không khoanh phạm vi tìm, không gửi sang RAG.
    suspected_disease: Optional[str] = None
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
RAG_EXTRA_QUERIES_MAX = 3
RAG_HISTORY_MAX_MESSAGES = 12
RAG_HISTORY_CONTENT_MAX_CHARS = 8000
RAG_SUBJECT_CONTEXT_MAX_CHARS = 1000

# LLM sinh câu trả lời bên RAG (`llm_provider` của RAG API).
LLM_PROVIDERS = ("vllm", "gemini", "ollama")
LlmProvider = Literal["vllm", "gemini", "ollama"]

RagQueryText = Annotated[str, Field(min_length=1, max_length=RAG_QUERY_MAX_CHARS)]

# Câu trả lời mẫu admin đã duyệt gửi kèm sang RAG (giới hạn theo AnswerRequest của RAG).
RAG_FEEDBACK_EXAMPLES_MAX = 3
RAG_FEEDBACK_QUESTION_MAX_CHARS = 1000
RAG_FEEDBACK_ANSWER_MAX_CHARS = 3000


class RagFeedbackExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=RAG_FEEDBACK_QUESTION_MAX_CHARS)
    answer: str = Field(min_length=1, max_length=RAG_FEEDBACK_ANSWER_MAX_CHARS)


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
    # Câu tìm bổ sung (câu gốc khi khác câu chuẩn hóa); RAG gộp kết quả bằng RRF.
    # None thay cho [] để payload không đổi khi không dùng.
    extra_queries: Optional[list[RagQueryText]] = Field(
        default=None, max_length=RAG_EXTRA_QUERIES_MAX
    )
    # Chỉ bật khi Groq lỗi: RAG tự viết lại câu hỏi nối tiếp theo lịch sử.
    # RAG không nhận cùng lúc với retrieval_query.
    rewrite_query: Optional[bool] = None
    plant_type: Optional[str] = Field(default=None, max_length=100)
    disease: Optional[str] = Field(default=None, max_length=100)
    history: list[RagChatMessage] = Field(
        default_factory=list, max_length=RAG_HISTORY_MAX_MESSAGES
    )
    subject_context: Optional[str] = Field(
        default=None, max_length=RAG_SUBJECT_CONTEXT_MAX_CHARS
    )
    # Model người dùng chọn trên web; None = LLM_PROVIDER mặc định bên RAG.
    llm_provider: Optional[LlmProvider] = None
    # RagHttpBackend điền từ Feedback RAG; None thay cho [] để payload không đổi khi không có.
    feedback_examples: Optional[list[RagFeedbackExample]] = Field(
        default=None, max_length=RAG_FEEDBACK_EXAMPLES_MAX
    )


class SourceLink(BaseModel):
    label: Optional[str] = None
    url: str


class SourceDocument(BaseModel):
    """Tài liệu đã dùng để trả lời: file trong kho và link nguồn tham khảo ghi trong file."""

    source: str
    title: str
    links: list[SourceLink] = Field(default_factory=list)


class RagAnswer(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    documents: list[SourceDocument] = Field(default_factory=list)
    # Trang web đã đọc khi trả lời bằng "Tìm trên web".
    web_sources: list[SourceLink] = Field(default_factory=list)
    grounded: bool
    scope_status: str
    # Model đã sinh câu trả lời; None khi không gọi LLM (câu từ chối dựng sẵn).
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
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
    # None khi tìm trên web (không chạy normalizer).
    intent: Optional[Intent] = None

    explicit_plant: Optional[str] = None
    explicit_disease: Optional[str] = None
    disease_named: Optional[bool] = None
    suspected_disease: Optional[str] = None

    # Groq lỗi: câu gốc được gửi thẳng sang RAG (rewrite_query=true), không qua router.
    normalizer_failed: bool = False

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

    # Tên tài liệu + link nguồn; lưu cùng lượt để mở lại hội thoại vẫn hiện được.
    source_documents: list[SourceDocument] = Field(default_factory=list)
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None

    web_search: bool = False
    web_sources: list[SourceLink] = Field(default_factory=list)


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    action: Action
    memory: Optional[DetectionResult] = None
    debug: PipelineDebug
    feedback_id: str | None = None
    sources: list[str] = Field(default_factory=list)
    source_documents: list[SourceDocument] = Field(default_factory=list)
    web_sources: list[SourceLink] = Field(default_factory=list)


class ModelOptions(BaseModel):
    """Model trả lời người dùng chọn được ở trang chat."""

    answer_backend: str = Field(description="`rag` hoặc `groq`; với `groq` không có lựa chọn nào.")
    providers: list[LlmProvider] = Field(description="Theo thứ tự hiển thị; rỗng = không cho chọn.")
    default: Optional[LlmProvider] = Field(None, description="Model chọn sẵn (phần tử đầu).")
    web_search: bool = Field(False, description="Có nút \"Tìm trên web\" (WEB_SEARCH_MODEL khác rỗng).")


class AdminSession(BaseModel):
    token: str = Field(description="Gửi lại trong header `Authorization: Bearer <token>`.")
    username: str
    expires_at: datetime


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
