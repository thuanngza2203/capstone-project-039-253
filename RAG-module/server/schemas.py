"""Schema request/response của RAG API; mô tả và ví dụ ở đây hiện thẳng lên Swagger."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, StringConstraints, model_validator

ScopeStatus = Literal[
    "none", "unknown_plant", "crop", "healthy", "document",
    "disease_multi_crop", "unsupported_disease", "unknown_disease",
]

SCOPE_STATUS_MEANING: dict[str, str] = {
    "none": "Không gửi cây/bệnh: tìm trên toàn bộ tài liệu.",
    "unknown_plant": "Tên cây không có trong taxonomy: tìm trên toàn bộ tài liệu.",
    "crop": "Chỉ có cây: tìm trong các tài liệu của cây đó.",
    "healthy": "Cây được báo là khỏe: tìm trong tài liệu của cây (thông tin phòng bệnh).",
    "document": "Cây + bệnh có tài liệu: chỉ tìm trong đúng tài liệu đó.",
    "disease_multi_crop": "Chỉ có bệnh và bệnh thuộc nhiều cây: tìm trong các tài liệu của bệnh.",
    "unsupported_disease": "Bệnh có trong taxonomy nhưng kho chưa có tài liệu: không tìm, không gọi LLM.",
    "unknown_disease": "Không nhận ra nhãn bệnh: không tìm, không gọi LLM; cần bổ sung alias.",
}

IndexName = Literal["recursive", "structure"]
ProviderName = Literal["ollama", "gemini", "vllm"]

Query = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
Label = Annotated[str, StringConstraints(strip_whitespace=True, max_length=100)]
MessageText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)]


class ErrorResponse(BaseModel):
    detail: str = Field(description="Lý do lỗi, đọc được bằng tiếng Việt.",
                        examples=["Thiếu hoặc sai API key."])


class HealthResponse(BaseModel):
    status: Literal["ok"] = Field("ok", description="Process còn sống. Không kiểm tra index hay LLM.")


class Scope(BaseModel):
    """Phạm vi tìm kiếm suy ra từ `plant_type` và `disease`."""

    status: ScopeStatus = Field(description="Xem bảng trạng thái trong mô tả API.")
    searchable: bool = Field(description="False nghĩa là không tìm tài liệu và không gọi LLM.")
    plant: str | None = Field(None, description="Key chuẩn của cây, ví dụ `apple`.")
    disease: str | None = Field(None, description="Key chuẩn của bệnh, ví dụ `apple_scab`.")
    crop: str | None = Field(None, description="Thư mục cây trong kho tài liệu (`pepper_bell` cho `pepper`).")
    sources: list[str] = Field(default_factory=list, description="Tài liệu được phép tìm; rỗng = không giới hạn theo tài liệu.")
    message: str = Field(description="Giải thích phạm vi, dành cho log và giao diện debug.")
    received: dict[str, str | None] = Field(default_factory=dict, description="Nhãn gốc đã nhận, để đối chiếu khi alias sai.")


class RetrieveRequest(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        json_schema_extra={"examples": [
            {"query": "Bệnh ghẻ táo xử lý như thế nào?", "plant_type": "apple", "disease": "apple_scab"},
            {"query": "Triệu chứng đốm vi khuẩn", "disease": "bacterial_spot", "top_k": 4},
            {"query": "Corynespora cassiicola gây bệnh gì?"},
        ]},
    )

    query: Query = Field(description=(
        "Câu hỏi đã chuẩn hóa. Khi gọi từ detection-server, dùng `retrieval_query` "
        "(hoặc `normalized_query`) sau khi đã giải quyết \"bệnh này\", \"nó\"."
    ))
    plant_type: Label | None = Field(
        None,
        validation_alias=AliasChoices("plant_type", "planttype"),
        description=(
            "Cây từ normalizer hoặc detector: `apple`, `Apple`, `tomato`, `cà chua`... "
            "Nhận cả tên cũ `planttype`. Bỏ trống thì không giới hạn theo cây."
        ),
    )
    disease: Label | None = Field(None, description=(
        "Nhãn bệnh: `apple_scab`, `Apple___Black_rot`, `Scab`, `healthy`... "
        "Tra qua bảng alias ở `GET /v1/taxonomy`."
    ))
    top_k: int | None = Field(None, ge=1, le=20, description="Số chunk trả về. Mặc định RAG_TOP_K (4).")
    mode: Literal["semantic", "bm25", "hybrid"] | None = Field(
        None, description="Ghi đè RETRIEVAL_MODE cho request này.")
    rerank: bool | None = Field(None, description="Ghi đè RERANKER_ENABLED cho request này.")
    index: IndexName | None = Field(None, description=(
        "Index theo cách chunk, để so sánh `recursive` và `structure`. Bỏ trống = index mặc "
        "định (CHUNKING_STRATEGY). Luồng detection không cần gửi."
    ))
    extra_queries: list[Query] = Field(default_factory=list, max_length=3, description=(
        "Câu tìm bổ sung, tìm y như câu chính rồi gộp mọi kết quả bằng RRF, ví dụ câu gốc người "
        "dùng gõ khi câu chính là câu đã chuẩn hóa. Câu trùng câu chính (khác hoa/thường, khoảng "
        "trắng) bị bỏ. Detection chỉ gửi khi bật RAG_SEARCH_ORIGINAL_QUERY."
    ))
    debug: bool = Field(False, description="Trả thêm toàn bộ ứng viên và điểm từng nhánh.")


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"] = Field(description="Ai nói lượt này.")
    content: MessageText = Field(description="Nội dung lượt đó.")


class AnswerRequest(RetrieveRequest):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        json_schema_extra={"examples": [
            {
                "query": "la tao nha toi bi vay co can nho cay ko",
                "retrieval_query": "Lá táo nhà tôi bị như vậy có cần nhổ cây không?",
                "plant_type": "apple",
                "disease": "apple_scab",
                "subject_context": "plant=Apple; disease=Scab; confidence=0.93; source=plant_ai_pipeline",
                "history": [
                    {"role": "user", "content": "Lá táo có đốm xanh ô liu là bệnh gì?"},
                    {"role": "assistant", "content": "Hệ thống nhận diện đây có thể là bệnh ghẻ táo."},
                ],
            },
            {"query": "Cháy muộn trên khoai tây phòng thế nào?", "plant_type": "potato", "disease": "late_blight"},
        ]},
    )

    history: list[ChatMessage] = Field(default_factory=list, max_length=50, description=(
        "Các lượt trước, cũ trước mới sau. Server không lưu lịch sử; chỉ giữ các lượt "
        "gần nhất theo CHAT_HISTORY_TURNS và CHAT_HISTORY_MAX_CHARS."
    ))
    subject_context: Annotated[str, StringConstraints(strip_whitespace=True, max_length=1000)] | None = Field(
        None, description="Kết quả nhận diện ảnh dạng chữ. Chỉ để LLM biết đối tượng, không phải bằng chứng.")
    retrieval_query: Query | None = Field(None, description=(
        "Câu dùng để TÌM tài liệu, khi khác câu người dùng hỏi (ví dụ câu đã làm rõ của "
        "detection-server). `query` vẫn là câu LLM nhìn thấy và trả lời. Bỏ trống = tìm bằng `query`."
    ))
    llm_provider: ProviderName | None = Field(None, description=(
        "LLM sinh câu trả lời cho request này. Bỏ trống = LLM_PROVIDER. Model của từng "
        "provider lấy trong .env (vLLM: model đang host)."
    ))
    rewrite_query: bool = Field(False, description=(
        "Viết lại câu hỏi nối tiếp thành câu độc lập bằng LLM trước khi tìm. Để false khi "
        "gọi từ detection-server vì normalizer ở đó đã làm việc này."
    ))

    @model_validator(mode="after")
    def _one_way_to_set_search_query(self) -> "AnswerRequest":
        if self.retrieval_query and self.rewrite_query:
            raise ValueError("Chọn một: gửi retrieval_query, hoặc bật rewrite_query.")
        return self


class Chunk(BaseModel):
    rank: int = Field(description="Thứ hạng sau khi gộp các nhánh, bắt đầu từ 1.")
    chunk_id: str = Field(description="ID ổn định của chunk; xem đầy đủ ở `GET /v1/admin/chunks/{chunk_id}`.")
    source: str = Field(description="Tài liệu gốc, cũng là nhãn trích dẫn trong câu trả lời.")
    heading_path: str | None = Field(None, description="Mục trong tài liệu (chỉ có với chunk structure).")
    content: str = Field(description="Nội dung chunk, gồm cả header định danh.")
    semantic_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


class LLMMeta(BaseModel):
    provider: ProviderName
    model: str | None = Field(description=(
        "Tên model do server LLM báo về. Với vLLM là served-model-name, có thể là alias "
        "(`rag-llm`); model thật xem `root` ở `GET /v1/llm?probe=true`."
    ))
    finish_reason: str | None = Field(description="`stop` là xong; `length`/`max_tokens` là bị cắt.")
    truncated: bool = Field(description="True khi câu trả lời bị cắt ở giới hạn token.")
    input_tokens: int | None = None
    output_tokens: int | None = None


class TimingMeta(BaseModel):
    rewrite: int | None = Field(None, description="ms viết lại câu hỏi; null khi không chạy.")
    retrieve: int | None = Field(None, description="ms tìm tài liệu; null khi scope chặn.")
    generate: int | None = Field(None, description="ms LLM sinh câu trả lời; null khi không gọi.")
    total: int | None = Field(None, description="ms cả request trong server.")


class ResponseMeta(BaseModel):
    """Cấu hình thật đã dùng: ghi lại khi so sánh model hoặc chunking."""

    index: IndexName
    chunking_strategy: str | None = Field(description="Theo manifest của index; null khi chưa mở index.")
    retrieval_mode: str | None = Field(description="null khi scope không cho tìm.")
    reranker_enabled: bool | None = None
    top_k: int
    search_queries: list[str] = Field(default_factory=list, description=(
        "Các câu thật sự dùng để tìm, câu chính trước, đã bỏ câu trùng. Rỗng khi scope không cho tìm."
    ))
    llm: LLMMeta | None = Field(description="null khi không gọi LLM (câu từ chối dựng sẵn).")
    timing_ms: TimingMeta


class RetrieveResponse(BaseModel):
    scope: Scope
    chunks: list[Chunk] = Field(description="Rỗng khi scope không cho tìm hoặc không có chunk nào khớp.")
    debug: dict | None = Field(None, description="Chỉ có khi `debug=true`.")
    meta: ResponseMeta


class Citations(BaseModel):
    cited: list[int] = Field(description="Các số [Nguồn n] câu trả lời đã dẫn, theo thứ tự xuất hiện.")
    invalid: list[int] = Field(description="Số nguồn nằm ngoài NGỮ CẢNH lượt này: dấu hiệu trích dẫn bịa.")


class AnswerResponse(BaseModel):
    answer: str = Field(description="Câu trả lời cho người dùng cuối.")
    sources: list[str] = Field(description="Tài liệu đã đưa vào ngữ cảnh, không trùng lặp.")
    grounded: bool = Field(description="True nếu có tài liệu và LLM đã được gọi; false là câu từ chối dựng sẵn.")
    retrieval_query: str = Field(description="Câu thật sự dùng để tìm (khác `query` khi bật rewrite_query).")
    scope: Scope
    citations: Citations
    chunks: list[Chunk] | None = Field(None, description="Chỉ có khi `debug=true`.")
    debug: dict | None = Field(None, description="Chỉ có khi `debug=true`.")
    meta: ResponseMeta


class IndexStatus(BaseModel):
    name: IndexName
    default: bool = Field(description="Index dùng khi request không gửi `index`.")
    directory: str
    exists: bool
    loaded: bool = Field(description="Đã mở trong process này.")
    has_manifest: bool
    strategy: str | None = None
    chunk_count: int | None = None
    built_at: str | None = None
    matches_data: bool | None = Field(None, description=(
        "Index có được build từ đúng data/ hiện tại không. False: phải index lại trước khi so sánh."
    ))
    detail: str | None = None


class StatusResponse(BaseModel):
    ready: bool = Field(description="Index đã mở được hay chưa.")
    detail: str | None = Field(None, description="Lý do chưa sẵn sàng.")
    index_directory: str
    chunk_count: int | None = None
    chunking_strategy: str | None = None
    embedding_model: str
    retrieval_mode: str
    reranker_enabled: bool
    llm_provider: str
    llm_model: str | None = None
    llm_endpoint: str | None = Field(None, description="Chỉ với vllm/ollama. Không bao giờ chứa key.")
    auth_enabled: bool
    default_index: IndexName
    indexes: list[IndexStatus] = Field(description="Mọi index chọn được qua trường `index`.")


class ServedModel(BaseModel):
    id: str = Field(description="Tên gọi qua API (vLLM: served-model-name).")
    root: str | None = Field(None, description="vLLM: model thật đang nạp (LLM_MODEL_ID).")
    max_model_len: int | None = None
    digest: str | None = Field(None, description="Ollama: 12 ký tự đầu digest của bản đang có.")
    details: str | None = Field(None, description="Ollama: số tham số và lượng tử hóa.")


class LLMInfo(BaseModel):
    provider: ProviderName
    default: bool = Field(description="Provider dùng khi request không gửi `llm_provider`.")
    model: str | None = Field(description="Model trong .env.")
    endpoint: str | None = Field(None, description="Không bao giờ chứa key.")
    probed: bool
    reachable: bool | None = Field(None, description="null khi không probe.")
    served: list[ServedModel] | None = None
    detail: str | None = None


# --- Kho tri thức (trang admin của web, chỉ đọc) ---------------------------------

IndexState = Literal["current", "changed", "not_indexed", "deleted", "unknown"]


class KbTaxonomyLink(BaseModel):
    plant: str
    disease: str


class KbTokenStats(BaseModel):
    mean: int
    median: int
    max: int


class KbIndexOverview(IndexStatus):
    documents: int | None = Field(None, description="Số tài liệu có chunk trong index.")
    chunks: int | None = None
    tokens: KbTokenStats | None = Field(None, description="Token mỗi chunk (tokenizer của embedding).")
    chunks_per_crop: dict[str, int] = Field(default_factory=dict)


class KbDataInfo(BaseModel):
    documents: int = Field(description="Số file .txt trong data/.")
    fingerprint: str = Field(description="12 ký tự đầu dấu vân tay của data/.")


class KbOverview(BaseModel):
    data: KbDataInfo
    default_index: IndexName
    indexes: list[KbIndexOverview]


class KbDocument(BaseModel):
    source: str = Field(description="Đường dẫn trong data/, ví dụ `apple/apple_scab.txt`.")
    crop: str
    title: str
    chars: int | None = Field(None, description="null khi tài liệu đã xóa khỏi data/.")
    chunks: int
    tokens: int
    index_state: IndexState = Field(description=(
        "`current`: index khớp file; `changed`: file đã sửa sau lần index; `not_indexed`: file mới; "
        "`deleted`: còn trong index nhưng đã xóa khỏi data/; `unknown`: index cũ không có manifest."
    ))
    taxonomy: list[KbTaxonomyLink] = Field(description="Bệnh trong taxonomy trỏ tới tài liệu này; rỗng = chưa gắn.")


class KbChunkSummary(BaseModel):
    chunk_id: str
    source: str
    position: int = Field(description="Thứ tự trong tài liệu, bắt đầu từ 0.")
    heading_path: str | None = None
    section: str | None = None
    tokens: int
    chars: int
    start_line: int | None = None
    end_line: int | None = None
    preview: str = Field(description="Hai dòng đầu của thân chunk.")


class KbDocumentDetail(KbDocument):
    index: IndexName
    chunk_list: list[KbChunkSummary]
    text: str | None = Field(None, description="Văn bản gốc, khi `include_text=true`.")


class KbChunk(KbChunkSummary):
    index: IndexName
    header: str = Field(description="Header định danh chèn vào đầu chunk khi index.")
    body: str
    metadata: dict
    total_in_document: int
    prev_id: str | None = None
    next_id: str | None = None


class TaxonomyDisease(BaseModel):
    disease: str
    source: str | None
    has_document: bool
    aliases: list[str]


class TaxonomyPlant(BaseModel):
    plant: str
    crop: str
    plant_aliases: list[str]
    diseases: list[TaxonomyDisease]


class TaxonomyResponse(BaseModel):
    plants: list[TaxonomyPlant]
    statuses: dict[str, str] = Field(description="Ý nghĩa từng trạng thái scope.")
