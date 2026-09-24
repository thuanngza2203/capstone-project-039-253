from app.schemas import Action, QueryAnalysis, RouteDecision


class RetrievalQueryBuilder:
    """
    Câu tìm tài liệu gửi sang RAG là câu Groq đã chuẩn hóa.

    Groq sửa dấu, chính tả, viết tắt và giữ đủ chi tiết user hỏi ("có cần nhổ
    cây không"). Câu mẫu theo intent trước đây ("Cách điều trị bệnh X trên cây Y")
    làm rơi các chi tiết đó.

    Không gắn tên cây/bệnh lấy từ ảnh hay lượt trước vào câu ("Bệnh này chữa sao?"
    giữ nguyên): `plant_type`/`disease` gửi kèm đã khoanh RAG vào đúng tài liệu, và
    đo 24/09 cho thấy gắn thêm "(bệnh …, cây …)" làm Hit@1 trong tài liệu giảm
    0,65 → 0,49 vì kéo chunk tổng quan lên đầu (RAG-module/reports/2026-09-24-query-normalization).
    """

    def __init__(self, *, search_original_query: bool = False):
        # Gửi kèm câu gốc làm câu tìm phụ (RAG gộp hai câu bằng RRF). Tắt mặc định:
        # đo 24/09 không thấy lợi, câu không dấu làm nhánh semantic kéo thứ hạng xuống.
        self.search_original_query = search_original_query

    def build(self, *, analysis: QueryAnalysis, decision: RouteDecision) -> str | None:
        if decision.action != Action.ACCEPT_QUERY:
            return None
        return " ".join(analysis.normalized_query.split()) or None

    def extra_queries(self, *, raw_query: str, analysis: QueryAnalysis) -> list[str]:
        """Câu gốc, khi bật và khi khác câu chuẩn hóa (bỏ qua hoa/thường, khoảng trắng)."""
        raw = " ".join(raw_query.split())
        if not self.search_original_query or not raw:
            return []
        if raw.casefold() == " ".join(analysis.normalized_query.split()).casefold():
            return []
        return [raw]
