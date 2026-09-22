from app.schemas import DetectionResult, QueryAnalysis, ResolvedQuery


class ContextResolver:
    """
    Ghép:
    - entity user nói ở query hiện tại
    - detection của ảnh hiện tại
    - memory detection của session

    Query hiện tại được ưu tiên.
    Session chỉ được dùng khi query phụ thuộc context trước.
    """

    def resolve(
        self,
        *,
        analysis: QueryAnalysis,
        current_detection: DetectionResult | None,
        session_detection: DetectionResult | None,
    ) -> ResolvedQuery:
        plant = analysis.plant
        disease = analysis.disease
        source_parts = []

        if plant or disease:
            source_parts.append("query")

        # Nếu user vừa upload ảnh, detector là context mạnh cho lượt hiện tại.
        if current_detection is not None:
            if plant is None:
                plant = current_detection.plant
            if disease is None:
                disease = current_detection.disease
            source_parts.append("current_image")

        # Chỉ lấy memory khi câu thực sự ám chỉ lượt trước.
        elif analysis.refers_to_previous_context and session_detection is not None:
            if plant is None:
                plant = session_detection.plant
            if disease is None:
                disease = session_detection.disease
            source_parts.append("session_memory")

        return ResolvedQuery(
            plant=plant,
            disease=disease,
            symptoms=analysis.symptoms,
            intent=analysis.intent,
            focus=analysis.focus,
            context_source="+".join(source_parts) if source_parts else "query_only",
        )
