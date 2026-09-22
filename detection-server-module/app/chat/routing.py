from app.schemas import Action, Intent, QueryAnalysis, ResolvedQuery, RouteDecision


class QueryRouter:
    def decide(
        self,
        *,
        analysis: QueryAnalysis,
        resolved: ResolvedQuery,
        has_current_image: bool,
    ) -> RouteDecision:
        if not analysis.is_plant_related:
            return RouteDecision(
                action=Action.OUT_OF_SCOPE,
                message="Mình đang hỗ trợ các câu hỏi về cây trồng và bệnh cây.",
            )

        # Muốn xác định bệnh nhưng chưa resolve được disease.
        if resolved.intent == Intent.DIAGNOSIS:
            if resolved.disease:
                return RouteDecision(action=Action.ACCEPT_QUERY)

            return RouteDecision(
                action=Action.REQUEST_IMAGE,
                requires_image=True,
                message="Bạn hãy tải ảnh lá cây lên để mình xác định bệnh trước nhé.",
            )

        if resolved.intent in {
            Intent.TREATMENT,
            Intent.CAUSE,
            Intent.PREVENTION,
        }:
            # Có tên bệnh từ query hoặc session/detector -> đi tiếp.
            if resolved.disease:
                return RouteDecision(action=Action.ACCEPT_QUERY)

            # Có symptom nhưng chưa biết disease -> cần ảnh.
            if resolved.symptoms:
                return RouteDecision(
                    action=Action.REQUEST_IMAGE,
                    requires_image=True,
                    message=(
                        "Bạn mới mô tả triệu chứng nên chưa xác định được bệnh cụ thể. "
                        "Hãy tải ảnh lá cây lên để mình nhận diện trước."
                    ),
                )

            # Câu kiểu "bệnh này chữa sao?" nhưng session chưa có detection.
            if analysis.refers_to_previous_context:
                return RouteDecision(
                    action=Action.REQUEST_IMAGE,
                    requires_image=True,
                    message=(
                        "Mình chưa có kết quả nhận diện bệnh trong cuộc chat này. "
                        "Bạn hãy tải ảnh lá cây lên trước nhé."
                    ),
                )

            return RouteDecision(
                action=Action.ASK_CLARIFICATION,
                message="Bạn đang hỏi về bệnh nào? Bạn có thể nêu tên bệnh hoặc tải ảnh lá cây.",
            )

        if resolved.intent == Intent.GENERAL_INFO:
            if resolved.disease or resolved.plant:
                return RouteDecision(action=Action.ACCEPT_QUERY)

            if analysis.refers_to_previous_context:
                return RouteDecision(
                    action=Action.REQUEST_IMAGE,
                    requires_image=True,
                    message=(
                        "Mình chưa biết bạn đang nói tới bệnh nào trong cuộc chat này. "
                        "Bạn hãy tải ảnh hoặc nêu tên bệnh."
                    ),
                )

            return RouteDecision(
                action=Action.ASK_CLARIFICATION,
                message="Bạn muốn hỏi thông tin về cây hoặc bệnh nào?",
            )

        return RouteDecision(
            action=Action.ASK_CLARIFICATION,
            message="Bạn có thể nói rõ hơn bạn muốn hỏi gì về cây hoặc bệnh cây không?",
        )
