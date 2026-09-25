from app.chat.labels import HEALTHY, PLANT_VI, canonical_disease, canonical_plant
from app.schemas import Action, DetectionResult, Intent, QueryAnalysis, ResolvedQuery, RouteDecision


# Dưới ngưỡng này câu trả lời "cây khỏe" nói rõ là chưa chắc và nhắc chụp lại.
HEALTHY_CONFIDENT = 0.8
# Hỏi cây có bệnh gì / chữa thế nào mà ảnh cho thấy lá khỏe: trả lời luôn. Hỏi phòng bệnh,
# chăm sóc... vẫn tra tài liệu (RAG có tài liệu phòng bệnh của cây).
HEALTHY_ANSWER_INTENTS = frozenset({Intent.DIAGNOSIS, Intent.TREATMENT})


def healthy_message(detection: DetectionResult) -> str:
    plant = PLANT_VI.get(canonical_plant(detection.plant) or "", detection.plant)
    percent = f"{detection.confidence * 100:.0f}%"
    if detection.confidence >= HEALTHY_CONFIDENT:
        return (
            f"Hệ thống nhận diện đây là **lá {plant} khỏe mạnh** (độ tin cậy {percent}), "
            "không thấy dấu hiệu bệnh.\n\n"
            "Bạn chỉ cần tiếp tục chăm sóc và theo dõi cây. Nếu sau này lá có đốm, vàng, xoăn hay "
            "héo, hãy chụp lại phần lá bất thường để hệ thống kiểm tra."
        )
    return (
        f"Hệ thống nghiêng về **lá {plant} khỏe mạnh** nhưng chưa chắc chắn (độ tin cậy {percent}).\n\n"
        "Nếu bạn thấy lá có đốm, vàng, xoăn hay héo, hãy chụp lại gần hơn, đủ sáng, vào đúng phần "
        "lá bất thường để kiểm tra lại."
    )


class QueryRouter:
    def healthy(
        self,
        *,
        analysis: QueryAnalysis,
        resolved: ResolvedQuery,
        detection: DetectionResult | None,
        image_only: bool,
    ) -> RouteDecision | None:
        """Ảnh (lượt này, hoặc lượt trước khi câu hỏi nhắc lại) là lá khỏe và người dùng hỏi
        cây có bệnh không: trả lời luôn, không gọi RAG.

        Kho chỉ có tài liệu về bệnh; tìm "lá này có bị gì không" trong đó chỉ ra câu "chưa đủ
        thông tin". Người dùng tự nêu một bệnh (resolved.disease khác nhãn ảnh) thì vẫn tra
        tài liệu bệnh đó. `image_only`: chỉ gửi ảnh, câu hỏi mặc định là hỏi bệnh.
        """
        if detection is None or canonical_disease(detection.disease) != HEALTHY:
            return None
        if resolved.disease != detection.disease:
            return None
        if not image_only and analysis.intent not in HEALTHY_ANSWER_INTENTS:
            return None
        return RouteDecision(action=Action.HEALTHY_PLANT, message=healthy_message(detection))

    def decide(
        self,
        *,
        analysis: QueryAnalysis,
        resolved: ResolvedQuery,
        has_current_image: bool,
    ) -> RouteDecision:
        # Ảnh đã qua bước kiểm tra "có lá cây" nên lượt này chắc chắn về cây trồng; Groq không
        # thấy ảnh, câu mặc định "Ảnh này đang bị bệnh gì?" hay bị đánh giá là ngoài phạm vi.
        if not analysis.is_plant_related and not has_current_image:
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

            # Bệnh chỉ đoán từ cách gọi chung chung ("bệnh đốm trên cây táo"): vẫn trả
            # lời, RAG tìm trong tài liệu của cây nên mọi bệnh hợp mô tả đều có cơ hội.
            if resolved.suspected_disease and resolved.plant:
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
