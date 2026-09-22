from app.schemas import Action, Intent, ResolvedQuery, RouteDecision


class RetrievalQueryBuilder:
    """
    Đây là normalize lần 2:
    sau khi đã resolve plant/disease từ query + ảnh + session.

    Output là câu đầy đủ để đưa vào RAG.
    """

    def build(
        self,
        *,
        resolved: ResolvedQuery,
        decision: RouteDecision,
        fallback_normalized_query: str,
    ) -> str | None:
        if decision.action != Action.ACCEPT_QUERY:
            return None

        plant = self._clean(resolved.plant)
        disease = self._clean(resolved.disease)
        focus = self._clean(resolved.focus)

        if resolved.intent == Intent.TREATMENT:
            base = self._with_entity(
                "Cách điều trị",
                disease=disease,
                plant=plant,
            )
            if focus:
                return f"{base}; tập trung vào {focus}"
            return base

        if resolved.intent == Intent.CAUSE:
            base = self._with_entity(
                "Nguyên nhân",
                disease=disease,
                plant=plant,
            )
            if focus:
                return f"{base}; tập trung vào {focus}"
            return base

        if resolved.intent == Intent.PREVENTION:
            base = self._with_entity(
                "Cách phòng ngừa",
                disease=disease,
                plant=plant,
            )
            if focus:
                return f"{base}; tập trung vào {focus}"
            return base

        if resolved.intent == Intent.DIAGNOSIS:
            return self._with_entity(
                "Thông tin nhận diện",
                disease=disease,
                plant=plant,
            )

        if resolved.intent == Intent.GENERAL_INFO:
            base = self._with_entity(
                "Thông tin",
                disease=disease,
                plant=plant,
            )
            if focus:
                return f"{base}; tập trung vào {focus}"
            return base

        return fallback_normalized_query.strip() or None

    @staticmethod
    def _with_entity(
        prefix: str,
        *,
        disease: str | None,
        plant: str | None,
    ) -> str:
        if disease and plant:
            return f"{prefix} bệnh {disease} trên cây {plant}"
        if disease:
            return f"{prefix} bệnh {disease}"
        if plant:
            return f"{prefix} về cây {plant}"
        return prefix

    @staticmethod
    def _clean(value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None
