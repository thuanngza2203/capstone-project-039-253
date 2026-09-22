import json
import re

from groq import AsyncGroq
from pydantic import ValidationError

from app.schemas import QueryAnalysis, RagDocument, ResolvedQuery
from app.prompts.answer_prompt import ANSWER_SYSTEM_PROMPT
from app.prompts.query_normalizer_prompt import QUERY_NORMALIZER_SYSTEM_PROMPT
from app.llm.base import AnswerLLM, QueryNormalizerLLM


class GroqQueryNormalizer(QueryNormalizerLLM):
    def __init__(self, api_key: str, model: str):
        self.client = AsyncGroq(api_key=api_key)
        self.model = model

    async def analyze(
        self,
        raw_query: str,
        session_context: str,
    ) -> QueryAnalysis:
        system_prompt = (
            QUERY_NORMALIZER_SYSTEM_PROMPT
            + """

==================================================
OUTPUT FORMAT BẮT BUỘC
==================================================

Chỉ trả về MỘT JSON object hợp lệ.

KHÔNG:
- giải thích
- markdown
- ```json
- reasoning
- <think>
- text trước JSON
- text sau JSON

JSON phải luôn có đầy đủ các field:

{
  "normalized_query": "string",
  "plant": null,
  "disease": null,
  "symptoms": [],
  "intent": "treatment",
  "focus": null,
  "refers_to_previous_context": false,
  "is_plant_related": true
}

intent chỉ được phép là:
- "diagnosis"
- "treatment"
- "cause"
- "prevention"
- "general_info"
- "other"

Nếu không biết plant:
"plant": null

Nếu không biết disease:
"disease": null

Nếu không có symptoms:
"symptoms": []

Nếu không có focus:
"focus": null

Không được bỏ bất kỳ field nào.

Chỉ xuất JSON cuối cùng.
"""
        )

        user_prompt = f"""
SESSION CONTEXT:
{session_context or "(không có)"}

CURRENT USER QUERY:
{raw_query}
"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0,
            max_tokens=800,

            # Normalizer của bạn hiện đang chạy được với none
            reasoning_effort="none",
        )

        message = response.choices[0].message
        text = message.content

        if not text:
            raise RuntimeError(
                f"Groq normalizer returned empty response. "
                f"Full message: {message}"
            )

        text = text.strip()

        # Xóa markdown fence nếu có.
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

        # Xóa reasoning <think>...</think> nếu model vẫn sinh.
        text = re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()

        # Tìm JSON object.
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise RuntimeError(
                f"Groq normalizer did not return JSON: {text}"
            )

        json_text = text[start:end + 1]

        try:
            data = json.loads(json_text)

        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Groq normalizer returned invalid JSON: {json_text}"
            ) from exc

        try:
            return QueryAnalysis.model_validate(data)

        except ValidationError as exc:
            raise RuntimeError(
                "Groq JSON does not match QueryAnalysis. "
                f"Generated JSON: {json_text}. "
                f"Validation error: {exc}"
            ) from exc


class GroqAnswerLLM(AnswerLLM):
    def __init__(
        self,
        api_key: str,
        model: str,
    ):
        self.client = AsyncGroq(api_key=api_key)
        self.model = model

    async def generate(
        self,
        *,
        original_query: str,
        retrieval_query: str,
        resolved: ResolvedQuery,
        rag_documents: list[RagDocument],
        recent_history: str,
        detector_context: str,
        feedback_examples: list[dict] | None = None,
    ) -> str:
        context = "\n\n".join(
            f"""
[{doc.id}] {doc.title}
{doc.content}
Nguồn: {doc.source}
""".strip()
            for doc in rag_documents
        )

        feedback_examples = feedback_examples or []
        feedback_context = self._build_feedback_context(feedback_examples)
        system_prompt = ANSWER_SYSTEM_PROMPT

        user_prompt = f"""
RECENT HISTORY:
{recent_history or "(không có)"}

DETECTOR CONTEXT:
{detector_context or "(không có ảnh ở lượt này)"}

RESOLVED DATA:
- plant: {resolved.plant}
- disease: {resolved.disease}
- intent: {resolved.intent.value}
- focus: {resolved.focus}

ORIGINAL USER QUERY:
{original_query}

NORMALIZED RETRIEVAL QUERY:
{retrieval_query}

DOMAIN RAG CONTEXT:
{context or "(không có context)"}

ADMIN FEEDBACK RAG CONTEXT:
{feedback_context or "(không có QA admin phù hợp)"}

Hãy trả lời câu hỏi người dùng dựa trên dữ liệu và context được cung cấp.

Nếu detector đã xác định được cây và bệnh thì dùng thông tin đó làm context chính.
Không được tự thay đổi tên cây hoặc tên bệnh đã resolve.
"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0.2,
            max_tokens=700,
            reasoning_effort="low",
        )

        message = response.choices[0].message
        text = message.content

        if not text:
            raise RuntimeError(
                f"Groq answer model returned empty response. "
                f"Full message: {message}"
            )

        text = text.strip()

        text = re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()

        if not text:
            raise RuntimeError(
                f"Groq answer model returned only reasoning. "
                f"Full message: {message}"
            )

        return text

    @staticmethod
    def _build_feedback_context(feedback_examples: list[dict]) -> str:
        """Only top reranked admin examples are inserted into the prompt."""
        if not feedback_examples:
            return ""

        blocks: list[str] = []
        for index, example in enumerate(feedback_examples, start=1):
            question = (example.get("question") or "").strip()
            preferred_answer = (example.get("preferred_answer") or "").strip()
            if not question or not preferred_answer:
                continue

            # Hard cap protects the final prompt even if admin pasted a long answer.
            question = question[:700]
            preferred_answer = preferred_answer[:1800]
            blocks.append(
                f"Example {index} (admin-verified):\n"
                f"Question: {question}\n"
                f"Preferred answer: {preferred_answer}"
            )

        return "\n\n".join(blocks)
