from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.config import get_settings
from app.feedback.database import feedback_training_collection


class FeedbackRAGService:
    """RAG riêng cho các QA đã được admin chọn Training.

    Kiến trúc Feedback RAG:
    1. MongoDB lưu QA + embedding vector.
    2. intfloat/multilingual-e5-small tạo embedding 384 chiều.
    3. Metadata filter theo planttype/disease trước.
    4. Cosine similarity lấy top-N candidate.
    5. CrossEncoder multilingual rerank và chỉ trả top-K nhỏ vào Answer LLM.

    Cách này không nhét toàn bộ QA training vào prompt nên số token không tăng
    tuyến tính theo số lượng feedback đã tích lũy.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.embedding_model_name = settings.feedback_embedding_model
        self.reranker_model_name = settings.feedback_reranker_model
        self.retrieve_k = settings.feedback_retrieve_k
        self.rerank_k = settings.feedback_rerank_k
        self.similarity_threshold = settings.feedback_similarity_threshold
        self.enable_reranker = settings.feedback_reranker_enabled

        # Lazy-load để server khởi động nhanh. Model chỉ tải khi admin bấm
        # Training hoặc khi query Feedback RAG lần đầu.
        self._embedder: SentenceTransformer | None = None
        self._reranker: CrossEncoder | None = None

    def _get_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def _get_reranker(self) -> CrossEncoder | None:
        if not self.enable_reranker:
            return None
        if self._reranker is None:
            self._reranker = CrossEncoder(self.reranker_model_name)
        return self._reranker

    async def _encode_passage(self, text: str) -> list[float]:
        def _run() -> list[float]:
            model = self._get_embedder()
            vector = model.encode(
                f"passage: {text}",
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return vector.astype(float).tolist()

        return await asyncio.to_thread(_run)

    async def _encode_query(self, text: str) -> np.ndarray:
        def _run() -> np.ndarray:
            model = self._get_embedder()
            vector = model.encode(
                f"query: {text}",
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return np.asarray(vector, dtype=np.float32)

        return await asyncio.to_thread(_run)

    async def index_example(
        self,
        *,
        feedback_id: str,
        question: str,
        preferred_answer: str,
        planttype: str | None,
        disease: str | None,
        rating: str | None,
        reasons: list[str] | None = None,
    ) -> None:
        """Upsert một QA đã được admin duyệt vào Feedback RAG index."""
        question = question.strip()
        preferred_answer = preferred_answer.strip()
        if not question or not preferred_answer:
            raise ValueError("Question và preferred_answer không được rỗng")

        searchable_text = (
            f"Question: {question}\n"
            f"Preferred answer: {preferred_answer}"
        )
        embedding = await self._encode_passage(searchable_text)

        now = datetime.now(timezone.utc)
        await feedback_training_collection.update_one(
            {"feedback_id": feedback_id},
            {
                "$set": {
                    "feedback_id": feedback_id,
                    "question": question,
                    "preferred_answer": preferred_answer,
                    "planttype": self._norm(planttype),
                    "disease": self._norm(disease),
                    "rating": rating,
                    "reasons": reasons or [],
                    "search_text": searchable_text,
                    "embedding": embedding,
                    "embedding_model": self.embedding_model_name,
                    "active": True,
                    "updated_at": now,
                },
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )

    async def delete_example(self, feedback_id: str) -> None:
        await feedback_training_collection.delete_one({"feedback_id": feedback_id})

    async def list_indexed(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        cursor = feedback_training_collection.find(
            {"active": True},
            {"embedding": 0},
        )
        async for item in cursor:
            item["_id"] = str(item["_id"])
            items.append(item)
        return items

    async def retrieve(
        self,
        *,
        query: str,
        planttype: str | None,
        disease: str | None,
    ) -> list[dict[str, Any]]:
        """Retrieve + rerank feedback examples.

        Interface cố ý chỉ nhận đúng 3 field theo contract của project:
        - query: normalized query
        - planttype: resolved plant
        - disease: resolved disease
        """
        query = (query or "").strip()
        if not query:
            return []

        # Check MongoDB first. If admin has never indexed a QA, do not even load
        # the local embedding model; this keeps ordinary chat startup cheap.
        candidates = await self._load_candidates(
            planttype=self._norm(planttype),
            disease=self._norm(disease),
        )

        if not candidates:
            return []

        query_vector = await self._encode_query(query)
        scored: list[dict[str, Any]] = []

        for item in candidates:
            vector = item.get("embedding")
            if not vector:
                continue

            doc_vector = np.asarray(vector, dtype=np.float32)
            if doc_vector.shape != query_vector.shape:
                continue

            # Vectors đã normalize nên dot product = cosine similarity.
            similarity = float(np.dot(query_vector, doc_vector))
            if similarity < self.similarity_threshold:
                continue

            scored.append(
                {
                    "feedback_id": item.get("feedback_id"),
                    "question": item.get("question", ""),
                    "preferred_answer": item.get("preferred_answer", ""),
                    "planttype": item.get("planttype"),
                    "disease": item.get("disease"),
                    "rating": item.get("rating"),
                    "reasons": item.get("reasons", []),
                    "similarity": round(similarity, 4),
                }
            )

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        shortlist = scored[: self.retrieve_k]
        if not shortlist:
            return []

        try:
            reranker = await asyncio.to_thread(self._get_reranker)
        except Exception:
            reranker = None

        if reranker is None:
            return shortlist[: self.rerank_k]

        try:
            pairs = [
                [query, f"{item['question']}\n{item['preferred_answer']}"]
                for item in shortlist
            ]

            def _predict() -> list[float]:
                scores = np.asarray(reranker.predict(pairs)).reshape(-1)
                return [float(x) for x in scores]

            rerank_scores = await asyncio.to_thread(_predict)

            for item, rerank_score in zip(shortlist, rerank_scores):
                item["rerank_score"] = round(rerank_score, 4)

            shortlist.sort(
                key=lambda x: x.get("rerank_score", float("-inf")),
                reverse=True,
            )
        except Exception:
            # Nếu reranker không load được, retrieval cosine vẫn hoạt động.
            for item in shortlist:
                item["rerank_score"] = None

        return shortlist[: self.rerank_k]

    async def _load_candidates(
        self,
        *,
        planttype: str | None,
        disease: str | None,
    ) -> list[dict[str, Any]]:
        """Metadata filter trước để giảm số vector cần so cosine.

        Local MongoDB chưa dùng Atlas Vector Search. Nếu metadata exact không
        có đủ candidate, fallback sang active examples toàn cục để QA cũ vẫn có
        thể được tìm thấy.
        """
        exact_filter: dict[str, Any] = {"active": True}
        if planttype:
            exact_filter["planttype"] = planttype
        if disease:
            exact_filter["disease"] = disease

        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()

        async for item in feedback_training_collection.find(exact_filter):
            key = str(item.get("feedback_id") or item.get("_id"))
            seen.add(key)
            candidates.append(item)

        # Dataset hiện tại thường nhỏ. Fallback global giúp examples cũ (chưa có
        # metadata plant/disease) vẫn dùng được. Giới hạn để tránh scan vô hạn.
        if len(candidates) < self.retrieve_k:
            cursor = feedback_training_collection.find({"active": True}).limit(300)
            async for item in cursor:
                key = str(item.get("feedback_id") or item.get("_id"))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(item)

        return candidates

    @staticmethod
    def _norm(value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip().lower().replace("-", "_").replace(" ", "_")
        return value or None


@lru_cache
def get_feedback_rag_service() -> FeedbackRAGService:
    return FeedbackRAGService()
