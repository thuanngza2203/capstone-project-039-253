from datetime import datetime, timezone

from bson import ObjectId
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.feedback.rag import get_feedback_rag_service
from app.feedback.database import feedback_collection

router = APIRouter(prefix="/admin", tags=["Admin"])


class TrainingItem(BaseModel):
    id: str
    correct_answer: str = ""


class TrainingRequest(BaseModel):
    items: list[TrainingItem] = Field(default_factory=list)


@router.get("/reviews")
async def reviews():
    """Only show messages that the user actually liked/disliked."""
    items = []
    cursor = feedback_collection.find(
        {"user_feedback.rating": {"$in": ["like", "unlike"]}}
    ).sort("created_at", -1)

    async for item in cursor:
        item["_id"] = str(item["_id"])
        items.append(item)

    return items


@router.delete("/review/{feedback_id}")
async def delete_review(feedback_id: str):
    """Delete raw QA feedback and its Feedback-RAG vector document."""
    if not ObjectId.is_valid(feedback_id):
        raise HTTPException(status_code=400, detail="Feedback id không hợp lệ")

    result = await feedback_collection.delete_one({"_id": ObjectId(feedback_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Không tìm thấy QA")

    await get_feedback_rag_service().delete_example(feedback_id)

    return {"success": True, "deleted_id": feedback_id}


@router.post("/training")
async def apply_training(data: TrainingRequest):
    """Index selected QA into Feedback RAG instead of appending all QA to prompt.

    Like:
        If admin leaves Correct Answer empty, the liked bot answer is the
        preferred answer.

    Unlike:
        Admin must provide Correct Answer before the QA is indexed.

    The index stores a multilingual embedding in MongoDB. During chat, only a
    small retrieved + reranked subset is sent to Answer LLM.
    """
    if not data.items:
        raise HTTPException(status_code=400, detail="Chưa chọn QA để training")

    rag = get_feedback_rag_service()
    prepared: list[dict] = []

    for item in data.items:
        if not ObjectId.is_valid(item.id):
            raise HTTPException(
                status_code=400,
                detail=f"Feedback id không hợp lệ: {item.id}",
            )

        object_id = ObjectId(item.id)
        doc = await feedback_collection.find_one({"_id": object_id})
        if doc is None:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy QA {item.id}")

        rating = doc.get("user_feedback", {}).get("rating")
        reasons = doc.get("user_feedback", {}).get("reasons", []) or []
        correct_answer = item.correct_answer.strip()
        previous_answer = (doc.get("answer") or "").strip()
        question = (doc.get("question") or "").strip()

        if rating == "unlike" and not correct_answer:
            raise HTTPException(
                status_code=400,
                detail=(
                    "QA bị Unlike phải có Correct Answer trước khi training: "
                    f"{question[:80]}"
                ),
            )

        preferred_answer = correct_answer or previous_answer
        if not question or not preferred_answer:
            raise HTTPException(
                status_code=400,
                detail=f"QA {item.id} thiếu Question/Answer để index",
            )

        metadata = doc.get("metadata", {}) or {}
        prepared.append(
            {
                "object_id": object_id,
                "feedback_id": item.id,
                "question": question,
                "preferred_answer": preferred_answer,
                "correct_answer": correct_answer,
                "planttype": metadata.get("planttype"),
                "disease": metadata.get("disease"),
                "rating": rating,
                "reasons": reasons,
            }
        )

    now = datetime.now(timezone.utc)

    # Embedding is computed locally. First call may take longer because the
    # sentence-transformers model is downloaded/lazy-loaded.
    for item in prepared:
        await rag.index_example(
            feedback_id=item["feedback_id"],
            question=item["question"],
            preferred_answer=item["preferred_answer"],
            planttype=item["planttype"],
            disease=item["disease"],
            rating=item["rating"],
            reasons=item["reasons"],
        )

        await feedback_collection.update_one(
            {"_id": item["object_id"]},
            {
                "$set": {
                    "admin_feedback.correct_answer": item["correct_answer"],
                    "training.selected": True,
                    "training.mode": "feedback_rag",
                    "training.selected_at": now,
                    "training.indexed_at": now,
                    "updated_at": now,
                }
            },
        )

    return {
        "success": True,
        "trained_count": len(prepared),
        "indexed_count": len(prepared),
        "mode": "feedback_rag",
    }


@router.get("/training")
async def training_dataset():
    """Inspect active Feedback-RAG index entries (embeddings omitted)."""
    return await get_feedback_rag_service().list_indexed()
