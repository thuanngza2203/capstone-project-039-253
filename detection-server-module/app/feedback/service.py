from datetime import datetime, timezone

from bson import ObjectId

from app.feedback.database import feedback_collection


async def save_feedback_record(
    session_id: str,
    question: str,
    answer: str,
    metadata: dict | None = None,
) -> str:
    """Create exactly one feedback record for one assistant answer.

    The record exists immediately so the frontend can receive an id, but it is
    not considered user feedback until user_feedback.rating becomes like/unlike.
    """
    now = datetime.now(timezone.utc)
    metadata = metadata or {}

    result = await feedback_collection.insert_one(
        {
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "action": metadata.get("action"),
            "image_filename": metadata.get("image_filename"),
            "metadata": metadata,
            "user_feedback": {
                "rating": None,
                "reasons": [],
            },
            "admin_feedback": {
                "correct_answer": "",
                "approved": False,
            },
            "training": {
                "selected": False,
            },
            "created_at": now,
            "updated_at": now,
        }
    )
    return str(result.inserted_id)


async def update_feedback(
    feedback_id: str,
    rating: str,
    reasons: list[str] | None = None,
) -> bool:
    reasons = reasons or []
    result = await feedback_collection.update_one(
        {"_id": ObjectId(feedback_id)},
        {
            "$set": {
                "user_feedback.rating": rating,
                "user_feedback.reasons": reasons,
                "updated_at": datetime.now(timezone.utc),
            }
        },
    )
    return result.matched_count > 0


async def get_negative_feedback() -> list[dict]:
    docs: list[dict] = []
    cursor = feedback_collection.find({"user_feedback.rating": "unlike"})
    async for item in cursor:
        item["_id"] = str(item["_id"])
        docs.append(item)
    return docs
