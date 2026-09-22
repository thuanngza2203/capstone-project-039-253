from datetime import datetime, timezone

from bson import ObjectId
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.feedback.database import feedback_collection

router = APIRouter(prefix="/feedback", tags=["Feedback"])


class FeedbackUpdate(BaseModel):
    feedback_id: str
    rating: str
    reasons: list[str] = Field(default_factory=list)


@router.put("")
async def update_feedback(data: FeedbackUpdate):
    if data.rating not in {"like", "unlike"}:
        raise HTTPException(status_code=400, detail="rating phải là like hoặc unlike")

    if not ObjectId.is_valid(data.feedback_id):
        raise HTTPException(status_code=400, detail="feedback_id không hợp lệ")

    if data.rating == "unlike" and not data.reasons:
        raise HTTPException(status_code=400, detail="Unlike cần ít nhất một lý do")

    reasons = [] if data.rating == "like" else list(dict.fromkeys(data.reasons))

    result = await feedback_collection.update_one(
        {"_id": ObjectId(data.feedback_id)},
        {
            "$set": {
                "user_feedback.rating": data.rating,
                "user_feedback.reasons": reasons,
                "updated_at": datetime.now(timezone.utc),
            }
        },
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Không tìm thấy feedback")

    return {
        "success": True,
        "feedback_id": data.feedback_id,
        "rating": data.rating,
        "reasons": reasons,
    }
