import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from bson import ObjectId

from app.schemas import ChatTurn, DetectionResult, SessionSnapshot
from app.feedback.database import conversation_collection, feedback_collection


def _chat_messages(messages) -> list[dict]:
    return [
        {"role": message.get("role"), "content": message.get("content")}
        for message in messages
        if message.get("role") in {"user", "assistant"}
        and (message.get("content") or "").strip()
    ]


class MongoSessionStore:
    """Persistent conversation/session store backed by MongoDB.

    Each session is one MongoDB document in ``chat_sessions`` and contains the
    whole conversation in ``messages``. This replaces the previous RAM-only
    store, so histories survive server restarts and can be displayed like
    ChatGPT's conversation sidebar.
    """

    def __init__(self, max_turns: int = 6):
        # Không cắt lịch sử lưu trong Mongo; chỉ giới hạn số message đưa vào LLM.
        self.max_turns = max_turns

    async def _ensure_session(self, session_id: str) -> None:
        now = datetime.now(timezone.utc)
        await conversation_collection.update_one(
            {"session_id": session_id},
            {
                "$setOnInsert": {
                    "session_id": session_id,
                    "title": "Cuộc trò chuyện mới",
                    "messages": [],
                    "last_detection": None,
                    "created_at": now,
                },
                "$set": {"updated_at": now},
            },
            upsert=True,
        )

    async def get_last_detection(self, session_id: str) -> DetectionResult | None:
        doc = await conversation_collection.find_one(
            {"session_id": session_id},
            {"last_detection": 1},
        )
        if not doc or not doc.get("last_detection"):
            return None
        return DetectionResult.model_validate(doc["last_detection"])

    async def set_last_detection(
        self,
        session_id: str,
        detection: DetectionResult,
    ) -> None:
        await self._ensure_session(session_id)
        await conversation_collection.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "last_detection": detection.model_dump(mode="json"),
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )

    async def add_turn(
        self,
        session_id: str,
        turn: ChatTurn,
    ) -> None:
        await self._ensure_session(session_id)
        now = datetime.now(timezone.utc)

        message = turn.model_dump(mode="json", exclude_none=True)
        message.setdefault("message_id", str(uuid4()))
        message.setdefault("created_at", now.isoformat())

        update: dict = {
            "$push": {"messages": message},
            "$set": {"updated_at": now},
        }

        # The first user message becomes the conversation title.
        if turn.role == "user":
            doc = await conversation_collection.find_one(
                {"session_id": session_id},
                {"title": 1, "messages": {"$slice": 1}},
            )
            if doc and (
                not doc.get("messages")
                or doc.get("title") in (None, "", "Cuộc trò chuyện mới")
            ):
                title = " ".join(turn.content.strip().split()) or "Cuộc trò chuyện mới"
                if len(title) > 48:
                    title = title[:45].rstrip() + "..."
                update["$set"]["title"] = title

        await conversation_collection.update_one(
            {"session_id": session_id},
            update,
        )

    async def recent_history_text(
        self,
        session_id: str,
        limit: int = 6,
    ) -> str:
        doc = await conversation_collection.find_one(
            {"session_id": session_id},
            {"messages": {"$slice": -limit}},
        )
        if not doc:
            return ""

        return "\n".join(
            f"{message.get('role', '').upper()}: {message.get('content', '')}"
            for message in doc.get("messages", [])
        )

    async def recent_messages(
        self,
        session_id: str,
        limit: int = 12,
    ) -> list[dict]:
        """Các message gần nhất dạng {"role", "content"}, cũ trước mới sau."""
        doc = await conversation_collection.find_one(
            {"session_id": session_id},
            {"messages": {"$slice": -limit}},
        )
        if not doc:
            return []
        return _chat_messages(doc.get("messages", []))

    async def snapshot(self, session_id: str) -> SessionSnapshot:
        doc = await conversation_collection.find_one({"session_id": session_id})
        if not doc:
            return SessionSnapshot(session_id=session_id)

        detection = None
        if doc.get("last_detection"):
            detection = DetectionResult.model_validate(doc["last_detection"])

        turns = [ChatTurn.model_validate(item) for item in doc.get("messages", [])]
        return SessionSnapshot(
            session_id=session_id,
            last_detection=detection,
            turns=turns,
        )

    async def list_conversations(self, limit: int = 100) -> list[dict]:
        items: list[dict] = []
        cursor = conversation_collection.find(
            {},
            {
                "_id": 0,
                "session_id": 1,
                "title": 1,
                "created_at": 1,
                "updated_at": 1,
                "messages": {"$slice": -1},
            },
        ).sort("updated_at", -1).limit(limit)

        async for doc in cursor:
            last_message = None
            messages = doc.get("messages") or []
            if messages:
                last_message = messages[-1].get("content")

            items.append(
                {
                    "session_id": doc["session_id"],
                    "title": doc.get("title") or "Cuộc trò chuyện mới",
                    "last_message": last_message or "",
                    "created_at": doc.get("created_at"),
                    "updated_at": doc.get("updated_at"),
                }
            )
        return items

    async def conversation_detail(self, session_id: str) -> dict | None:
        doc = await conversation_collection.find_one(
            {"session_id": session_id},
            {"_id": 0},
        )
        if not doc:
            return None

        feedback_ids: list[ObjectId] = []
        for message in doc.get("messages", []):
            feedback_id = message.get("feedback_id")
            if feedback_id and ObjectId.is_valid(feedback_id):
                feedback_ids.append(ObjectId(feedback_id))

        feedback_map: dict[str, dict] = {}
        if feedback_ids:
            cursor = feedback_collection.find(
                {"_id": {"$in": feedback_ids}},
                {"user_feedback": 1},
            )
            async for feedback in cursor:
                feedback_map[str(feedback["_id"])] = feedback.get("user_feedback", {})

        for message in doc.get("messages", []):
            feedback_id = message.get("feedback_id")
            if feedback_id:
                state = feedback_map.get(feedback_id, {})
                message["feedback_rating"] = state.get("rating")
                message["feedback_reasons"] = state.get("reasons") or []

        return doc

    async def clear(self, session_id: str) -> None:
        await conversation_collection.delete_one({"session_id": session_id})



@dataclass
class _SessionData:
    last_detection: DetectionResult | None = None
    turns: list[ChatTurn] = field(default_factory=list)


class InMemorySessionStore:
    """Small RAM store kept for unit tests/backward compatibility.

    The running API uses MongoSessionStore.
    """

    def __init__(self, max_turns: int = 6):
        self._sessions: dict[str, _SessionData] = {}
        self._lock = asyncio.Lock()
        self.max_turns = max_turns

    async def _get_or_create(self, session_id: str) -> _SessionData:
        async with self._lock:
            return self._sessions.setdefault(session_id, _SessionData())

    async def get_last_detection(self, session_id: str) -> DetectionResult | None:
        data = await self._get_or_create(session_id)
        return data.last_detection.model_copy() if data.last_detection else None

    async def set_last_detection(self, session_id: str, detection: DetectionResult) -> None:
        data = await self._get_or_create(session_id)
        async with self._lock:
            data.last_detection = detection.model_copy()

    async def add_turn(self, session_id: str, turn: ChatTurn) -> None:
        data = await self._get_or_create(session_id)
        async with self._lock:
            data.turns.append(turn)
            if len(data.turns) > self.max_turns:
                data.turns = data.turns[-self.max_turns:]

    async def recent_history_text(self, session_id: str, limit: int = 6) -> str:
        data = await self._get_or_create(session_id)
        turns = data.turns[-limit:]
        return "\n".join(f"{turn.role.upper()}: {turn.content}" for turn in turns)

    async def recent_messages(self, session_id: str, limit: int = 12) -> list[dict]:
        data = await self._get_or_create(session_id)
        return _chat_messages(
            turn.model_dump(include={"role", "content"}) for turn in data.turns[-limit:]
        )

    async def snapshot(self, session_id: str) -> SessionSnapshot:
        data = await self._get_or_create(session_id)
        return SessionSnapshot(
            session_id=session_id,
            last_detection=(
                data.last_detection.model_copy() if data.last_detection else None
            ),
            turns=[turn.model_copy() for turn in data.turns],
        )

    async def clear(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)
