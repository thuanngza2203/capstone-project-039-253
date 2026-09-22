from motor.motor_asyncio import AsyncIOMotorClient

from app.config import get_settings

settings = get_settings()

client = AsyncIOMotorClient(settings.mongo_uri)
database = client[settings.mongo_db_name]

# One record per assistant answer. User feedback/admin review are updated here.
feedback_collection = database["chat_feedback"]

# One document per conversation/session. Messages are embedded in the document.
conversation_collection = database["chat_sessions"]

# Admin-curated QA embeddings used by Feedback RAG.
feedback_training_collection = database["feedback_training_vectors"]
