"""Mục 5 (P2) của plan: Swagger đầy đủ, /admin có xác thực, MAX_HISTORY_TURNS có tác dụng."""

import asyncio
import base64

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from app.config import get_settings
from app.schemas import ChatTurn, Intent, ReviewItem


@pytest.fixture
def client():
    from app.api import app

    # Không dùng `with`: không chạy lifespan (tải checkpoint) trong test.
    return TestClient(app)


@pytest.fixture
def admin_password(monkeypatch):
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "s3cret")
    get_settings.cache_clear()
    yield "s3cret"
    get_settings.cache_clear()


def basic(user, password):
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def test_every_api_route_has_response_model_and_tag():
    from app.api import app

    routes = [r for r in app.routes if isinstance(r, APIRoute) and r.include_in_schema]
    assert routes
    missing = [r.path for r in routes if r.response_model is None or not r.tags]
    assert missing == []


def test_admin_requires_login_when_password_is_set(client, admin_password):
    assert client.get("/admin/status").status_code == 401
    assert client.get("/admin/status", headers=basic("admin", "sai")).status_code == 401
    assert client.get("/admin").status_code == 401
    response = client.get("/admin/status", headers=basic("admin", admin_password))
    assert response.status_code == 200
    assert response.json() == {"answer_backend": "groq", "feedback_examples_used": True}
    assert client.get("/admin", headers=basic("admin", admin_password)).status_code == 200
    # Trang chat và health không bị khóa.
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200


def test_admin_is_open_without_password(client, monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "")
    monkeypatch.delenv("ANSWER_BACKEND", raising=False)
    get_settings.cache_clear()
    try:
        assert client.get("/admin/status").status_code == 200
    finally:
        get_settings.cache_clear()


def test_review_item_keeps_mongo_id_and_extra_fields():
    raw = {"_id": "abc", "question": "q", "answer": "a", "image_filename": "la.jpg",
           "metadata": {"sources": ["apple/apple_scab.txt"]}}
    dumped = ReviewItem.model_validate(raw).model_dump(by_alias=True)
    assert dumped["_id"] == "abc"
    assert dumped["image_filename"] == "la.jpg"
    assert dumped["metadata"]["sources"] == ["apple/apple_scab.txt"]


def test_max_history_turns_limits_llm_history():
    from app.chat.session import InMemorySessionStore
    from test_pipeline import Harness, analysis

    harness = Harness()
    harness.service.sessions = harness.sessions = InMemorySessionStore(max_turns=2)
    seen = {}
    original = harness.normalizer.analyze

    async def capture(raw_query, session_context):
        seen["context"] = session_context
        return await original(raw_query, session_context)

    harness.normalizer.analyze = capture
    for text in ["câu 1", "đáp 1", "câu 2", "đáp 2"]:
        role = "user" if text.startswith("câu") else "assistant"
        asyncio.run(harness.sessions.add_turn("s1", ChatTurn(role=role, content=text)))

    harness.chat("hỏi tiếp", analysis(plant="apple", disease="apple_scab", intent=Intent.TREATMENT))
    assert "câu 2" in seen["context"] and "đáp 2" in seen["context"]
    assert "câu 1" not in seen["context"]


def test_chat_route_returns_400_with_message_for_invalid_image(client):
    from app.api import app, get_chat_service
    from app.plant_ai.pipeline.prediction import NOT_A_LEAF, InvalidImageError

    class Service:
        async def chat(self, **kwargs):
            raise InvalidImageError(NOT_A_LEAF)

    app.dependency_overrides[get_chat_service] = lambda: Service()
    try:
        response = client.post(
            "/api/chat",
            data={"session_id": "s1", "message": ""},
            files={"image": ("meo.jpg", b"fake", "image/jpeg")},
        )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 400
    assert response.json() == {"detail": NOT_A_LEAF}


def test_reviews_filter_defaults_to_rated_and_can_include_all():
    from datetime import datetime, timezone

    from app.routes.admin import review_filter

    assert review_filter() == {"user_feedback.rating": {"$in": ["like", "unlike"]}}
    since = datetime(2026, 9, 1, tzinfo=timezone.utc)
    assert review_filter(include_all=True, since=since) == {"created_at": {"$gte": since}}


def test_reviews_accepts_all_from_to_params():
    from app.api import app

    params = {p["name"] for p in app.openapi()["paths"]["/admin/reviews"]["get"]["parameters"]}
    assert {"all", "from", "to", "limit"} <= params
