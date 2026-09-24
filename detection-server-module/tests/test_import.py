from app.answer.rag_http import RagHttpBackend
from app.config import Settings


def test_api_imports():
    import app.api  # noqa: F401


def test_rag_backend_is_config_only():
    from app.api import build_answer_backend

    backend = build_answer_backend(Settings(
        groq_api_key="x",
        answer_backend="rag",
        rag_api_url="http://rag:8010/",
        rag_api_key="k",
        rag_api_timeout=5,
    ))
    assert isinstance(backend, RagHttpBackend)
    assert backend.base_url == "http://rag:8010"
    assert backend.timeout == 5


def test_groq_is_the_default_backend(monkeypatch):
    from app.answer.groq import GroqAnswerBackend
    from app.api import build_answer_backend

    # .env cá nhân (đã nạp vào os.environ) không được ảnh hưởng mặc định trong code.
    monkeypatch.delenv("ANSWER_BACKEND", raising=False)

    settings = Settings(groq_api_key="x", _env_file=None)
    assert settings.answer_backend == "groq"
    assert isinstance(build_answer_backend(settings), GroqAnswerBackend)
