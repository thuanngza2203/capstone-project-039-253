from __future__ import annotations

import sys
from types import SimpleNamespace

import config
import main
import pytest
from langchain_core.documents import Document


def test_create_chat_model_fails_before_provider_import_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        config.create_chat_model()


def test_create_chat_model_builds_configured_ollama_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeChatOllama:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3.5:9b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test:11434")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "8192")
    monkeypatch.setenv("OLLAMA_NUM_PREDICT", "800")
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "30m")
    monkeypatch.setenv("OLLAMA_THINK", "false")
    monkeypatch.setitem(
        sys.modules,
        "langchain_ollama",
        SimpleNamespace(ChatOllama=FakeChatOllama),
    )

    model = config.create_chat_model()

    assert isinstance(model, FakeChatOllama)
    assert captured == {
        "model": "qwen3.5:9b",
        "base_url": "http://ollama.test:11434",
        "temperature": 0,
        "num_ctx": 8192,
        "num_predict": 800,
        "reasoning": False,
        "keep_alive": "30m",
        "validate_model_on_init": True,
    }


def test_create_chat_model_rejects_unknown_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "unknown")

    with pytest.raises(RuntimeError, match="LLM_PROVIDER không hợp lệ"):
        config.create_chat_model()


def test_cli_index_prints_counts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(main, "build_index", lambda: (2, 7))

    exit_code = main.run(["index"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Đã index 2 tài liệu thành 7 chunk" in captured.out
    assert captured.err == ""


def test_cli_search_prints_source_and_passes_top_k(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    call: dict[str, object] = {}

    def fake_retrieve(question: str, *, k: int):
        call.update(question=question, k=k)
        return [
            Document(
                page_content="Đốm xanh ô liu.",
                metadata={"source": "apple/apple_scab.txt"},
            )
        ]

    monkeypatch.setattr(main, "retrieve", fake_retrieve)

    exit_code = main.run(["search", "ghẻ táo", "--top-k", "2"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert call == {"question": "ghẻ táo", "k": 2}
    assert "[Nguồn 1: apple/apple_scab.txt]" in captured.out
    assert "Đốm xanh ô liu." in captured.out


def test_cli_ask_prints_answer_and_sources(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    call: dict[str, object] = {}

    def fake_ask(question: str, *, k: int):
        call.update(question=question, k=k)
        return "Câu trả lời [Nguồn 1].", ["apple/apple_scab.txt"]

    monkeypatch.setattr(main, "ask", fake_ask)

    exit_code = main.run(["ask", "ghẻ táo", "--top-k", "1"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert call == {"question": "ghẻ táo", "k": 1}
    assert "Câu trả lời [Nguồn 1]." in captured.out
    assert "Nguồn đã truy xuất:" in captured.out
    assert "- apple/apple_scab.txt" in captured.out


def test_cli_returns_nonzero_and_prints_clear_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail():
        raise RuntimeError("Không có dữ liệu")

    monkeypatch.setattr(main, "build_index", fail)

    exit_code = main.run(["index"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "Lỗi: Không có dữ liệu" in captured.err


def test_cli_search_passes_mode_and_reranker_override(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    captured = {}

    def search(question, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(main, "retrieve", search)
    assert main.run(["search", "ghẻ táo", "--mode", "bm25", "--no-rerank"]) == 0
    assert captured == {"k": 4, "mode": "bm25", "rerank": False}


def test_cli_debug_search_prints_trace_and_documents(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from retrieval import SearchHit, SearchResult, chunk_key

    doc = Document(page_content="Thông tin ghẻ táo", metadata={"source": "apple.txt"})
    hit = SearchHit(doc, chunk_key(doc), bm25_rank=1, bm25_score=2.5)
    result = SearchResult("bm25", [hit], [hit], 0, 1, 1, False)
    monkeypatch.setattr(main, "retrieve_with_debug", lambda *a, **kw: result)
    assert main.run(["search", "ghẻ táo", "--debug", "--mode", "bm25"]) == 0
    output = capsys.readouterr().out
    assert '"bm25_rank": 1' in output
    assert '"selected": true' in output
    assert "Thông tin ghẻ táo" in output


def test_cli_ask_passes_retrieval_options(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def ask(question, **kwargs):
        captured.update(kwargs)
        return "answer", []

    monkeypatch.setattr(main, "ask", ask)
    assert main.run(["ask", "ghẻ táo", "--mode", "hybrid", "--rerank"]) == 0
    assert captured == {"k": 4, "mode": "hybrid", "rerank": True}


def test_chat_reuses_session_and_continues_after_query_error(monkeypatch, capsys):
    calls = []
    questions = iter(["  ", "query bị lỗi", "câu hỏi tiếp", "/exit"])

    class FakeSession:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def warmup(self):
            calls.append(("warmup",))

        def ask(self, question):
            calls.append(("ask", question))
            if question == "query bị lỗi":
                raise RuntimeError("provider unavailable")
            return "câu trả lời", ["apple.txt"]

    monkeypatch.setattr(main, "RAGSession", FakeSession)
    monkeypatch.setattr("builtins.input", lambda _: next(questions))
    assert main.run(["chat", "--top-k", "2", "--mode", "hybrid", "--no-rerank"]) == 0
    assert calls == [
        ("init", {"k": 2, "mode": "hybrid", "rerank": False}),
        ("warmup",), ("ask", "query bị lỗi"), ("ask", "câu hỏi tiếp"),
    ]
    output = capsys.readouterr()
    assert "provider unavailable" in output.err
    assert "câu trả lời" in output.out
    assert "apple.txt" in output.out
    assert "Thời gian xử lý:" in output.out


def test_chat_search_only_prints_debug_without_calling_llm(monkeypatch, capsys):
    from retrieval import SearchHit, SearchResult, chunk_key

    doc = Document(page_content="Thông tin bệnh cây", metadata={"source": "plant.txt"})
    hit = SearchHit(doc, chunk_key(doc), bm25_rank=1, bm25_score=1.0)
    result = SearchResult("bm25", [hit], [hit], 0, 1, 1, False)
    questions = iter(["bệnh cây", "/quit"])
    session = SimpleNamespace(warmup=lambda: None, search=lambda question: result)
    monkeypatch.setattr(main, "RAGSession", lambda **kwargs: session)
    monkeypatch.setattr("builtins.input", lambda _: next(questions))
    assert main.run(["chat", "--search-only", "--mode", "bm25", "--debug"]) == 0
    output = capsys.readouterr().out
    assert '"bm25_rank": 1' in output
    assert "Thông tin bệnh cây" in output


@pytest.mark.parametrize("signal", [EOFError, KeyboardInterrupt])
def test_chat_exits_cleanly_on_input_signal(monkeypatch, signal):
    def stop(_):
        raise signal()

    monkeypatch.setattr(main, "RAGSession", lambda **kwargs: SimpleNamespace(warmup=lambda: None))
    monkeypatch.setattr("builtins.input", stop)
    assert main.run(["chat"]) == 0


def test_chat_rejects_invalid_options_before_loading(monkeypatch, capsys):
    import rag

    monkeypatch.setattr(rag, "_load_vector_store", lambda *a, **kw: pytest.fail("Không được nạp model"))
    assert main.run(["chat", "--history-turns", "-1"]) == 1
    assert "CHAT_HISTORY_TURNS" in capsys.readouterr().err


def test_chat_reports_warmup_failure_and_exits(monkeypatch, capsys):
    def fail():
        raise RuntimeError("Index đang rỗng")

    monkeypatch.setattr(main, "RAGSession", lambda **kwargs: SimpleNamespace(warmup=fail))
    assert main.run(["chat"]) == 1
    assert "Index đang rỗng" in capsys.readouterr().err


def test_chat_debug_and_reset_keep_the_same_session(monkeypatch, capsys):
    calls = []
    questions = iter(["bệnh ghẻ táo", "/reset", "bệnh khoai tây", "/exit"])

    class FakeSession:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def warmup(self):
            calls.append(("warmup",))

        def clear_history(self):
            calls.append(("reset",))

        def ask_with_debug(self, question):
            calls.append(("ask", question))
            return SimpleNamespace(
                answer="answer", sources=["source.txt"],
                to_debug_dict=lambda: {"retrieval_query": question, "history_turns_used": 0},
            )

    monkeypatch.setattr(main, "RAGSession", FakeSession)
    monkeypatch.setattr("builtins.input", lambda _: next(questions))
    assert main.run(["chat", "--debug", "--history-turns", "2"]) == 0
    assert calls == [
        ("init", {"k": 4, "history_turns": 2}), ("warmup",),
        ("ask", "bệnh ghẻ táo"), ("reset",), ("ask", "bệnh khoai tây"),
    ]
    output = capsys.readouterr().out
    assert '"retrieval_query": "bệnh khoai tây"' in output
    assert "Đã xóa lịch sử" in output
