from __future__ import annotations

import config
import main
import pytest
from langchain_core.documents import Document


def test_create_chat_model_fails_before_provider_import_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
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

