"""Công cụ thí nghiệm: generate (chạy tiếp, chặn trộn cấu hình), survey build/analyze, stats.

Gọi RAG API thật qua TestClient; server LLM giả qua httpx.MockTransport. Không mạng, không model.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from experiments import stats, survey
from experiments.generate import ExperimentError, RunConfig, Runner, load_questions, load_rows
from server.app import create_app
from server.runtime import RAGRuntime
from test_api_server import AUTH, DOCS, KEY, FilterStore

QUESTIONS = [
    {"id": "q1", "question": "Bệnh ghẻ táo xử lý thế nào?", "plant_type": "apple", "disease": "apple_scab",
     "topic": "không gửi sang API"},
    {"id": "q2", "question": "Cháy sớm khoai tây xử lý ra sao? FAIL"},
    {"id": "q3", "question": "Mốc sương cà chua xử lý?"},
]


class FakeLLM:
    """Trả lời khác nhau theo câu hỏi; `fail` làm câu chứa FAIL lỗi như tunnel đứt."""

    def __init__(self, label: str) -> None:
        self.label, self.fail, self.calls = label, False, 0
        self.runnable = RunnableLambda(self._call)

    def _call(self, prompt_value) -> AIMessage:
        text = prompt_value.to_string()
        self.calls += 1
        question = text.split("CÂU HỎI:\n", 1)[1].split("\n", 1)[0]
        if self.fail and "FAIL" in question:
            raise ConnectionError("tunnel đứt")
        # Dựa vào chunk đầu: hai index xếp chunk khác nhau thì câu trả lời khác nhau.
        first = re.search(r"\[Nguồn 1: [^\]]+\]\n([^\n]+)", text)
        return AIMessage(content=f"**{self.label}**: {question} → {first.group(1) if first else ''} [Nguồn 1].")


class LLMServer:
    """vLLM giả: chỉ trả /v1/models; `root` đổi được để giả lập đổi model trên Vast."""

    def __init__(self, root: str = "Qwen/Qwen3.5-4B") -> None:
        self.root = root

    def __call__(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "rag-llm", "root": self.root}]})


@pytest.fixture
def vllm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "vllm")
    monkeypatch.setenv("VLLM_HOST", "127.0.0.1")
    monkeypatch.setenv("VLLM_PORT", "8001")
    monkeypatch.setenv("VLLM_MODEL", "rag-llm")


def api(llm: FakeLLM, server: LLMServer) -> TestClient:
    runtime = RAGRuntime(
        stores={"recursive": FilterStore(DOCS[::-1]), "structure": FilterStore(DOCS)},
        llms={"vllm": llm.runnable},
        http_client=httpx.Client(transport=httpx.MockTransport(server)),
    )
    return TestClient(create_app(runtime, api_key=KEY), headers=AUTH)


def write_questions(tmp_path: Path, questions=QUESTIONS) -> Path:
    path = tmp_path / "questions.jsonl"
    path.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in questions), encoding="utf-8")
    return path


def runner(client: TestClient, tmp_path: Path, variant: str, **config) -> Runner:
    # Store giả không có manifest nên luôn cần allow_stale; test riêng bên dưới kiểm tra việc chặn.
    return Runner(client, RunConfig(variant, **config), tmp_path / "runs", allow_stale=True,
                  retries=0, backoff_seconds=0, log=lambda _: None)


# --- generate -----------------------------------------------------------------

def test_generate_records_answers_and_resumes_only_failed(tmp_path: Path, vllm_env) -> None:
    llm, questions_path = FakeLLM("A"), write_questions(tmp_path)
    client = api(llm, LLMServer())
    llm.fail = True
    first = runner(client, tmp_path, "qwen4b-structure", index="structure").run(questions_path, QUESTIONS)
    assert first["answered"] == 2 and first["errors"] == 1

    rows = load_rows(tmp_path / "runs/qwen4b-structure.jsonl")
    assert rows["q2"]["error"]["status_code"] == 502
    q1 = rows["q1"]
    assert q1["request"] == {"query": QUESTIONS[0]["question"], "debug": True, "plant_type": "apple",
                             "disease": "apple_scab", "index": "structure"}  # `topic` không gửi đi
    assert q1["scope_status"] == "document" and q1["meta"]["index"] == "structure"
    assert q1["chunks"] and set(q1["chunks"][0]) == {"rank", "source", "heading_path", "content"}

    llm.fail, calls = False, llm.calls
    second = runner(client, tmp_path, "qwen4b-structure", index="structure").run(questions_path, QUESTIONS)
    assert llm.calls == calls + 1  # chỉ chạy lại q2
    assert second["answered"] == 3 and second["errors"] == 0
    lines = (tmp_path / "runs/qwen4b-structure.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["question_id"] for line in lines] == ["q1", "q2", "q3"]

    run = json.loads((tmp_path / "runs/qwen4b-structure.run.json").read_text(encoding="utf-8"))
    assert run["model"] == "Qwen/Qwen3.5-4B" and run["index"] == "structure" and run["llm_provider"] == "vllm"
    assert run["settings"]["top_k"] == 4 and len(run["sessions"]) == 2


def test_generate_refuses_to_mix_models_behind_the_same_alias(tmp_path: Path, vllm_env) -> None:
    server, questions_path = LLMServer("Qwen/Qwen3.5-4B"), write_questions(tmp_path)
    client = api(FakeLLM("A"), server)
    runner(client, tmp_path, "run-1").run(questions_path, QUESTIONS[:1])

    server.root = "Qwen/Qwen3-32B-FP8"  # đổi model trên Vast, tên API vẫn là rag-llm
    with pytest.raises(ExperimentError, match="Qwen/Qwen3.5-4B.*Qwen/Qwen3-32B-FP8.*--variant mới"):
        runner(client, tmp_path, "run-1").run(questions_path, QUESTIONS)


def test_generate_refuses_index_not_built_from_current_data(tmp_path: Path, vllm_env) -> None:
    client = api(FakeLLM("A"), LLMServer())
    strict = Runner(client, RunConfig("x", index="structure"), tmp_path / "runs", log=lambda _: None)
    with pytest.raises(ExperimentError, match="allow-stale"):
        strict.run(write_questions(tmp_path), QUESTIONS)


def test_generate_stops_when_server_settings_change_midway(tmp_path: Path) -> None:
    checker = runner(TestClient(create_app(RAGRuntime(vector_store=FilterStore(DOCS)))), tmp_path, "x")
    run = {"settings": {"index": "structure", "chunking_strategy": "structure", "retrieval_mode": "hybrid",
                        "reranker_enabled": False, "top_k": 4, "llm_provider": "vllm"}}
    meta = {"index": "structure", "chunking_strategy": "structure", "retrieval_mode": "hybrid",
            "reranker_enabled": True, "top_k": 4, "llm": {"provider": "vllm"}}
    with pytest.raises(ExperimentError, match="reranker_enabled"):
        checker.check_settings(run, {"meta": meta})


def test_load_questions_rejects_duplicates_and_missing_fields(tmp_path: Path) -> None:
    with pytest.raises(ExperimentError, match="trùng id"):
        load_questions(write_questions(tmp_path, [QUESTIONS[0], QUESTIONS[0]]))
    with pytest.raises(ExperimentError, match=":2: cần có `id` và `question`"):
        load_questions(write_questions(tmp_path, [QUESTIONS[0], {"id": "x"}]))


# --- survey -------------------------------------------------------------------

def test_plain_text_removes_markdown_and_citations() -> None:
    text = "## Xử lý\n\n**Thu gom** lá bệnh [Nguồn 1].\n- Phun *thuốc* gốc đồng [Nguồn 2, 3]\n  - Lần 2\n---\n\n\n\nHết."
    assert survey.plain_text(text) == "Xử lý\n\nThu gom lá bệnh.\n• Phun thuốc gốc đồng\n   ◦ Lần 2\n\nHết."
    assert "[Nguồn 1]" in survey.plain_text(text, keep_citations=True)


def row(qid: str, variant: str, answer: str, **extra) -> dict:
    meta = {"llm": {"truncated": extra.pop("truncated", False)}}
    return {"question_id": qid, "variant": variant, "question": f"Câu hỏi {qid}", "answer": answer,
            "meta": meta, **extra}


def test_build_pairs_filters_and_balances_positions() -> None:
    rows_a = {f"q{i}": row(f"q{i}", "A", f"A trả lời {i}") for i in range(12)}
    rows_b = {f"q{i}": row(f"q{i}", "B", f"B trả lời {i}") for i in range(12)}
    rows_b["q9"] = row("q9", "B", "A trả lời 9")  # giống hệt → bỏ
    rows_b["q10"] = {**rows_b["q10"], "error": {"status_code": 502}}
    rows_b["q11"] = row("q11", "B", "B bị cắt", truncated=True)
    pairs, excluded = survey.build_pairs(rows_a, rows_b)
    assert [e["question_id"] for e in excluded] == ["q9", "q10", "q11"]
    assert len(pairs) == 9

    forms = survey.assign(pairs, seed=7, per_form=4)
    assert [len(form) for form in forms] == [3, 3, 3]
    items = [item for form in forms for item in form]
    firsts = [item["first"] for item in items]
    assert abs(firsts.count("a") - firsts.count("b")) == 1
    for item in items:  # câu trả lời ở vị trí 1 đúng là của variant `first`
        assert item["answer1"].startswith(item["first"].upper())
    assert survey.assign(pairs, seed=7, per_form=4) == forms  # cùng seed, cùng kết quả


def write_run(path: Path, variant: str, answers: dict[str, str]) -> None:
    path.write_text("".join(json.dumps(row(qid, variant, text), ensure_ascii=False) + "\n"
                            for qid, text in answers.items()), encoding="utf-8")


def test_build_writes_form_script_key_and_preview(tmp_path: Path) -> None:
    write_run(tmp_path / "a.jsonl", "qwen4b", {"q1": "**Có** [Nguồn 1]", "q2": "A2"})
    write_run(tmp_path / "b.jsonl", "gemini", {"q1": "Không", "q2": "B2"})
    key = survey.build(tmp_path / "a.jsonl", tmp_path / "b.jsonl", tmp_path / "out",
                       title="Thử", seed=1, per_form=12)
    assert key["variants"] == {"a": "qwen4b", "b": "gemini"}
    script = (tmp_path / "out/form.gs").read_text(encoding="utf-8")
    forms = json.loads(re.search(r"const FORMS = (\[.*?\]);\n\nfunction", script, re.DOTALL).group(1))
    for item, spec in zip(key["items"], forms[0]["items"]):
        assert spec["n"] == item["n"]
        expected = "Có" if item["question_id"] == "q1" else "A2"
        assert (spec["answer1"] if item["first"] == "a" else spec["answer2"]) == expected
    assert "qwen4b" not in script and "gemini" not in script  # form không lộ variant
    assert "function createForms()" in script
    assert "Câu trả lời 1" in (tmp_path / "out/preview.md").read_text(encoding="utf-8")


def google_csv(path: Path, key: dict, answers: list[tuple[str, list[str]]]) -> None:
    headers = ["Dấu thời gian", key["background"]["title"]] + [
        f"Câu {item['n']}. {key['item_question']}" for item in key["items"]]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:  # Google xuất CSV có BOM
        writer = csv.writer(handle)
        writer.writerow(headers)
        for group, choices in answers:
            writer.writerow(["2026/09/24 10:00:00", group, *choices])


def survey_fixture() -> tuple[dict, list[tuple[str, list[str]]]]:
    key = {"variants": {"a": "A", "b": "B"}, "choices": list(survey.CHOICES),
           "item_question": survey.ITEM_QUESTION, "background": survey.BACKGROUND,
           "items": [{"n": 1, "question_id": "q1", "question": "x", "first": "a"},
                     {"n": 2, "question_id": "q2", "question": "y", "first": "b"}]}
    one, two, tie = survey.CHOICES
    farmer, expert = survey.BACKGROUND["choices"][:2]
    return key, [(farmer, [one, one]), (farmer, [one, tie]), (expert, [two, one])]


def test_analyze_end_to_end(tmp_path: Path) -> None:
    key, answers = survey_fixture()
    google_csv(tmp_path / "responses.csv", key, answers)
    result = survey.analyze(key, [tmp_path / "responses.csv"])
    # Câu 1: A ở vị trí 1 → chọn "1" là A. Câu 2: B ở vị trí 1 → chọn "1" là B.
    assert result["respondents"] == 3 and result["judgments"] == 6
    # Người 1: câu 1 → A, câu 2 → B. Người 2: A, hòa. Người 3: câu 1 chọn "2" → B, câu 2 → B.
    assert (result["overall"]["a"], result["overall"]["b"], result["overall"]["tie"]) == (2, 3, 1)
    assert result["items"][0]["a"] == 2 and result["items"][1]["b"] == 2
    assert result["by_question"] == {"a": 1, "b": 1, "tie": 0, "p_value": 1.0}
    farmer = survey.BACKGROUND["choices"][0]
    assert (result["groups"][farmer]["a"], result["groups"][farmer]["b"]) == (2, 1)
    assert "A = `A`, B = `B`" in survey.report(result)


def test_analyze_rejects_csv_of_another_form(tmp_path: Path) -> None:
    key, _ = survey_fixture()
    (tmp_path / "other.csv").write_text("Dấu thời gian,Tên\n1,x\n", encoding="utf-8")
    with pytest.raises(survey.SurveyError, match="không thấy cột"):
        survey.analyze(key, [tmp_path / "other.csv"])


def test_sign_test_and_wilson_match_known_values() -> None:
    assert survey.sign_test(8, 2) == pytest.approx(112 / 1024)
    assert survey.sign_test(5, 5) == 1.0 and survey.sign_test(0, 0) == 1.0
    low, high = survey.wilson(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-4) and high == pytest.approx(0.7634, abs=1e-4)


# --- Từ generate tới khảo sát ---------------------------------------------------

def test_two_chunking_variants_to_survey_and_stats(tmp_path: Path, vllm_env) -> None:
    llm, questions_path = FakeLLM("RAG"), write_questions(tmp_path)
    client = api(llm, LLMServer())
    for index in ("recursive", "structure"):
        summary = runner(client, tmp_path, f"qwen4b-{index}", index=index).run(questions_path, QUESTIONS)
        assert summary["errors"] == 0

    runs = tmp_path / "runs"
    key = survey.build(runs / "qwen4b-structure.jsonl", runs / "qwen4b-recursive.jsonl",
                       tmp_path / "survey", title="Chunking", seed=3, per_form=12)
    # q1 lọc theo đúng tài liệu ghẻ táo nên hai index có thể cho cùng câu trả lời (bị loại);
    # các câu còn lại lấy chunk đầu khác nhau nên thành cặp khảo sát.
    assert key["items"] and len(key["items"]) + len(key["excluded"]) == len(QUESTIONS)
    assert all(e["reason"] == "hai câu trả lời giống hệt nhau" for e in key["excluded"])
    table = stats.table([stats.describe(runs / "qwen4b-structure.jsonl"),
                         stats.describe(runs / "qwen4b-recursive.jsonl")])
    assert "qwen4b-structure" in table and "Qwen/Qwen3.5-4B" in table and "| structure |" in table

    csv_path, md_path = survey.export(runs / "qwen4b-structure.jsonl", runs / "qwen4b-recursive.jsonl",
                                      tmp_path / "export")
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        exported = list(csv.DictReader(handle))
    assert [r["id"] for r in exported] == ["q1", "q2", "q3"]
    assert exported[0]["model_a"] == "qwen4b-structure (Qwen/Qwen3.5-4B)"
    assert exported[0]["answer_a"].startswith("**RAG**")  # nguyên văn, không bỏ markdown
    assert "## q1. " in md_path.read_text(encoding="utf-8")
