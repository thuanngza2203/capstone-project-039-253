"""Khảo sát ẩn danh so sánh hai variant bằng Google Form.

build   : ghép câu trả lời của hai variant thành từng cặp, đảo vị trí 1/2 có cân bằng,
          sinh Apps Script tạo form (`form.gs`), file khóa (`key.json`) và bản xem trước.
export  : bảng so sánh có tên model (side_by_side.csv/.md) để người làm khảo sát đọc.
analyze : đọc CSV phản hồi (Google Form → Phản hồi → Tải xuống CSV), giải mã bằng
          `key.json`, tính tỉ lệ thắng, khoảng tin cậy và kiểm định dấu.

    python -m experiments.survey build --a experiments/runs/qwen4b-structure.jsonl \\
        --b experiments/runs/qwen4b-recursive.jsonl --out experiments/surveys/chunking
    python -m experiments.survey analyze --key experiments/surveys/chunking/key.json \\
        --responses "Khảo sát (1_2).csv" "Khảo sát (2_2).csv"
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.generate import load_rows

CHOICES = ("Câu trả lời 1 tốt hơn", "Câu trả lời 2 tốt hơn", "Hai câu ngang nhau")
ITEM_QUESTION = "Câu trả lời nào tốt hơn?"
BACKGROUND = {
    "title": "Bạn là",
    "choices": ["Nông dân hoặc người trồng cây", "Sinh viên, kỹ sư hoặc chuyên gia nông nghiệp", "Khác"],
}
DESCRIPTION = (
    "Mỗi trang có một câu hỏi về bệnh cây trồng và hai câu trả lời do hai hệ thống khác nhau viết. "
    "Hãy chọn câu trả lời bạn thấy đúng, rõ ràng và hữu ích hơn cho người trồng cây. "
    "Nếu không phân biệt được, chọn \"Hai câu ngang nhau\". Không cần tra cứu thêm."
)
SEPARATOR = "——————————"

# [Nguồn 1], [Nguồn 1, 2], [Nguồn 2: apple/apple_scab.txt]: người khảo sát không thấy danh sách nguồn.
_CITATION = re.compile(r"\s*\[\s*Nguồn\s*\d+[^\]]*\]", re.IGNORECASE)
_ITEM_COLUMN = re.compile(r"^\s*Câu\s+(\d+)\.")


class SurveyError(RuntimeError):
    pass


def plain_text(text: str, *, keep_citations: bool = False) -> str:
    """Google Form chỉ hiện chữ thường: bỏ markdown, giữ xuống dòng và gạch đầu dòng."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if not keep_citations:
        text = _CITATION.sub("", text)
    lines = []
    for line in text.splitlines():
        indent = len(line) - len(line.lstrip())
        body = line.strip()
        if re.fullmatch(r"[-*_]{3,}", body):
            continue
        body = re.sub(r"^#{1,6}\s+", "", body)
        bullet = re.match(r"^[-*+]\s+", body)
        if bullet:
            body = ("   ◦ " if indent >= 2 else "• ") + body[bullet.end():]
        body = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: m.group(1) or m.group(2), body)
        body = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\1", body)
        body = body.replace("`", "")
        lines.append(re.sub(r"[ \t]+([.,;:!?])", r"\1", body))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _truncated(row: dict[str, Any]) -> bool:
    return bool((row.get("meta", {}).get("llm") or {}).get("truncated"))


def build_pairs(
    rows_a: dict[str, dict], rows_b: dict[str, dict], *,
    keep_citations: bool = False, include_truncated: bool = False,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    pairs, excluded = [], []
    for qid, a in rows_a.items():
        b = rows_b.get(qid)
        reason = None
        if b is None:
            reason = "chỉ variant A có câu này"
        elif a["question"] != b["question"]:
            raise SurveyError(f"{qid}: hai variant có nội dung câu hỏi khác nhau; không ghép được.")
        elif "error" in a or "error" in b:
            reason = "lỗi khi sinh câu trả lời"
        elif not include_truncated and (_truncated(a) or _truncated(b)):
            sides = "+".join(side for side, row in (("A", a), ("B", b)) if _truncated(row))
            reason = f"câu trả lời bị cắt ({sides})"
        if reason is None:
            text_a = plain_text(a["answer"], keep_citations=keep_citations)
            text_b = plain_text(b["answer"], keep_citations=keep_citations)
            if text_a == text_b:
                reason = "hai câu trả lời giống hệt nhau"
            else:
                pairs.append({"question_id": qid, "question": a["question"], "a": text_a, "b": text_b})
        if reason:
            excluded.append({"question_id": qid, "reason": reason})
    excluded += [{"question_id": qid, "reason": "chỉ variant B có câu này"} for qid in rows_b if qid not in rows_a]
    return pairs, excluded


def assign(pairs: list[dict[str, str]], *, seed: int, per_form: int) -> list[list[dict[str, Any]]]:
    """Trộn thứ tự câu; mỗi variant đứng ở vị trí 1 đúng một nửa số cặp (lệch tối đa 1)."""
    if per_form < 1:
        raise SurveyError("--per-form phải lớn hơn 0.")
    rng = random.Random(seed)
    order = pairs[:]
    rng.shuffle(order)
    firsts = ["a"] * (len(order) // 2) + ["b"] * (len(order) // 2)
    if len(order) % 2:
        firsts.append(rng.choice("ab"))
    rng.shuffle(firsts)
    items = []
    for number, (pair, first) in enumerate(zip(order, firsts), start=1):
        second = "b" if first == "a" else "a"
        items.append({"n": number, "question_id": pair["question_id"], "question": pair["question"],
                      "first": first, "answer1": pair[first], "answer2": pair[second]})
    # Chia đều: 30 câu, tối đa 12/form → 3 form × 10 câu, không phải 12 + 12 + 6.
    count = max(1, math.ceil(len(items) / per_form))
    base, extra = divmod(len(items), count)
    forms, start = [], 0
    for index in range(count):
        size = base + (1 if index < extra else 0)
        forms.append(items[start:start + size])
        start += size
    return forms


def apps_script(forms: list[list[dict[str, Any]]], title: str, created: str) -> str:
    specs = [{
        "title": title if len(forms) == 1 else f"{title} ({index}/{len(forms)})",
        "items": [{k: item[k] for k in ("n", "question", "answer1", "answer2")} for item in form],
    } for index, form in enumerate(forms, start=1)]

    def js(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, indent=1)

    return f"""// Tạo bởi experiments/survey.py lúc {created}.
// Cách dùng: vào https://script.google.com → Dự án mới → xóa code mẫu, dán toàn bộ file này,
// chọn hàm createForms rồi bấm Chạy (lần đầu Google hỏi quyền tạo Form). Link các form
// in ở "Nhật ký thực thi". Không sửa thứ tự hay số câu: key.json giải mã theo đúng số này.

const DESCRIPTION = {js(DESCRIPTION)};
const BACKGROUND = {js(BACKGROUND)};
const CHOICES = {js(list(CHOICES))};
const ITEM_QUESTION = {js(ITEM_QUESTION)};
const SEPARATOR = {js(SEPARATOR)};
const FORMS = {js(specs)};

function createForms() {{
  const links = [];
  FORMS.forEach(function (spec) {{
    const form = FormApp.create(spec.title);
    form.setDescription(DESCRIPTION)
        .setProgressBar(true)
        .setCollectEmail(false)
        .setShuffleQuestions(false);
    form.addMultipleChoiceItem()
        .setTitle(BACKGROUND.title)
        .setChoiceValues(BACKGROUND.choices)
        .setRequired(true);
    spec.items.forEach(function (item) {{
      form.addPageBreakItem()
          .setTitle('Câu ' + item.n + '. ' + item.question)
          .setHelpText('CÂU TRẢ LỜI 1\\n\\n' + item.answer1 + '\\n\\n' + SEPARATOR +
                       '\\n\\nCÂU TRẢ LỜI 2\\n\\n' + item.answer2);
      form.addMultipleChoiceItem()
          .setTitle('Câu ' + item.n + '. ' + ITEM_QUESTION)
          .setChoiceValues(CHOICES)
          .setRequired(true);
    }});
    links.push(spec.title + '\\n  Gửi người khảo sát: ' + form.getPublishedUrl() +
               '\\n  Sửa form: ' + form.getEditUrl());
  }});
  Logger.log(links.join('\\n\\n'));
}}
"""


def preview(forms: list[list[dict[str, Any]]], title: str) -> str:
    parts = [f"# {title}\n\n{DESCRIPTION}\n\nCâu nền: **{BACKGROUND['title']}**: "
             + " / ".join(BACKGROUND["choices"])]
    for index, form in enumerate(forms, start=1):
        parts.append(f"\n## Form {index}/{len(forms)} ({len(form)} câu)")
        for item in form:
            parts.append(
                f"\n### Câu {item['n']}. {item['question']}\n\n**Câu trả lời 1**\n\n{item['answer1']}\n\n"
                f"**Câu trả lời 2**\n\n{item['answer2']}\n\n_{ITEM_QUESTION}_ " + " / ".join(CHOICES)
            )
    return "\n".join(parts) + "\n"


def build(
    path_a: Path, path_b: Path, out_dir: Path, *, title: str, seed: int, per_form: int,
    keep_citations: bool = False, include_truncated: bool = False,
) -> dict[str, Any]:
    rows_a, rows_b = load_rows(path_a), load_rows(path_b)
    if not rows_a or not rows_b:
        raise SurveyError("Một trong hai file không có câu trả lời nào.")
    variant_a = next(iter(rows_a.values()))["variant"]
    variant_b = next(iter(rows_b.values()))["variant"]
    if variant_a == variant_b:
        raise SurveyError("Hai file là cùng một variant.")
    pairs, excluded = build_pairs(rows_a, rows_b, keep_citations=keep_citations,
                                  include_truncated=include_truncated)
    if not pairs:
        raise SurveyError("Không còn cặp nào để khảo sát sau khi lọc.")
    forms = assign(pairs, seed=seed, per_form=per_form)
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    key = {
        "created_at": created, "title": title, "seed": seed,
        "variants": {"a": variant_a, "b": variant_b},
        "files": {"a": str(path_a), "b": str(path_b)},
        "keep_citations": keep_citations, "include_truncated": include_truncated,
        "choices": list(CHOICES), "item_question": ITEM_QUESTION, "background": BACKGROUND,
        "items": [{"n": item["n"], "form": index, "question_id": item["question_id"],
                   "question": item["question"], "first": item["first"]}
                  for index, form in enumerate(forms, start=1) for item in form],
        "excluded": excluded,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "form.gs").write_text(apps_script(forms, title, created), encoding="utf-8")
    (out_dir / "preview.md").write_text(preview(forms, title), encoding="utf-8")
    return key


# --- Bảng so sánh có tên model (cho người làm khảo sát đọc, không đưa vào form) ---------

def _run_info(path: Path) -> dict[str, Any]:
    run_path = path.with_suffix(".run.json")
    return json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {}


def _llm(row: dict[str, Any]) -> dict[str, Any]:
    return (row.get("meta") or {}).get("llm") or {}


def export(path_a: Path, path_b: Path, out_dir: Path) -> tuple[Path, Path]:
    """Ghi side_by_side.csv và side_by_side.md: câu hỏi, câu trả lời gốc của hai model, nguồn, số liệu."""
    rows_a, rows_b = load_rows(path_a), load_rows(path_b)
    names = {}
    for side, path, rows in (("a", path_a, rows_a), ("b", path_b, rows_b)):
        info = _run_info(path)
        variant = info.get("variant") or next(iter(rows.values()))["variant"]
        names[side] = f"{variant} ({info['model']})" if info.get("model") else variant
    records = []
    for qid, a in rows_a.items():
        b = rows_b.get(qid, {})
        chunks = a.get("chunks") or b.get("chunks") or []
        records.append({
            "id": qid, "question": a["question"],
            "model_a": names["a"], "answer_a": a.get("answer", f"LỖI: {a.get('error')}"),
            "model_b": names["b"], "answer_b": b.get("answer", f"LỖI: {b.get('error')}" if b else "không có"),
            "sources_a": " | ".join(a.get("sources", [])), "sources_b": " | ".join(b.get("sources", [])),
            "same_context": [c["content"] for c in a.get("chunks") or []] == [c["content"] for c in b.get("chunks") or []],
            "retrieved_sections": " | ".join(f"{c['source']} › {c.get('heading_path') or ''}".rstrip(" ›") for c in chunks),
            "generate_ms_a": ((a.get("meta") or {}).get("timing_ms") or {}).get("generate"),
            "generate_ms_b": ((b.get("meta") or {}).get("timing_ms") or {}).get("generate"),
            "output_tokens_a": _llm(a).get("output_tokens"), "output_tokens_b": _llm(b).get("output_tokens"),
            "truncated_a": _llm(a).get("truncated"), "truncated_b": _llm(b).get("truncated"),
            "invalid_citations_a": (a.get("citations") or {}).get("invalid"),
            "invalid_citations_b": (b.get("citations") or {}).get("invalid"),
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, md_path = out_dir / "side_by_side.csv", out_dir / "side_by_side.md"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:  # Excel đọc đúng tiếng Việt
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        for record in records:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v
                             for k, v in record.items()})
    parts = [f"# So sánh câu trả lời: {names['a']} và {names['b']}", "",
             "Bản có tên model, để người làm khảo sát đọc. Form gửi người trả lời lấy từ `survey build` "
             "(ẩn tên, đảo vị trí). Câu trả lời giữ nguyên văn, kể cả markdown và `[Nguồn n]`.", ""]
    for record in records:
        timing = lambda side: (f"{record[f'generate_ms_{side}']} ms, {record[f'output_tokens_{side}']} token ra"
                               + (" · BỊ CẮT" if record[f"truncated_{side}"] else ""))
        parts += [f"## {record['id']}. {record['question']}", "",
                  f"Ngữ cảnh RAG ({'giống nhau cho hai model' if record['same_context'] else 'KHÁC nhau'}): "
                  f"{record['retrieved_sections']}", "",
                  f"### {record['model_a']} — {timing('a')}", "", record["answer_a"], "",
                  f"### {record['model_b']} — {timing('b')}", "", record["answer_b"], ""]
    md_path.write_text("\n".join(parts), encoding="utf-8")
    return csv_path, md_path


# --- Phân tích ------------------------------------------------------------------

def sign_test(wins: int, losses: int) -> float:
    """Kiểm định dấu hai phía (binomial chính xác, p = 0,5), đã bỏ hòa."""
    total = wins + losses
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, i) for i in range(min(wins, losses) + 1)) / 2 ** total
    return min(1.0, 2 * tail)


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return 0.0, 1.0
    rate = successes / total
    denominator = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def read_judgments(key: dict[str, Any], paths: list[Path]) -> tuple[list[dict[str, Any]], int]:
    items = {item["n"]: item for item in key["items"]}
    choices = key["choices"]
    judgments, respondents = [], 0
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = [unicodedata.normalize("NFC", h) for h in reader.fieldnames or []]
            columns = {}
            for raw, header in zip(reader.fieldnames or [], headers):
                match = _ITEM_COLUMN.match(header)
                if match and int(match.group(1)) in items:
                    columns[raw] = int(match.group(1))
            background = next((raw for raw, header in zip(reader.fieldnames or [], headers)
                               if header.strip() == key["background"]["title"]), None)
            if not columns:
                raise SurveyError(f"{path}: không thấy cột 'Câu n.' nào. Đúng file CSV của form này chưa?")
            for row in reader:
                respondents += 1
                group = (row.get(background) or "").strip() if background else ""
                for column, number in columns.items():
                    value = unicodedata.normalize("NFC", (row.get(column) or "").strip())
                    if not value:
                        continue
                    first = items[number]["first"]
                    if value == choices[0]:
                        winner = first
                    elif value == choices[1]:
                        winner = "b" if first == "a" else "a"
                    elif value == choices[2]:
                        winner = "tie"
                    else:
                        raise SurveyError(f"{path}: câu {number} có lựa chọn lạ {value!r}.")
                    judgments.append({"n": number, "winner": winner, "group": group or "không rõ"})
    return judgments, respondents


def tally(judgments: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(j["winner"] for j in judgments)
    a, b, tie = counts["a"], counts["b"], counts["tie"]
    low, high = wilson(a, a + b)
    return {"a": a, "b": b, "tie": tie, "a_rate": a / (a + b) if a + b else None,
            "ci95": [low, high], "p_value": sign_test(a, b)}


def analyze(key: dict[str, Any], paths: list[Path]) -> dict[str, Any]:
    judgments, respondents = read_judgments(key, paths)
    if not judgments:
        raise SurveyError("Chưa có lượt đánh giá nào trong các file CSV.")
    per_item: dict[int, list] = defaultdict(list)
    per_group: dict[str, list] = defaultdict(list)
    for judgment in judgments:
        per_item[judgment["n"]].append(judgment)
        per_group[judgment["group"]].append(judgment)
    items = {item["n"]: item for item in key["items"]}
    item_rows, majority = [], Counter()
    for number in sorted(per_item):
        counts = tally(per_item[number])
        winner = "a" if counts["a"] > counts["b"] else "b" if counts["b"] > counts["a"] else "tie"
        majority[winner] += 1
        item_rows.append({"n": number, "question_id": items[number]["question_id"],
                          "question": items[number]["question"], **counts, "majority": winner})
    return {
        "variants": key["variants"], "respondents": respondents, "judgments": len(judgments),
        "overall": tally(judgments),
        # Các lượt của cùng một người/một câu không độc lập; đếm theo câu là phép thử thận trọng hơn.
        "by_question": {"a": majority["a"], "b": majority["b"], "tie": majority["tie"],
                        "p_value": sign_test(majority["a"], majority["b"])},
        "groups": {group: tally(rows) for group, rows in sorted(per_group.items())},
        "items": item_rows,
    }


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%".replace(".", ",")


def _p(value: float) -> str:
    return "< 0,001" if value < 0.001 else f"{value:.3f}".replace(".", ",")


def report(result: dict[str, Any]) -> str:
    a, b = result["variants"]["a"], result["variants"]["b"]
    overall, by_question = result["overall"], result["by_question"]
    low, high = overall["ci95"]
    lines = [
        f"# Kết quả khảo sát: A = `{a}`, B = `{b}`", "",
        f"- Người trả lời: {result['respondents']}; lượt đánh giá: {result['judgments']} "
        f"(A thắng {overall['a']}, B thắng {overall['b']}, hòa {overall['tie']}).",
        f"- Tỉ lệ A thắng khi bỏ hòa: **{_percent(overall['a_rate'])}** "
        f"(khoảng tin cậy 95%: {_percent(low)}–{_percent(high)}); kiểm định dấu p = {_p(overall['p_value'])}.",
        f"- Theo đa số từng câu: A thắng {by_question['a']} câu, B thắng {by_question['b']} câu, "
        f"hòa {by_question['tie']} câu; kiểm định dấu p = {_p(by_question['p_value'])}.",
        "", "Các lượt đánh giá của cùng một người hoặc cùng một câu không độc lập, nên p-value theo "
        "lượt lạc quan hơn thực tế. Khi hai cách đếm cho kết luận khác nhau, tin cách đếm theo câu.",
        "", "## Theo nhóm người trả lời", "",
        "| Nhóm | A thắng | B thắng | Hòa | Tỉ lệ A thắng (bỏ hòa) | p |", "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines += [f"| {group} | {c['a']} | {c['b']} | {c['tie']} | {_percent(c['a_rate'])} | {_p(c['p_value'])} |"
              for group, c in result["groups"].items()]
    lines += ["", "## Từng câu", "", "| Câu | id | Câu hỏi | A | B | Hòa | Đa số |",
              "| ---: | --- | --- | ---: | ---: | ---: | --- |"]
    names = {"a": "A", "b": "B", "tie": "hòa"}
    lines += [f"| {r['n']} | {r['question_id']} | {r['question']} | {r['a']} | {r['b']} | {r['tie']} | "
              f"{names[r['majority']]} |" for r in result["items"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build", help="Dựng form từ hai file runs/<variant>.jsonl.")
    build_parser.add_argument("--a", required=True, type=Path)
    build_parser.add_argument("--b", required=True, type=Path)
    build_parser.add_argument("--out", required=True, type=Path)
    build_parser.add_argument("--title", default="Khảo sát câu trả lời về bệnh cây trồng")
    build_parser.add_argument("--seed", type=int, default=2026)
    build_parser.add_argument("--per-form", type=int, default=12, help="Số cặp tối đa mỗi form.")
    build_parser.add_argument("--keep-citations", action="store_true", help="Giữ [Nguồn n] trong câu trả lời.")
    build_parser.add_argument("--include-truncated", action="store_true", help="Giữ cả cặp có câu bị cắt.")
    export_parser = commands.add_parser("export", help="Bảng so sánh có tên model (CSV + markdown).")
    export_parser.add_argument("--a", required=True, type=Path)
    export_parser.add_argument("--b", required=True, type=Path)
    export_parser.add_argument("--out", required=True, type=Path)
    analyze_parser = commands.add_parser("analyze", help="Đọc CSV phản hồi của Google Form.")
    analyze_parser.add_argument("--key", required=True, type=Path)
    analyze_parser.add_argument("--responses", required=True, nargs="+", type=Path)
    analyze_parser.add_argument("--out", type=Path, help="Ghi báo cáo markdown; bỏ trống thì in ra.")
    args = parser.parse_args(argv)

    try:
        if args.command == "build":
            key = build(args.a, args.b, args.out, title=args.title, seed=args.seed, per_form=args.per_form,
                        keep_citations=args.keep_citations, include_truncated=args.include_truncated)
            forms = max(item["form"] for item in key["items"])
            print(f"{len(key['items'])} cặp trong {forms} form → {args.out}/form.gs "
                  f"(A = {key['variants']['a']}, B = {key['variants']['b']}).")
            for item in key["excluded"]:
                print(f"  bỏ {item['question_id']}: {item['reason']}")
            print("Đọc preview.md trước, rồi chạy form.gs trên script.google.com. "
                  "Giữ key.json riêng, đừng gửi kèm form.")
            return 0
        if args.command == "export":
            csv_path, md_path = export(args.a, args.b, args.out)
            print(f"Đã ghi {csv_path} và {md_path}")
            return 0
        result = analyze(json.loads(args.key.read_text(encoding="utf-8")), args.responses)
    except SurveyError as exc:
        print(f"Dừng: {exc}", file=sys.stderr)
        return 1
    text = report(result)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"Đã ghi {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
