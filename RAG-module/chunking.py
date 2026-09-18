"""Chunk theo cấu trúc: parse heading -> unit có span -> chia unit quá dài.

Không suy heading từ chữ HOA hay tên bệnh. Offset là ký tự trong source đã
chuẩn hóa BOM/newline; header sinh thêm không thuộc span nguồn.
"""

from __future__ import annotations

import re
import unicodedata
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Sequence

from langchain_core.documents import Document

from config import ChunkingSettings

CHUNKING_VERSION = "structure-v1"
HEADER_VERSION = "compact-v1"
HEADING = re.compile(r"^(#{1,3})[ \t]+(.+?)\s*$")
# Đây là field của định dạng tài liệu, không phải luật match query/cây/bệnh.
METADATA_FIELDS = {
    "ten tai lieu": "document_label", "loai tai lieu": "document_type",
    "cay trong": "hosts", "ten benh": "disease_names",
    "tu khoa truy xuat": "keywords",
}


def canonical_text(text: str) -> str:
    return text.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")


def normalized(text: str) -> str:
    folded = "".join(c for c in unicodedata.normalize("NFKD", text.casefold())
                     if not unicodedata.combining(c)).replace("đ", "d")
    return " ".join(re.findall(r"[a-z0-9]+", folded))


def without_heading_markers(text: str) -> str:
    """View dùng cho metadata cũ; không thay đổi text hay offset của Document."""
    return re.sub(r"(?m)^#{1,3}[ \t]+", "", text)


@dataclass(frozen=True)
class ParsedUnit:
    heading_path: tuple[str, ...]
    start_index: int
    end_index: int
    start_line: int
    end_line: int
    section_id: str
    unit_id: str
    metadata_field: str = ""


@dataclass
class ParsedDocument:
    text: str
    title: str
    units: list[ParsedUnit]
    warnings: list[str]


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def parse_document(document: Document) -> ParsedDocument:
    """Một unit là phần body liên tục sau heading, trước heading kế tiếp.

    Phần dẫn của parent và phần dẫn trước H1 cũng được giữ. Heading lặp có ID
    riêng theo vị trí, nên hai mục trùng tên không bị gộp.
    """
    text = canonical_text(document.page_content)
    source = str(document.metadata.get("source", "<document>"))
    headings: list[tuple[int, int, int, str, int]] = []
    offset, fence, fence_length = 0, "", 0
    for line_number, line in enumerate(text.splitlines(keepends=True), 1):
        stripped = line.strip()
        marker = re.match(r"^(`{3,}|~{3,})", stripped)
        if marker:
            run = marker[1]
            if not fence:
                fence, fence_length = run[0], len(run)
            elif run[0] == fence and len(run) >= fence_length and not stripped[len(run):].strip():
                fence = ""
        elif not fence:
            match = HEADING.match(line.rstrip("\n"))
            if match:
                headings.append((offset, offset + len(line), len(match[1]), match[2], line_number))
            elif re.match(r"^#{1,6}(?:\s|$)", line):
                raise ValueError(f"{source}:{line_number}: heading cần nội dung và cấp #, ## hoặc ###.")
        offset += len(line)
    roots = [h for h in headings if h[2] == 1]
    if len(roots) != 1 or not headings or headings[0][2] != 1:
        raise ValueError(f"{source}: structure cần đúng một H1 (# Tiêu đề) trước các heading con.")
    title = roots[0][3]
    units: list[ParsedUnit] = []
    notices: list[str] = []
    path: list[str] = []
    section_id = f"{source}:0"

    def append_unit(start: int, end: int, heading_start: int, field: str = "") -> None:
        start, end = _trim_span(text, start, end)
        if start == end:
            return
        units.append(ParsedUnit(
            heading_path=tuple(path or [title]), start_index=start, end_index=end,
            start_line=text.count("\n", 0, start) + 1,
            end_line=text.count("\n", 0, end - 1) + 1,
            section_id=section_id, unit_id=f"{source}:{heading_start}", metadata_field=field,
        ))

    append_unit(0, headings[0][0], -1)
    for i, (start, body_start, level, heading, line_number) in enumerate(headings):
        if level > len(path) + 1:
            raise ValueError(f"{source}:{line_number}: heading nhảy cấp từ {len(path)} lên {level}.")
        path = path[:level - 1] + [heading]
        if level <= 2:
            section_id = f"{source}:{start}"
        end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
        field = METADATA_FIELDS.get(normalized(heading), "") if level == 2 else ""
        append_unit(body_start, end, start, field)
        # Parent chỉ nhóm các heading con là hợp lệ; leaf rỗng mới cần cảnh báo.
        if not text[body_start:end].strip() and (i + 1 == len(headings) or headings[i + 1][2] <= level):
            notices.append(f"{source}:{line_number}: mục '{heading}' không có body; không tạo chunk rỗng.")
    return ParsedDocument(text=text, title=title, units=units, warnings=notices)


@lru_cache(maxsize=4)
def load_tokenizer(model: str) -> Any:
    """Chỉ tải tokenizer, không tải embedding/LLM weights; dùng cache mỗi process."""
    from transformers import AutoTokenizer

    # Thử cache trước để preview offline không phải đợi request kiểm tra HF.
    try:
        return AutoTokenizer.from_pretrained(model, local_files_only=True)
    except (OSError, ValueError):
        try:
            return AutoTokenizer.from_pretrained(model)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"Không tải được tokenizer '{model}'. Kiểm tra cache/mạng hoặc CHUNK_TOKENIZER_MODEL."
            ) from exc


def token_counter(tokenizer: Any) -> Callable[[str], int]:
    return lambda text: len(tokenizer.encode(text, add_special_tokens=True, truncation=False))


def compact_header(metadata: dict[str, Any], path: tuple[str, ...]) -> str:
    values = [
        ("Tài liệu", metadata.get("document_title", path[0])),
        ("Tên tra cứu", metadata.get("title", "")),
        ("Mục", " > ".join(path[1:])),
    ]
    seen: set[str] = set()
    lines = []
    for label, value in values:
        key = normalized(str(value))
        if key and key not in seen:
            lines.append(f"{label}: {value}")
            seen.add(key)
    return "\n".join(lines)


def split_unit_spans(
    text: str, unit: ParsedUnit, header: str, count: Callable[[str], int],
    settings: ChunkingSettings,
) -> list[tuple[int, int]]:
    """Ưu tiên ngắt đoạn/câu/dòng; chỉ lùi overlap trong cùng unit.

    Binary search tìm một prefix vừa budget. Tokenizer không luôn đơn điệu:
    kết quả không nhất thiết dài nhất, nhưng mỗi chunk đều được đếm lại chính xác.
    """
    prefix = header + "\n\n"
    if count(prefix) >= settings.max_tokens:
        raise ValueError(f"{unit.unit_id}: header đã chiếm hết CHUNK_MAX_TOKENS.")
    spans = []
    start, limit = unit.start_index, unit.end_index
    while start < limit:
        if count(prefix + text[start:limit]) <= settings.max_tokens:
            end = limit
        else:
            low, high, end = start + 1, limit, start
            while low <= high:
                middle = (low + high) // 2
                if count(prefix + text[start:middle]) <= settings.max_tokens:
                    end, low = middle, middle + 1
                else:
                    high = middle - 1
            if end == start:
                raise ValueError(f"{unit.unit_id}: token budget không đủ cho header và một ký tự body.")
            # Tránh sinh chunk rất nhỏ chỉ vì có newline gần đầu đoạn.
            floor = start + (end - start) // 2
            for pattern in (r"\n\s*\n", r"[.!?;](?:\s+|$)", r"\n", r"\s+"):
                candidates = [start + m.end() for m in re.finditer(pattern, text[start:end])
                              if start + m.end() >= floor]
                fitting = [pos for pos in reversed(candidates)
                           if count(prefix + text[start:pos].rstrip()) <= settings.max_tokens]
                if fitting:
                    end = fitting[0]
                    break
        left, right = _trim_span(text, start, end)
        if left < right:
            if count(prefix + text[left:right]) > settings.max_tokens:
                # Bỏ whitespace có thể thay đổi tokenization tại boundary.
                raise ValueError(f"{unit.unit_id}: chunk vượt budget sau render; tăng budget hoặc đổi tokenizer.")
            spans.append((left, right))
        if end >= limit:
            break
        next_start = end
        if settings.overlap_tokens:
            low, high = start + 1, end
            while low <= high:
                middle = (low + high) // 2
                if count(text[middle:end]) <= settings.overlap_tokens:
                    next_start, high = middle, middle - 1
                else:
                    low = middle + 1
            # Overlap không được chiếm gần hết chunk, phải có tiến triển hữu ích.
            next_start = max(next_start, start + max(1, (end - start) // 2))
            while next_start < end and not text[next_start - 1].isspace():
                next_start += 1
        start = max(start + 1, next_start)
        while start < limit and text[start].isspace():
            start += 1
    return spans


def structure_records(
    documents: Sequence[Document], settings: ChunkingSettings, *,
    tokenizer: Any = None, header_builder: Callable | None = None,
) -> list[dict[str, Any]]:
    """Records cho preview; record kind=chunk mới được đưa vào vector index."""
    parsed_documents = [(document, parse_document(document)) for document in documents]
    count = token_counter(tokenizer if tokenizer is not None else load_tokenizer(settings.tokenizer_model))
    records = []
    for document, parsed in parsed_documents:
        metadata = {**document.metadata, "document_title": parsed.title}
        for unit in parsed.units:
            if unit.metadata_field:
                metadata[unit.metadata_field] = parsed.text[unit.start_index:unit.end_index]
        for notice in parsed.warnings:
            warnings.warn(notice, stacklevel=2)
            records.append({"kind": "warning", "source": metadata.get("source"), "message": notice})
        chunk_index = 0
        for unit in parsed.units:
            common = {
                **metadata, "heading_path": " > ".join(unit.heading_path),
                "section": unit.heading_path[1] if len(unit.heading_path) > 1 else "",
                "subsection": unit.heading_path[2] if len(unit.heading_path) > 2 else "",
                "section_id": unit.section_id, "unit_id": unit.unit_id,
                "chunking_strategy": "structure", "chunking_version": CHUNKING_VERSION,
            }
            if unit.metadata_field:
                records.append({
                    "kind": "metadata", "field": unit.metadata_field,
                    "body": parsed.text[unit.start_index:unit.end_index],
                    "metadata": {**common, "start_index": unit.start_index, "end_index": unit.end_index,
                                 "start_line": unit.start_line, "end_line": unit.end_line},
                })
                continue
            header = (header_builder(metadata) if header_builder else compact_header(metadata, unit.heading_path))
            for start, end in split_unit_spans(parsed.text, unit, header, count, settings):
                body = parsed.text[start:end]
                rendered = f"{header}\n\n{body}"
                records.append({
                    "kind": "chunk", "header": header, "body": body, "text": rendered,
                    "tokens": count(rendered), "header_tokens": count(header),
                    "metadata": {**common, "chunk_index": chunk_index, "start_index": start,
                                 "end_index": end, "start_line": parsed.text.count("\n", 0, start) + 1,
                                 "end_line": parsed.text.count("\n", 0, end - 1) + 1},
                })
                chunk_index += 1
    return records
