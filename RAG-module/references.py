"""Tiêu đề và link nguồn tham khảo của tài liệu trong data/, để người dùng biết câu trả lời lấy từ đâu.

Mọi URL trong kho là link tham khảo: nằm riêng một dòng, dòng ngay phía trên là tên nguồn.
Phần lớn nằm trong mục "NGUỒN THAM KHẢO", vài tài liệu ghi ở đầu file, nên đọc cả file
thay vì chỉ một heading.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from chunking import canonical_text, normalized
from config import DATA_DIR

_URL_LINE = re.compile(r"^https?://\S+$")


def document_title(text: str, fallback: str) -> str:
    """Dòng đầu tiên có chữ (H1 của tài liệu), bỏ dấu #."""
    for line in text.splitlines():
        if line.strip():
            return line.strip().lstrip("#").strip() or fallback
    return fallback


def _label(line: str) -> str | None:
    # "Nguồn tham khảo chính:" chỉ là nhãn mục, không phải tên nguồn.
    if not line or line.startswith("#") or _URL_LINE.match(line) or normalized(line).startswith("nguon tham khao"):
        return None
    return line.rstrip(":").strip() or None


def reference_links(text: str) -> list[dict[str, str | None]]:
    """Các link theo thứ tự xuất hiện, không trùng; `label` là dòng có chữ ngay phía trên."""
    links: list[dict[str, str | None]] = []
    seen: set[str] = set()
    previous = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _URL_LINE.match(line) and line not in seen:
            seen.add(line)
            links.append({"label": _label(previous), "url": line})
        previous = line
    return links


def describe_sources(sources: Iterable[str], data_dir: Path | str = DATA_DIR) -> list[dict[str, Any]]:
    """Tiêu đề + link của từng tài liệu; thiếu file (server không kèm data/) thì dùng tên file."""
    root = Path(data_dir).resolve()
    described = []
    for source in sources:
        path = (root / source).resolve()
        text = ""
        if path.is_relative_to(root) and path.is_file():
            try:
                text = canonical_text(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError):
                text = ""
        described.append({"source": source, "title": document_title(text, source),
                          "links": reference_links(text)})
    return described
