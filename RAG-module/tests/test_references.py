"""Tiêu đề và link nguồn tham khảo lấy từ tài liệu trong data/."""

from __future__ import annotations

from pathlib import Path

from config import DATA_DIR
from references import describe_sources, document_title, reference_links

TEXT = """# BỆNH THỐI ĐEN TRÊN CÂY NHO

Nguồn tham khảo chính:
https://example.org/top

## TRIỆU CHỨNG
Lá có đốm nâu.

## NGUỒN THAM KHẢO
Penn State Extension — Grape Disease: Black Rot
https://extension.psu.edu/grape-disease-black-rot

Cornell CALS
https://cals.cornell.edu/black-rot
https://extension.psu.edu/grape-disease-black-rot
"""


def test_links_keep_order_label_and_skip_duplicates() -> None:
    assert reference_links(TEXT) == [
        # Dòng phía trên chỉ là nhãn mục, không phải tên nguồn.
        {"label": None, "url": "https://example.org/top"},
        {"label": "Penn State Extension — Grape Disease: Black Rot",
         "url": "https://extension.psu.edu/grape-disease-black-rot"},
        {"label": "Cornell CALS", "url": "https://cals.cornell.edu/black-rot"},
    ]


def test_title_is_the_first_heading() -> None:
    assert document_title(TEXT, "x.txt") == "BỆNH THỐI ĐEN TRÊN CÂY NHO"
    assert document_title("\n\n", "x.txt") == "x.txt"


def test_missing_file_and_path_outside_data_fall_back_to_name(tmp_path: Path) -> None:
    (tmp_path.parent / "secret.txt").write_text("# Bí mật\nhttps://leak.example\n", encoding="utf-8")
    assert describe_sources(["nope.txt", "../secret.txt"], tmp_path) == [
        {"source": "nope.txt", "title": "nope.txt", "links": []},
        {"source": "../secret.txt", "title": "../secret.txt", "links": []},
    ]


def test_every_link_in_the_corpus_has_a_source_name() -> None:
    """Người dùng thấy tên nguồn thay cho URL trần: tài liệu mới phải ghi tên ngay trên link."""
    sources = sorted(path.relative_to(DATA_DIR).as_posix() for path in DATA_DIR.rglob("*.txt"))
    unnamed = [(item["source"], link["url"]) for item in describe_sources(sources)
               for link in item["links"] if not link["label"]]
    assert unnamed == []
