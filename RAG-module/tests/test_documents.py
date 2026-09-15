from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document
from rag import format_context, load_documents, split_documents


def test_load_two_utf8_apple_documents_with_metadata(
    apple_data_dir: Path,
) -> None:
    documents = load_documents(apple_data_dir)

    assert len(documents) == 2
    by_source = {document.metadata["source"]: document for document in documents}
    assert set(by_source) == {
        "apple/apple_black_rot.txt",
        "apple/apple_scab.txt",
    }

    black_rot = by_source["apple/apple_black_rot.txt"]
    assert "BỆNH THỐI ĐEN TRÊN CÂY TÁO" in black_rot.page_content
    assert black_rot.metadata["source"] == "apple/apple_black_rot.txt"
    assert black_rot.metadata["crop"] == "apple"
    assert black_rot.metadata["title"] == "Apple Black Rot"
    assert black_rot.metadata["disease"].casefold().startswith("bệnh thối đen")
    assert "black rot" in black_rot.metadata["disease_aliases"].casefold()
    assert black_rot.metadata["disease_id"]

    apple_scab = by_source["apple/apple_scab.txt"]
    assert "Bệnh ghẻ táo" in apple_scab.page_content
    assert "Venturia inaequalis" in apple_scab.page_content
    assert apple_scab.metadata["source"] == "apple/apple_scab.txt"
    assert apple_scab.metadata["crop"] == "apple"
    assert apple_scab.metadata["title"] == "Apple Scab"
    assert apple_scab.metadata["disease"] == "Bệnh ghẻ táo"
    scab_aliases = apple_scab.metadata["disease_aliases"].casefold()
    assert "apple scab" in scab_aliases
    assert "bệnh sẹo táo" in scab_aliases
    assert apple_scab.metadata["disease_id"]

    for document in documents:
        aliases = document.metadata["disease_aliases"]
        assert isinstance(aliases, str)
        assert all(alias.strip() for alias in aliases.split(" | "))
    assert black_rot.metadata["disease_id"] != apple_scab.metadata["disease_id"]


def test_load_documents_rejects_missing_directory(tmp_path: Path) -> None:
    missing = tmp_path / "khong-ton-tai"

    with pytest.raises(RuntimeError, match="Không tìm thấy thư mục dữ liệu"):
        load_documents(missing)


def test_load_documents_reports_empty_corpus(tmp_path: Path) -> None:
    empty_file = tmp_path / "apple" / "empty.txt"
    empty_file.parent.mkdir()
    empty_file.write_text("  \n\n", encoding="utf-8")

    with pytest.warns(  # noqa: SIM117 - verify warning and raised error together.
        UserWarning, match="Bỏ qua file rỗng: apple/empty.txt"
    ):
        with pytest.raises(RuntimeError, match="Không có file \\.txt UTF-8"):
            load_documents(tmp_path)


def test_load_documents_reports_invalid_utf8_filename(tmp_path: Path) -> None:
    invalid_file = tmp_path / "apple" / "broken.txt"
    invalid_file.parent.mkdir()
    invalid_file.write_bytes(b"\xff\xfe\x00")

    with pytest.raises(RuntimeError, match=r"UTF-8.*apple/broken\.txt"):
        load_documents(tmp_path)


def test_split_documents_preserves_metadata_and_unicode() -> None:
    metadata = {
        "source": "apple/apple_scab.txt",
        "crop": "apple",
        "title": "Apple Scab",
        "disease": "Bệnh ghẻ táo",
        "disease_aliases": "Bệnh ghẻ táo | Apple scab | bệnh sẹo táo",
        "disease_id": "apple_scab",
    }
    document = Document(
        page_content=(
            "Triệu chứng ghẻ táo là đốm xanh ô liu trên lá non. "
            "Quả bệnh có thể nứt và biến dạng.\n\n"
        )
        * 8,
        metadata=metadata,
    )

    chunks = split_documents([document], chunk_size=120, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(chunk.page_content.strip() for chunk in chunks)
    assert any("Triệu chứng ghẻ táo" in chunk.page_content for chunk in chunks)
    for chunk in chunks:
        assert chunk.metadata["source"] == metadata["source"]
        assert chunk.metadata["crop"] == metadata["crop"]
        assert chunk.metadata["title"] == metadata["title"]
        assert chunk.metadata["disease"] == metadata["disease"]
        assert chunk.metadata["disease_aliases"] == metadata["disease_aliases"]
        assert chunk.metadata["disease_id"] == metadata["disease_id"]
        assert isinstance(chunk.metadata["start_index"], int)
        header = "\n".join(chunk.page_content.splitlines()[:3])
        assert "Tài liệu:" in header
        assert "Bệnh: Bệnh ghẻ táo" in header
        assert "Tên gọi:" in header


def test_every_real_apple_chunk_is_enriched_with_disease_identity(
    apple_data_dir: Path,
) -> None:
    documents = load_documents(apple_data_dir)
    chunks = split_documents(documents, chunk_size=300, chunk_overlap=30)

    assert len(chunks) > len(documents)
    for chunk in chunks:
        disease = chunk.metadata["disease"]
        aliases = chunk.metadata["disease_aliases"]
        first_lines = "\n".join(chunk.page_content.splitlines()[:3])
        assert "Tài liệu:" in first_lines
        assert f"Bệnh: {disease}" in first_lines
        assert f"Tên gọi: {aliases}" in first_lines


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap", "message"),
    [
        (0, 0, "chunk_size"),
        (100, -1, "chunk_overlap"),
        (100, 100, "chunk_overlap"),
    ],
)
def test_split_documents_validates_configuration(
    chunk_size: int,
    chunk_overlap: int,
    message: str,
) -> None:
    document = Document(page_content="Nội dung", metadata={})

    with pytest.raises(ValueError, match=message):
        split_documents(
            [document],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


def test_split_documents_rejects_empty_input() -> None:
    with pytest.raises(RuntimeError, match="Không tạo được chunk"):
        split_documents([])


def test_format_context_adds_ordered_source_labels() -> None:
    documents = [
        Document(
            page_content="  Nội dung ghẻ táo.  ",
            metadata={"source": "apple/apple_scab.txt"},
        ),
        Document(page_content="Nội dung bổ sung.", metadata={}),
    ]

    context = format_context(documents)

    assert context == (
        "[Nguồn 1: apple/apple_scab.txt]\nNội dung ghẻ táo.\n\n"
        "[Nguồn 2: không rõ nguồn]\nNội dung bổ sung."
    )
