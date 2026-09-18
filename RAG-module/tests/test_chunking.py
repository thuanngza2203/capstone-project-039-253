from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from langchain_core.documents import Document

import chunk_preview
import config
import main
import rag
from chunking import canonical_text, parse_document, structure_records
from config import ChunkingSettings, get_chunking_settings, get_index_directory
from index_manifest import manifest_path, read_manifest


class CharacterTokenizer:
    """Mỗi ký tự là một token, thêm 2 token đặc biệt: test budget độc lập weights."""

    def encode(self, text, **kwargs):
        return [0] * (len(text) + 2)


def records(text, maximum=250, overlap=20):
    return structure_records(
        [Document(page_content=text, metadata={"source": "test.txt", "title": "Lookup"})],
        ChunkingSettings("structure", maximum, overlap, "fake"), tokenizer=CharacterTokenizer(),
    )


def test_env_switches_strategy_and_default_index(monkeypatch):
    monkeypatch.setenv("CHUNKING_STRATEGY", "structure")
    assert get_chunking_settings().strategy == "structure"
    assert get_index_directory() == config.PROJECT_ROOT / "chroma_db_structure"
    monkeypatch.setenv("CHUNKING_STRATEGY", "recursive")
    assert get_index_directory() == config.CHROMA_DIR
    monkeypatch.setenv("CHROMA_DIR", "artifacts/custom")
    assert get_index_directory() == config.PROJECT_ROOT / "artifacts/custom"
    assert get_index_directory("explicit") == config.PROJECT_ROOT / "explicit"
    assert get_chunking_settings(strategy="structure").strategy == "structure"


@pytest.mark.parametrize(("name", "value"), [
    ("CHUNKING_STRATEGY", "semantic"), ("CHUNK_MAX_TOKENS", "0"),
    ("CHUNK_MAX_TOKENS", "bad"), ("CHUNK_OVERLAP_TOKENS", "-1"),
    ("CHUNK_OVERLAP_TOKENS", "400"), ("CHUNK_OVERLAP_TOKENS", "bad"),
])
def test_invalid_env(name, value, monkeypatch):
    monkeypatch.setenv("CHUNKING_STRATEGY", "structure")
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        get_chunking_settings()


def test_recursive_does_not_load_tokenizer_or_apply_token_settings(monkeypatch):
    monkeypatch.setenv("CHUNK_MAX_TOKENS", "invalid")
    monkeypatch.setattr(rag, "structure_records", lambda *a, **kw: pytest.fail("structure called"))
    chunks = rag.split_documents([Document(page_content="Văn bản cũ. " * 80)], chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    assert all(c.metadata["chunking_strategy"] == "recursive" for c in chunks)
    assert all(len(c.page_content.split("\n\n", 1)[1]) <= 100 for c in chunks)


def test_intro_parent_duplicate_headings_and_canonical_offsets():
    source = "\ufeffIntro trước title.\r\n# Title\r\nIntro.\r\n## Parent\r\nDẫn.\r\n### Leaf\r\nLặp lại.\r\n### Leaf\r\nLặp lại.\r\n"
    parsed = parse_document(Document(page_content=source))
    assert [parsed.text[u.start_index:u.end_index] for u in parsed.units] == [
        "Intro trước title.", "Intro.", "Dẫn.", "Lặp lại.", "Lặp lại.",
    ]
    assert parsed.units[-1].heading_path == ("Title", "Parent", "Leaf")
    assert parsed.units[-1].unit_id != parsed.units[-2].unit_id
    assert parsed.units[-1].start_line == 9
    assert parsed.units[-2].start_line == 7


@pytest.mark.parametrize("text", [
    "Không có H1", "# One\n# Two\nBody", "# Title\n### Skip\nBody",
    "## Before\n# Title\nBody", "# Title\n#### Unsupported\nBody", "# Title\n##\nBody",
])
def test_invalid_hierarchy_reports_source(text):
    with pytest.raises(ValueError, match="invalid.txt"):
        parse_document(Document(page_content=text, metadata={"source": "invalid.txt"}))


def test_fences_bullets_urls_and_formulas_are_body():
    body = "DM = S²IR\n- V1.\nhttps://example.org/#part\n```txt\n# code\n## code\n```\n~~~\n# code2\n~~~"
    parsed = parse_document(Document(page_content="# Title\n## ToMV\n" + body))
    assert len(parsed.units) == 1
    unit = parsed.units[0]
    assert unit.heading_path == ("Title", "ToMV")
    assert parsed.text[unit.start_index:unit.end_index] == body


def test_empty_parent_is_valid_empty_leaf_warns():
    result = records("# Title\n## Parent\n### Leaf\nBody\n## Empty\n")
    assert len([r for r in result if r["kind"] == "chunk"]) == 1
    assert len([r for r in result if r["kind"] == "warning"]) == 1


def test_oversized_repeated_body_has_exact_spans_and_no_cross_unit_overlap():
    text = "# Title\n## One\n" + "Lặp lại câu này. " * 40 + "\n## Two\nRiêng mục hai."
    chunks = [r for r in records(text, maximum=180, overlap=20) if r["kind"] == "chunk"]
    parsed = parse_document(Document(page_content=text, metadata={"source": "test.txt"}))
    covered = set()
    for r in chunks:
        meta = r["metadata"]
        start, end = meta["start_index"], meta["end_index"]
        assert r["body"] == text[start:end]
        assert r["tokens"] <= 180
        assert len(r["text"]) + 2 <= 180
        unit = next(u for u in parsed.units if u.unit_id == meta["unit_id"])
        assert unit.start_index <= start < end <= unit.end_index
        covered.update(range(start, end))
    for unit in parsed.units:
        assert all(i in covered for i in range(unit.start_index, unit.end_index) if not text[i].isspace())
    assert chunks[-1]["body"] == "Riêng mục hai."


def test_small_leaf_keeps_list_and_qualifier_together():
    body = "Các bước:\n- Bước 1.\n- Bước 2.\nChỉ áp dụng trong điều kiện nêu trên."
    chunks = records("# Title\n## Steps\n" + body)
    assert len(chunks) == 1
    assert chunks[0]["body"] == body


def test_header_budget_is_checked():
    with pytest.raises(ValueError, match="header"):
        records("# Tiêu đề cực dài\n## Mục cũng rất dài\nBody", maximum=20, overlap=0)


def test_metadata_fields_exported_but_not_indexed_as_evidence():
    text = "# Title\n## TỪ KHÓA TRUY XUẤT\nword1\nword2\n## Body\nBằng chứng."
    result = records(text)
    meta = next(r for r in result if r["kind"] == "metadata")
    chunk = next(r for r in result if r["kind"] == "chunk")
    assert meta["field"] == "keywords"
    assert chunk["metadata"]["keywords"] == "word1\nword2"
    assert "word1" not in chunk["text"]


def test_real_corpus_body_coverage_and_known_boundaries():
    documents = rag.load_documents()
    settings = ChunkingSettings("structure", 1800, 40, "fake")
    result = structure_records(documents, settings, tokenizer=CharacterTokenizer())
    for document in documents:
        parsed = parse_document(document)
        covered = set()
        for r in result:
            if r.get("metadata", {}).get("source") != document.metadata["source"]:
                continue
            meta = r["metadata"]
            start, end = meta["start_index"], meta["end_index"]
            assert r["body"] == parsed.text[start:end]
            covered.update(range(start, end))
        for unit in parsed.units:
            assert all(i in covered for i in range(unit.start_index, unit.end_index)
                       if not parsed.text[i].isspace()), document.metadata["source"]
    black_rot = [r for r in result if r.get("metadata", {}).get("source") == "apple/apple_black_rot.txt"]
    for r in black_rot:
        assert not re.search(r"(?m)^#{1,3} ", r["body"])
    assert any("Minnesota" in r["body"] and "Cắt tỉa" in r["metadata"]["subsection"] for r in black_rot)
    scab = [r for r in result if r.get("metadata", {}).get("source") == "apple/apple_scab.txt"]
    assert len([r for r in scab if r["kind"] == "metadata"]) == 5
    assert all("TỪ KHÓA TRUY XUẤT" not in r["metadata"]["heading_path"] for r in scab if r["kind"] == "chunk")
    tylcv = next(d for d in documents if "yello" in d.metadata["source"])
    parsed = parse_document(tylcv)
    v1 = next(u for u in parsed.units if u.heading_path[-1] == "V1")
    assert v1.heading_path[1] == "CẤU TRÚC BỘ GEN"
    pepper = next(d for d in documents if "pepper" in d.metadata["source"])
    assert "DM = S²IR" in pepper.page_content
    assert all("DM =" not in u.heading_path[-1] for u in parse_document(pepper).units)
    mites = next(d for d in documents if "spider" in d.metadata["source"])
    assert "NHỆN" in parse_document(mites).title


def test_preview_does_not_load_models_or_chroma(tmp_path, apple_data_dir, monkeypatch):
    fail = lambda *a, **kw: pytest.fail("Preview touched a model/store")
    monkeypatch.setattr(rag, "create_embeddings", fail)
    monkeypatch.setattr(rag, "create_chat_model", fail)
    monkeypatch.setattr(rag, "_new_vector_store", fail)
    monkeypatch.setattr(chunk_preview, "load_tokenizer", lambda model: CharacterTokenizer())
    monkeypatch.setenv("CHUNK_MAX_TOKENS", "1800")
    path = tmp_path / "preview.jsonl"
    assert main.run(["preview-chunks", "--strategy", "structure", "--data-dir", str(apple_data_dir),
                     "--source", "apple/apple_black_rot.txt", "--output", str(path)]) == 0
    assert path.exists() and path.with_suffix(".summary.json").exists()
    assert all(r["metadata"]["source"] == "apple/apple_black_rot.txt"
               for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()))


def test_both_indexes_persist_and_env_selects_correct_snapshot(tmp_path, apple_data_dir, fake_embeddings, monkeypatch):
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma_db")
    monkeypatch.setenv("CHUNK_MAX_TOKENS", "1800")
    rag.build_index(apple_data_dir, embeddings=fake_embeddings, strategy="recursive")
    rag.build_index(apple_data_dir, embeddings=fake_embeddings, strategy="structure", tokenizer=CharacterTokenizer())
    for strategy in ("recursive", "structure"):
        monkeypatch.setenv("CHUNKING_STRATEGY", strategy)
        result = rag.retrieve("Minnesota", mode="bm25")
        assert result and all(d.metadata["chunking_strategy"] == strategy for d in result)
        manifest = read_manifest(get_index_directory(), config.COLLECTION_NAME)
        assert manifest["chunking"]["strategy"] == strategy
    with pytest.raises(ValueError, match="strategy khác"):
        rag.build_index(apple_data_dir, tmp_path / "chroma_db", strategy="structure", embeddings=fake_embeddings)


def test_manifest_rejects_embedding_mismatch_before_loading_weights(tmp_path, apple_data_dir, fake_embeddings, monkeypatch):
    rag.build_index(apple_data_dir, tmp_path / "index", embeddings=fake_embeddings)
    monkeypatch.setattr(rag, "create_embeddings", lambda: pytest.fail("Weights loaded before validation"))
    with pytest.raises(RuntimeError, match="Embedding không khớp"):
        rag.retrieve("test", persist_directory=tmp_path / "index", mode="semantic")
    assert rag.retrieve("Venturia", persist_directory=tmp_path / "index", mode="bm25")


def test_partial_index_cannot_be_queried(tmp_path, apple_data_dir, fake_embeddings, monkeypatch):
    class BrokenStore:
        def reset_collection(self):
            pass
        def add_documents(self, chunks):
            raise RuntimeError("write failed")
    monkeypatch.setattr(rag, "_new_vector_store", lambda *a: BrokenStore())
    directory = tmp_path / "broken"
    with pytest.raises(RuntimeError, match="write failed"):
        rag.build_index(apple_data_dir, directory, embeddings=fake_embeddings)
    assert manifest_path(directory, config.COLLECTION_NAME).exists()
    with pytest.raises(RuntimeError, match="chưa hoàn tất"):
        rag.retrieve("test", mode="bm25", persist_directory=directory)


@pytest.mark.parametrize("command", ["search", "ask", "chat"])
def test_existing_cli_accepts_index_override(command):
    arguments = [command] + ([] if command == "chat" else ["test question"])
    args = main.create_parser().parse_args(arguments + ["--index-dir", "artifacts/chunking/test"])
    assert args.index_dir == "artifacts/chunking/test"
