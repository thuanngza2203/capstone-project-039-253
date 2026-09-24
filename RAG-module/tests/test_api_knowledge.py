"""API kho tri thức cho trang admin của web: chỉ đọc, không cần model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from index_manifest import manifest_path
from retrieval import chunk_key
from server.app import create_app
from server.knowledge import KnowledgeBase, split_header
from server.runtime import RAGRuntime
from test_api_server import AUTH, KEY, FilterStore

COLLECTION = "plant_disease_vi"
SCAB = "Bệnh ghẻ táo (Apple Scab)\n\n## TRIỆU CHỨNG\nĐốm xanh ô liu trên lá.\n\n## XỬ LÝ\nThu gom lá bệnh."
RUST = "BỆNH GỈ SẮT TRÊN CÂY NGÔ\n\nMụn gỉ sắt trên hai mặt lá."


def chunk(body: str, source: str, start: int, heading: str, index: int) -> Document:
    header = f"Tài liệu: {heading.split(' > ')[0]}\nMục: {heading.split(' > ')[-1]}"
    return Document(page_content=f"{header}\n\n{body}", metadata={
        "source": source, "crop": source.split("/")[0], "start_index": start, "chunk_index": index,
        "heading_path": heading, "section": heading.split(" > ")[-1], "start_line": index + 3, "end_line": index + 4,
    })


# Thứ tự trong store cố ý đảo: API phải sắp theo vị trí trong tài liệu.
CHUNKS = [
    chunk("Thu gom lá bệnh.", "apple/apple_scab.txt", 60, "Bệnh ghẻ táo > XỬ LÝ", 1),
    chunk("Đốm xanh ô liu trên lá.", "apple/apple_scab.txt", 20, "Bệnh ghẻ táo > TRIỆU CHỨNG", 0),
    chunk("Mụn gỉ sắt trên hai mặt lá.", "corn/corn_common_rust.txt", 30, "Gỉ sắt ngô > TỔNG QUAN", 0),
    chunk("Nội dung cũ.", "grape/grape_old.txt", 0, "Nho cũ > TỔNG QUAN", 0),
]


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    data = tmp_path / "data"
    for source, text in (("apple/apple_scab.txt", SCAB), ("corn/corn_common_rust.txt", RUST),
                         ("tomato/tomato_new.txt", "Tài liệu mới")):
        (data / source).parent.mkdir(parents=True, exist_ok=True)
        (data / source).write_text(text, encoding="utf-8")
    index_dir = tmp_path / "structure_db"
    index_dir.mkdir()
    # Manifest: ghẻ táo khớp file, gỉ sắt ngô đã sửa sau lần index, cà chua chưa index.
    import hashlib

    files = [{"source": "apple/apple_scab.txt", "sha256": hashlib.sha256(SCAB.encode()).hexdigest()},
             {"source": "corn/corn_common_rust.txt", "sha256": "cu"}]
    manifest_path(index_dir, COLLECTION).write_text(json.dumps({
        "collection": COLLECTION, "status": "ready", "chunking": {"strategy": "structure"}, "files": files,
        "corpus_sha256": "x", "chunk_count": 4, "created_at": "2026-09-24T00:00:00+00:00",
    }), encoding="utf-8")
    runtime = RAGRuntime(stores={"structure": FilterStore(CHUNKS)},
                         index_directories={"structure": index_dir, "recursive": tmp_path / "missing"})
    knowledge = KnowledgeBase(runtime, data_dir=data, count_tokens=lambda text: len(text.split()))
    return TestClient(create_app(runtime, api_key=KEY, knowledge=knowledge))


def test_split_header_keeps_body_only() -> None:
    assert split_header("Tài liệu: A\nMục: B\n\nThân") == ("Tài liệu: A\nMục: B", "Thân")
    assert split_header("Đoạn đầu\n\nĐoạn hai") == ("", "Đoạn đầu\n\nĐoạn hai")


def test_overview_lists_every_index_even_when_missing(client: TestClient) -> None:
    body = client.get("/v1/admin/overview", headers=AUTH).json()
    assert body["data"]["documents"] == 3
    indexes = {entry["name"]: entry for entry in body["indexes"]}
    structure = indexes["structure"]
    assert (structure["documents"], structure["chunks"]) == (3, 4)
    assert structure["chunks_per_crop"] == {"apple": 2, "corn": 1, "grape": 1}
    assert structure["tokens"]["max"] >= structure["tokens"]["mean"] > 0
    assert indexes["recursive"]["chunks"] is None and indexes["recursive"]["detail"]


def test_documents_report_chunks_index_state_and_taxonomy(client: TestClient) -> None:
    rows = {row["source"]: row for row in client.get("/v1/admin/documents?index=structure", headers=AUTH).json()}
    assert rows["apple/apple_scab.txt"]["title"] == "Bệnh ghẻ táo (Apple Scab)"
    assert rows["apple/apple_scab.txt"]["chunks"] == 2
    assert rows["apple/apple_scab.txt"]["index_state"] == "current"
    assert rows["apple/apple_scab.txt"]["taxonomy"] == [{"plant": "apple", "disease": "apple_scab"}]
    assert rows["corn/corn_common_rust.txt"]["index_state"] == "changed"
    assert rows["tomato/tomato_new.txt"]["index_state"] == "not_indexed"
    assert rows["tomato/tomato_new.txt"]["taxonomy"] == []
    assert rows["grape/grape_old.txt"]["index_state"] == "deleted"


def test_document_detail_orders_chunks_and_can_include_text(client: TestClient) -> None:
    body = client.get("/v1/admin/documents/apple/apple_scab.txt?index=structure&include_text=true",
                      headers=AUTH).json()
    assert [c["section"] for c in body["chunk_list"]] == ["TRIỆU CHỨNG", "XỬ LÝ"]
    assert body["chunk_list"][0]["preview"] == "Đốm xanh ô liu trên lá."
    assert body["text"] == SCAB
    assert client.get("/v1/admin/documents/apple/khong_co.txt?index=structure", headers=AUTH).status_code == 404


def test_chunk_search_ignores_accents_and_case(client: TestClient) -> None:
    found = client.get("/v1/admin/chunks", params={"q": "DOM XANH o liu", "index": "structure"}, headers=AUTH).json()
    assert [c["source"] for c in found] == ["apple/apple_scab.txt"]


def test_chunk_detail_has_neighbours(client: TestClient) -> None:
    first = chunk_key(CHUNKS[1])
    body = client.get(f"/v1/admin/chunks/{first}?index=structure", headers=AUTH).json()
    assert body["header"].startswith("Tài liệu: Bệnh ghẻ táo")
    assert body["body"] == "Đốm xanh ô liu trên lá."
    assert body["prev_id"] is None and body["next_id"] == chunk_key(CHUNKS[0])
    assert body["position"] == 0 and body["total_in_document"] == 2
    assert client.get("/v1/admin/chunks/khong-co?index=structure", headers=AUTH).status_code == 404


def test_missing_index_is_503(client: TestClient) -> None:
    assert client.get("/v1/admin/documents?index=recursive", headers=AUTH).status_code == 503


def test_cors_allows_any_origin_by_default(client: TestClient) -> None:
    response = client.get("/health", headers={"Origin": "http://192.168.1.20:5173"})
    assert response.headers["access-control-allow-origin"] == "*"
