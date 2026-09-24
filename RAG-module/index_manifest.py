"""Mô tả snapshot đã index; kiểm tra trước khi query hoặc ghi đè khác strategy."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from chunking import HEADER_VERSION
from config import ChunkingSettings


def manifest_path(directory: Path, collection: str) -> Path:
    # Collection không được dùng trực tiếp làm filename/path.
    suffix = hashlib.sha256(collection.encode()).hexdigest()[:16]
    return directory / f"manifest-{suffix}.json"


def read_manifest(directory: Path, collection: str) -> dict | None:
    path = manifest_path(directory, collection)
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("collection") != collection or manifest.get("status") != "ready":
        raise RuntimeError(f"Index chưa hoàn tất hoặc manifest không hợp lệ: {path}. Hãy index lại.")
    return manifest


def _file_hashes(documents) -> list[dict]:
    return [{"source": doc.metadata["source"],
             "sha256": hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()}
            for doc in documents]


def corpus_fingerprint(documents) -> str:
    """Cùng công thức với `corpus_sha256` trong manifest: so để biết index có cũ không."""
    return hashlib.sha256(json.dumps(_file_hashes(documents), sort_keys=True).encode()).hexdigest()


def describe_manifest(documents, settings: ChunkingSettings, *, collection: str,
                      embedding_model: str, chunk_size: int, chunk_overlap: int) -> dict:
    files = _file_hashes(documents)
    return {
        "schema_version": 1, "status": "building", "collection": collection,
        "created_at": datetime.now(timezone.utc).isoformat(), "files": files,
        "corpus_sha256": corpus_fingerprint(documents),
        "chunking": asdict(settings), "embedding_model": embedding_model,
        "chunking_version": "structure-v1" if settings.strategy == "structure" else "recursive-v1",
        "legacy_chunk_size": chunk_size, "legacy_chunk_overlap": chunk_overlap,
        "header_version": HEADER_VERSION if settings.strategy == "structure" else "identity-v1",
    }


def write_manifest(directory: Path, collection: str, manifest: dict) -> None:
    target = manifest_path(directory, collection)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def validate_query_manifest(directory: Path, collection: str, embedding_model: str | None) -> None:
    manifest = read_manifest(directory, collection)
    if manifest is None:
        warnings.warn(f"Index legacy chưa có manifest: {directory}. Đang dùng snapshot đã lưu.", stacklevel=2)
        return
    stored_model = manifest["embedding_model"]
    if embedding_model is not None and stored_model != embedding_model:
        raise RuntimeError(
            f"Embedding không khớp index: '{stored_model}' khác '{embedding_model}'. "
            "Chọn đúng EMBEDDING_MODEL hoặc tạo index riêng."
        )
