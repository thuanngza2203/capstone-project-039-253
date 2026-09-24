"""Đọc kho tri thức cho trang admin của web: tài liệu trong data/ và chunk trong từng index.

Chỉ đọc. Chunk được đọc thẳng từ Chroma (không cần embedding) và giữ trong bộ nhớ theo
snapshot của collection: index build lại thì danh sách ID đổi, cache tự làm mới.
`chunk_id` là `retrieval.chunk_key`, giống ID trong debug của /v1/retrieve.
"""

from __future__ import annotations

import statistics
import threading
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from config import DATA_DIR, EMBEDDING_MODEL, INDEX_STRATEGIES
from index_manifest import _file_hashes, corpus_fingerprint, read_manifest
from rag import load_documents
from retrieval import chunk_key
from taxonomy import DISEASES

HEADER_LABELS = ("Tài liệu:", "Tên tra cứu:", "Mục:")
PREVIEW_LINES = 2


def fold(text: str) -> str:
    """So khớp không phân biệt hoa/thường và dấu, như BM25 của RAG."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in decomposed if not unicodedata.combining(c)).replace("đ", "d")


def split_header(content: str) -> tuple[str, str]:
    """Header định danh ("Tài liệu: …", "Mục: …") và thân chunk; chunk không có header thì header rỗng."""
    head, separator, body = content.partition("\n\n")
    lines = head.splitlines()
    if separator and lines and all(line.startswith(HEADER_LABELS) for line in lines):
        return head, body
    return "", content


def preview(body: str) -> str:
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    return "\n".join(lines[:PREVIEW_LINES])


@dataclass
class ChunkRecord:
    chunk_id: str
    source: str
    position: int  # thứ tự trong tài liệu (theo vị trí ký tự)
    content: str
    header: str
    body: str
    tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def crop(self) -> str:
        return str(self.metadata.get("crop") or self.source.split("/", 1)[0])

    def summary(self) -> dict[str, Any]:
        meta = self.metadata
        return {
            "chunk_id": self.chunk_id, "source": self.source, "position": self.position,
            "heading_path": meta.get("heading_path"), "section": meta.get("section"),
            "tokens": self.tokens, "chars": len(self.content),
            "start_line": meta.get("start_line"), "end_line": meta.get("end_line"),
            "preview": preview(self.body),
        }


@dataclass
class IndexSnapshot:
    ids: tuple[str, ...]
    chunks: list[ChunkRecord]
    by_id: dict[str, ChunkRecord]
    by_source: dict[str, list[ChunkRecord]]


class KnowledgeBase:
    def __init__(self, runtime, *, data_dir: Path | str = DATA_DIR,
                 count_tokens: Callable[[str], int] | None = None) -> None:
        """`count_tokens` để test inject; bỏ trống thì dùng tokenizer của embedding model."""
        self.runtime = runtime
        self.data_dir = Path(data_dir)
        self._count_tokens = count_tokens
        self._snapshots: dict[str, IndexSnapshot] = {}
        self._lock = threading.Lock()

    # --- Tài liệu trong data/ ---------------------------------------------------

    def documents(self) -> list[Any]:
        return load_documents(self.data_dir)

    @staticmethod
    def title_of(text: str, source: str) -> str:
        for line in text.splitlines():
            if line.strip():
                return line.strip().lstrip("#").strip() or source
        return source

    @staticmethod
    def taxonomy_of(source: str) -> list[dict[str, str]]:
        return [{"plant": entry.plant, "disease": entry.key} for entry in DISEASES if entry.source == source]

    # --- Chunk trong index ------------------------------------------------------

    def counter(self) -> Callable[[str], int]:
        if self._count_tokens is None:
            from chunking import load_tokenizer, token_counter

            try:
                self._count_tokens = token_counter(load_tokenizer(EMBEDDING_MODEL))
            except RuntimeError:
                # Không có tokenizer (offline, chưa cache): ước lượng theo số từ, vẫn xem được chunk.
                self._count_tokens = lambda text: len(text.split())
        return self._count_tokens

    def snapshot(self, index: str | None = None) -> IndexSnapshot:
        name = index or self.runtime.default_index
        store = self.runtime.load(name).store
        ids = tuple(store.get(include=[]).get("ids") or ())
        cached = self._snapshots.get(name)
        if cached is not None and cached.ids == ids:
            return cached
        with self._lock:
            cached = self._snapshots.get(name)
            if cached is not None and cached.ids == ids:
                return cached
            records = store.get(include=["documents", "metadatas"])
            count = self.counter()
            chunks = []
            for content, metadata in zip(records.get("documents") or [], records.get("metadatas") or []):
                if not content:
                    continue
                metadata = dict(metadata or {})
                header, body = split_header(content)
                chunks.append(ChunkRecord(
                    chunk_id=chunk_key(_as_document(content, metadata)),
                    source=str(metadata.get("source", "không rõ nguồn")), position=0,
                    content=content, header=header, body=body, tokens=count(content), metadata=metadata,
                ))
            chunks.sort(key=lambda c: (c.source, _int(c.metadata.get("start_index")),
                                       _int(c.metadata.get("chunk_index")), c.chunk_id))
            by_source: dict[str, list[ChunkRecord]] = {}
            for chunk in chunks:
                group = by_source.setdefault(chunk.source, [])
                chunk.position = len(group)
                group.append(chunk)
            snapshot = IndexSnapshot(ids, chunks, {c.chunk_id: c for c in chunks}, by_source)
            self._snapshots[name] = snapshot
            return snapshot

    # --- Dữ liệu cho từng API ---------------------------------------------------

    def overview(self) -> dict[str, Any]:
        documents = self.documents()
        fingerprint = corpus_fingerprint(documents)
        indexes = []
        for name in INDEX_STRATEGIES:
            status = self.runtime.index_status(name, fingerprint)
            entry = {**status, "documents": None, "chunks": None, "tokens": None, "chunks_per_crop": {}}
            try:
                snapshot = self.snapshot(name)
            except Exception as exc:  # noqa: BLE001 - index hỏng/chưa build vẫn phải hiện trên overview.
                entry["detail"] = entry["detail"] or getattr(exc, "detail", str(exc))
            else:
                tokens = [chunk.tokens for chunk in snapshot.chunks]
                entry.update(
                    documents=len(snapshot.by_source), chunks=len(snapshot.chunks),
                    tokens={"mean": round(statistics.fmean(tokens)), "median": round(statistics.median(tokens)),
                            "max": max(tokens)} if tokens else None,
                    chunks_per_crop=dict(Counter(chunk.crop for chunk in snapshot.chunks).most_common()),
                )
            indexes.append(entry)
        return {"data": {"documents": len(documents), "fingerprint": fingerprint[:12]},
                "default_index": self.runtime.default_index, "indexes": indexes}

    def document_rows(self, index: str | None = None) -> list[dict[str, Any]]:
        name = index or self.runtime.default_index
        snapshot = self.snapshot(name)
        indexed = _indexed_hashes(self.runtime.directory_for(name), self.runtime.collection_name)
        rows = [self._document_row(document, snapshot, indexed) for document in self.documents()]
        # Tài liệu đã xóa khỏi data/ nhưng vẫn còn chunk trong index.
        present = {row["source"] for row in rows}
        for source, chunks in snapshot.by_source.items():
            if source not in present:
                rows.append({"source": source, "crop": chunks[0].crop, "title": source, "chars": None,
                             "chunks": len(chunks), "tokens": sum(c.tokens for c in chunks),
                             "index_state": "deleted", "taxonomy": self.taxonomy_of(source)})
        return rows

    def _document_row(self, document, snapshot: IndexSnapshot, indexed: dict[str, str] | None) -> dict[str, Any]:
        source = document.metadata["source"]
        chunks = snapshot.by_source.get(source, [])
        current = {item["source"]: item["sha256"] for item in _file_hashes([document])}[source]
        if indexed is None:
            state = "unknown"  # index cũ không có manifest
        elif source not in indexed:
            state = "not_indexed"
        else:
            state = "current" if indexed[source] == current else "changed"
        return {"source": source, "crop": source.split("/", 1)[0],
                "title": self.title_of(document.page_content, source), "chars": len(document.page_content),
                "chunks": len(chunks), "tokens": sum(c.tokens for c in chunks),
                "index_state": state, "taxonomy": self.taxonomy_of(source)}

    def document_detail(self, source: str, index: str | None = None, include_text: bool = False) -> dict[str, Any] | None:
        name = index or self.runtime.default_index
        snapshot = self.snapshot(name)
        document = next((d for d in self.documents() if d.metadata["source"] == source), None)
        chunks = snapshot.by_source.get(source, [])
        if document is None and not chunks:
            return None
        indexed = _indexed_hashes(self.runtime.directory_for(name), self.runtime.collection_name)
        if document is not None:
            info = self._document_row(document, snapshot, indexed)
        else:
            info = {"source": source, "crop": chunks[0].crop, "title": source, "chars": None,
                    "chunks": len(chunks), "tokens": sum(c.tokens for c in chunks),
                    "index_state": "deleted", "taxonomy": self.taxonomy_of(source)}
        return {**info, "index": name, "chunk_list": [c.summary() for c in chunks],
                "text": document.page_content if include_text and document is not None else None}

    def search(self, query: str, index: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        snapshot = self.snapshot(index)
        needle = " ".join(fold(query).split())
        if not needle:
            return []
        found = [c for c in snapshot.chunks if needle in " ".join(fold(c.content).split())]
        return [c.summary() for c in found[:limit]]

    def chunk_detail(self, chunk_id: str, index: str | None = None) -> dict[str, Any] | None:
        name = index or self.runtime.default_index
        snapshot = self.snapshot(name)
        chunk = snapshot.by_id.get(chunk_id)
        if chunk is None:
            return None
        siblings = snapshot.by_source[chunk.source]
        return {
            **chunk.summary(), "index": name, "header": chunk.header, "body": chunk.body,
            "metadata": chunk.metadata, "total_in_document": len(siblings),
            "prev_id": siblings[chunk.position - 1].chunk_id if chunk.position > 0 else None,
            "next_id": siblings[chunk.position + 1].chunk_id if chunk.position + 1 < len(siblings) else None,
        }


def _as_document(content: str, metadata: dict[str, Any]):
    from langchain_core.documents import Document

    return Document(page_content=content, metadata=metadata)


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _indexed_hashes(directory: Path, collection: str) -> dict[str, str] | None:
    try:
        manifest = read_manifest(directory, collection)
    except RuntimeError:
        return None
    if manifest is None:
        return None
    return {item["source"]: item["sha256"] for item in manifest.get("files", [])}
