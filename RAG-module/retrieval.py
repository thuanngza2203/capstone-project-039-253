"""Các bước tìm kiếm độc lập: semantic, BM25, RRF và reranker.

Đọc từ dưới lên ở search_store() để thấy luồng tổng thể. Các hàm phía trên
thực hiện từng bước; không có luật suy tên cây/bệnh từ câu hỏi.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any, Callable
from weakref import WeakKeyDictionary

from langchain_core.documents import Document

from config import RetrievalSettings


@dataclass
class SearchHit:
    document: Document
    chunk_id: str
    semantic_rank: int | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


@dataclass(frozen=True)
class MetadataScope:
    """Giới hạn tìm kiếm theo metadata đã lưu trong index; rỗng = toàn corpus.

    Chỉ nhận giá trị có cấu trúc (đường dẫn tài liệu, thư mục cây), không suy từ
    câu hỏi. Bảng ánh xạ nhãn → phạm vi nằm ở taxonomy.py.
    """

    sources: tuple[str, ...] = ()
    crop: str | None = None

    @property
    def active(self) -> bool:
        return bool(self.sources or self.crop)

    def matches(self, document: Document) -> bool:
        metadata = document.metadata
        if self.sources and metadata.get("source") not in self.sources:
            return False
        return not self.crop or metadata.get("crop") == self.crop

    def chroma_filter(self) -> dict | None:
        conditions = []
        if self.sources:
            conditions.append({"source": self.sources[0]} if len(self.sources) == 1
                              else {"source": {"$in": list(self.sources)}})
        if self.crop:
            conditions.append({"crop": self.crop})
        if not conditions:
            return None
        return conditions[0] if len(conditions) == 1 else {"$and": conditions}


@dataclass
class SearchResult:
    mode: str
    hits: list[SearchHit]
    candidates: list[SearchHit]
    semantic_count: int
    bm25_count: int
    merged_count: int
    reranked: bool
    scope: MetadataScope | None = None

    @property
    def documents(self) -> list[Document]:
        return [hit.document for hit in self.hits]

    def to_debug_dict(self) -> dict[str, Any]:
        """Không thêm debug vào metadata lưu trong Chroma hoặc prompt LLM."""
        selected = {hit.chunk_id for hit in self.hits}
        return {
            "mode": self.mode,
            "semantic_candidates": self.semantic_count,
            "bm25_candidates": self.bm25_count,
            "merged_candidates": self.merged_count,
            "reranked": self.reranked,
            "scope": None if self.scope is None else {
                "sources": list(self.scope.sources), "crop": self.scope.crop,
            },
            "candidates": [
                {
                    "rank": rank,
                    "selected": hit.chunk_id in selected,
                    "chunk_id": hit.chunk_id,
                    "source": hit.document.metadata.get("source"),
                    "start_index": hit.document.metadata.get("start_index"),
                    "end_index": hit.document.metadata.get("end_index"),
                    "heading_path": hit.document.metadata.get("heading_path"),
                    "start_line": hit.document.metadata.get("start_line"),
                    "end_line": hit.document.metadata.get("end_line"),
                    "chunking_strategy": hit.document.metadata.get("chunking_strategy", "legacy"),
                    "semantic_rank": hit.semantic_rank,
                    "bm25_rank": hit.bm25_rank,
                    "bm25_score": hit.bm25_score,
                    "rrf_score": hit.rrf_score,
                    "rerank_score": hit.rerank_score,
                }
                for rank, hit in enumerate(self.candidates, start=1)
            ],
        }


def chunk_key(document: Document) -> str:
    """Cùng source/vị trí/nội dung có cùng ID ở cả hai nhánh và index cũ.

    Không chỉ dùng source: nhiều đoạn khác nhau trong cùng file đều cần giữ.
    Không phụ thuộc UUID do Chroma tạo khi index.
    """
    identity = "\0".join([
        str(document.metadata.get("source", "")),
        str(document.metadata.get("start_index", "")),
        document.page_content,
    ])
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def unique_documents(documents: list[Document]) -> list[Document]:
    by_id: dict[str, Document] = {}
    for document in documents:
        if document.page_content.strip():
            by_id.setdefault(chunk_key(document), document)
    return list(by_id.values())


def read_chunks(store: Any) -> list[Document]:
    """Đọc snapshot text/metadata đang được index; không tải embedding.

    Đọc cả collection nên corpus lớn hơn sẽ cần pagination/index BM25 riêng.
    Kết quả được dùng lại qua ``corpus_index()``; cache ở đó tự hết hiệu lực khi
    collection đổi, nên rebuild index không để lại BM25 cũ.
    """
    if not callable(getattr(store, "get", None)):
        raise ValueError(
            "BM25/hybrid cần vector_store.get(include=['documents', 'metadatas']). "
            "Store chỉ có similarity_search có thể dùng mode='semantic'."
        )
    records = store.get(include=["documents", "metadatas"])
    texts = records.get("documents")
    metadatas = records.get("metadatas")
    if texts is None or metadatas is None or len(texts) != len(metadatas):
        raise RuntimeError("Vector store trả về documents/metadatas không hợp lệ.")
    documents = [
        Document(page_content=text, metadata=metadata or {})
        for text, metadata in zip(texts, metadatas)
        if text and text.strip()
    ]
    # Thứ tự ổn định để tie-break BM25 không phụ thuộc thứ tự Chroma trả về.
    return sorted(unique_documents(documents), key=chunk_key)


def tokenize(text: str) -> list[str]:
    """Cùng một tokenizer cho query và chunk, hỗ trợ gõ tiếng Việt không dấu.

    Dùng từ đơn và cặp từ liền nhau (bigram), ví dụ 'khoai_tay'. Đây là cặp
    token tự sinh, không phải danh sách cây/bệnh hay bộ tách từ tiếng Việt.
    """
    folded = "".join(
        char for char in unicodedata.normalize("NFKD", text.casefold())
        if not unicodedata.combining(char)
    ).replace("đ", "d")
    words = re.findall(r"[a-z0-9]+", folded)
    bigrams = [f"{left}_{right}" for left, right in zip(words, words[1:])]
    return words + bigrams


@dataclass
class CorpusIndex:
    """Corpus đã tokenize kèm BM25, dựng một lần cho mỗi snapshot của index.

    Tách khỏi query để phần đắt (đọc Chroma, tokenize cả corpus, dựng BM25)
    không phải làm lại ở mỗi câu hỏi trong cùng một phiên.
    """

    documents: list[Document] = field(default_factory=list)
    tokens: list[list[str]] = field(default_factory=list)
    bm25: Any = None


def build_corpus_index(documents: list[Document]) -> CorpusIndex:
    """Tokenize và dựng BM25; chunk không có token nào bị loại khỏi corpus."""
    from rank_bm25 import BM25Plus

    corpus = [(doc, tokenize(doc.page_content)) for doc in unique_documents(documents)]
    corpus = [(doc, tokens) for doc, tokens in corpus if tokens]
    if not corpus:
        return CorpusIndex()
    # BM25Plus có IDF dương, tránh trường hợp Okapi cho IDF=0 khi từ xuất hiện
    # trong đúng một nửa corpus (rất dễ gặp ở corpus/test chỉ có vài tài liệu).
    return CorpusIndex(
        documents=[doc for doc, _ in corpus],
        tokens=[tokens for _, tokens in corpus],
        bm25=BM25Plus([tokens for _, tokens in corpus]),
    )


_CORPUS_CACHE: WeakKeyDictionary = WeakKeyDictionary()


def _collection_fingerprint(store: Any) -> tuple[str, ...] | None:
    """ID của collection: đủ để thấy index bị build lại, rẻ hơn đọc nội dung.

    Trả None khi store không cung cấp ``ids`` (ví dụ store giả trong test); khi
    đó bỏ qua cache thay vì đoán rằng corpus không đổi.
    """
    try:
        records = store.get(include=[])
    except TypeError:
        return None
    ids = records.get("ids") if isinstance(records, dict) else None
    return None if ids is None else tuple(ids)


def corpus_index(store: Any) -> CorpusIndex:
    """Dùng lại corpus/BM25 khi snapshot của index không đổi trong process này."""
    fingerprint = _collection_fingerprint(store)
    if fingerprint is not None:
        cached = _CORPUS_CACHE.get(store)
        if cached is not None and cached[0] == fingerprint:
            return cached[1]

    index = build_corpus_index(read_chunks(store))
    if fingerprint is not None:
        try:
            _CORPUS_CACHE[store] = (fingerprint, index)
        except TypeError:
            # Store không hỗ trợ weakref: vẫn trả kết quả đúng, chỉ không cache.
            pass
    return index


def bm25_search_index(
    question: str, index: CorpusIndex, limit: int,
    keep: Callable[[Document], bool] | None = None,
) -> list[SearchHit]:
    """Chấm điểm một query trên corpus đã dựng sẵn.

    `keep` lọc theo phạm vi **trước** khi cắt top, để phạm vi hẹp không trả rỗng
    chỉ vì top của toàn corpus nằm ngoài phạm vi. IDF vẫn tính trên toàn corpus.
    """
    query_tokens = tokenize(question)
    if not query_tokens or index.bm25 is None:
        return []

    scores = index.bm25.get_scores(query_tokens)
    query_terms = set(query_tokens)
    hits = [
        SearchHit(document=doc, chunk_id=chunk_key(doc), bm25_score=float(score))
        for doc, tokens, score in zip(index.documents, index.tokens, scores)
        if query_terms.intersection(tokens) and (keep is None or keep(doc))
    ]
    # Không lấy chunk không có từ nào khớp chỉ để lấp đầy top-k.
    # Điểm BM25 là điểm xếp hạng, không phải xác suất liên quan.
    hits.sort(key=lambda hit: (-hit.bm25_score, hit.chunk_id))
    for rank, hit in enumerate(hits, start=1):
        hit.bm25_rank = rank
    return hits[:limit]


def bm25_search(question: str, documents: list[Document], limit: int) -> list[SearchHit]:
    """BM25 trên danh sách chunk truyền thẳng vào; dựng corpus mới mỗi lần gọi."""
    return bm25_search_index(question, build_corpus_index(documents), limit)


def semantic_search(
    question: str, store: Any, limit: int, scope: MetadataScope | None = None,
) -> list[SearchHit]:
    """Gửi nguyên query vào embedding search; filter chỉ đến từ phạm vi có cấu trúc."""
    if scope is not None and scope.active:
        found = store.similarity_search(question, k=limit, filter=scope.chroma_filter())
    else:
        found = store.similarity_search(question, k=limit)
    documents = unique_documents(list(found))
    return [
        SearchHit(document=doc, chunk_id=chunk_key(doc), semantic_rank=rank)
        for rank, doc in enumerate(documents[:limit], start=1)
    ]


def reciprocal_rank_fusion(
    semantic_hits: list[SearchHit], bm25_hits: list[SearchHit], rrf_k: int
) -> list[SearchHit]:
    """Mỗi nhánh góp 1 / (rrf_k + rank); không cộng hai loại raw score."""
    merged: dict[str, SearchHit] = {}
    for branch in (semantic_hits, bm25_hits):
        seen: set[str] = set()
        for hit in branch:
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            combined = merged.setdefault(
                hit.chunk_id,
                SearchHit(document=hit.document, chunk_id=hit.chunk_id, rrf_score=0.0),
            )
            # Dùng thứ hạng sau khi loại trùng trong chính nhánh này.
            combined.rrf_score += 1.0 / (rrf_k + len(seen))
            if hit.semantic_rank is not None:
                combined.semantic_rank = hit.semantic_rank
            if hit.bm25_rank is not None:
                combined.bm25_rank = hit.bm25_rank
                combined.bm25_score = hit.bm25_score
    return sorted(merged.values(), key=lambda hit: (-hit.rrf_score, hit.chunk_id))


@lru_cache(maxsize=1)
def load_reranker(model_name: str, device: str):
    """Chỉ tải model khi bật rerank, dùng lại ở các query trong cùng process."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device=device, max_length=512)


def rerank_hits(
    question: str, hits: list[SearchHit], settings: RetrievalSettings
) -> list[SearchHit]:
    if not hits:
        return []
    try:
        model = load_reranker(settings.reranker_model, settings.reranker_device)
        pairs = [(question, hit.document.page_content) for hit in hits]
        scores = list(model.predict(pairs, batch_size=8, show_progress_bar=False))
        if len(scores) != len(hits):
            raise ValueError("Số điểm reranker không khớp số chunk.")
        ranked = [replace(hit, rerank_score=float(score)) for hit, score in zip(hits, scores)]
        if not all(math.isfinite(hit.rerank_score) for hit in ranked):
            raise ValueError("Reranker trả về điểm NaN/infinity.")
    except Exception as exc:
        raise RuntimeError(
            "Reranker thất bại. Kiểm tra RERANKER_MODEL/DEVICE hoặc đặt "
            "RERANKER_ENABLED=false để chạy retrieval không rerank."
        ) from exc
    # Stable sort: điểm bằng nhau giữ thứ tự trước rerank.
    return sorted(ranked, key=lambda hit: hit.rerank_score, reverse=True)


def search_store(
    question: str, store: Any, *, k: int, settings: RetrievalSettings,
    scope: MetadataScope | None = None,
) -> SearchResult:
    """Luồng chính: lấy ứng viên → gộp → rerank tùy chọn → chọn top-k."""
    limit = max(k, settings.candidate_k)
    semantic_hits = []
    bm25_hits = []
    active = scope if scope is not None and scope.active else None

    if settings.mode in {"semantic", "hybrid"}:
        semantic_hits = semantic_search(question, store, limit, active)
    if settings.mode in {"bm25", "hybrid"}:
        bm25_hits = bm25_search_index(
            question, corpus_index(store), limit, active.matches if active else None,
        )

    if settings.mode == "hybrid":
        candidates = reciprocal_rank_fusion(semantic_hits, bm25_hits, settings.rrf_k)
    elif settings.mode == "bm25":
        candidates = bm25_hits
    else:
        candidates = semantic_hits

    merged_count = len(candidates)
    if settings.reranker_enabled:
        # Chỉ rerank shortlist có giới hạn, không đưa cả corpus qua CrossEncoder.
        candidates = rerank_hits(question, candidates[:limit], settings)

    return SearchResult(
        mode=settings.mode,
        hits=candidates[:k],
        candidates=candidates,
        semantic_count=len(semantic_hits),
        bm25_count=len(bm25_hits),
        merged_count=merged_count,
        reranked=settings.reranker_enabled and bool(candidates),
        scope=active,
    )
