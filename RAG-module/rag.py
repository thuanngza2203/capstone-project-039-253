"""Pipeline RAG tối giản: load -> split -> embed -> retrieve -> generate."""

from __future__ import annotations

import re
import unicodedata
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL,
    TOP_K,
    create_chat_model,
    create_embeddings,
    get_chat_settings,
    get_chunking_settings,
    get_index_directory,
    get_retrieval_settings,
)
from chunking import canonical_text, structure_records, without_heading_markers
from index_manifest import (
    describe_manifest, manifest_path, validate_query_manifest, write_manifest,
)
from conversation import ChatTurn, ConversationMemory, rewrite_question
from retrieval import MetadataScope, SearchResult, load_reranker, search_store
from langchain_chroma import Chroma
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter

SYSTEM_PROMPT = """Bạn là trợ lý tra cứu kiến thức bệnh cây.

Quy tắc bắt buộc:
- Chỉ dùng thông tin trong phần NGỮ CẢNH được cung cấp.
- Nếu ngữ cảnh không đủ để trả lời, hãy nói rõ rằng kho tài liệu hiện tại chưa có đủ thông tin.
- Không tự tạo nguồn, tên thuốc, hoạt chất, liều lượng hoặc lịch phun.
- Khi dùng một thông tin, hãy dẫn nhãn [Nguồn n] tương ứng.
- Với thuốc bảo vệ thực vật, luôn nhắc người dùng tuân thủ nhãn sản phẩm và quy định địa phương.
- Trả lời cùng ngôn ngữ với câu hỏi; nếu không xác định được thì trả lời bằng tiếng Việt.
- Ưu tiên câu trả lời ngắn gọn, thực tế và dễ hiểu.
- Lịch sử chỉ giúp hiểu ý định và đối tượng của câu hỏi, không phải bằng chứng
  về bệnh cây. Không dùng câu trả lời cũ thay cho NGỮ CẢNH được truy xuất lượt này.
- Nhãn [Nguồn n] chỉ có hiệu lực trong NGỮ CẢNH lượt hiện tại, không tái sử dụng
  số nguồn của câu trả lời cũ. Nếu đối tượng còn mơ hồ, hỏi lại người dùng.
- CÂU HỎI ĐÃ LÀM RÕ giúp xác định đối tượng người dùng nhắc lại. Chỉ dùng chunk
  liên quan đối tượng đó; tài liệu về bệnh khác trong NGỮ CẢNH không có nghĩa
  người dùng đang hỏi thêm bệnh đó. Query đã làm rõ cũng không phải bằng chứng.
"""

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("history", optional=True),
        (
            "human",
            (
                "{subject_block}{examples_block}NGỮ CẢNH:\n{context}\n\nCÂU HỎI:\n{question}\n\n"
                "CÂU HỎI ĐÃ LÀM RÕ:\n{retrieval_query}\n\n"
                "Hãy trả lời theo các quy tắc trên."
            ),
        ),
    ]
)

# Đối tượng đi trong lượt human chứ không phải SystemMessage thứ hai: nhiều chat
# template (Gemma, một số bản Qwen) chỉ nhận system message ở đầu hội thoại.
SUBJECT_HEADER = (
    "ĐỐI TƯỢNG ĐANG HỎI (hệ thống cung cấp để biết người dùng hỏi về cây/bệnh nào; "
    "không phải bằng chứng, không trích dẫn):"
)
NO_CONTEXT_ANSWER = (
    "Kho tài liệu hiện tại chưa có đủ thông tin liên quan để trả lời câu hỏi này."
)
# Câu trả lời admin đã duyệt/sửa (Feedback RAG của detection). Quy tắc nằm ngay trong khối
# này, không thêm vào SYSTEM_PROMPT: không có ví dụ thì prompt giữ nguyên như trước.
EXAMPLES_HEADER = (
    "CÂU TRẢ LỜI MẪU ĐÃ ĐƯỢC QUẢN TRỊ VIÊN DUYỆT (cho các câu hỏi tương tự trước đây). "
    "Nếu câu hỏi hiện tại cùng ý với một mẫu, ưu tiên nội dung của mẫu đó: quản trị viên đã "
    "kiểm tra hoặc sửa lại, được dùng cả khi NGỮ CẢNH không có và thay NGỮ CẢNH khi hai bên "
    "mâu thuẫn. Không gắn nhãn [Nguồn n] cho phần lấy từ mẫu. Mẫu không cùng ý thì bỏ qua:"
)


def examples_block(examples: Sequence[dict[str, str]] | None) -> str:
    if not examples:
        return ""
    items = [
        f"Mẫu {number}\nHỏi: {example['question'].strip()}\nĐáp: {example['answer'].strip()}"
        for number, example in enumerate(examples, start=1)
    ]
    return f"{EXAMPLES_HEADER}\n\n" + "\n\n".join(items) + "\n\n"

_ALIAS_SEPARATOR = " | "
_DISEASE_LABELS = {
    "ten benh",
    "ten tieng anh",
    "ten tieng viet",
    "english name",
    "vietnamese name",
}
_SECTION_HEADINGS = {
    "bo phan bi gay hai",
    "cay trong",
    "chu ky benh",
    "dieu kien phat sinh va lay lan",
    "loai tai lieu",
    "nguon tham khao",
    "phan biet voi benh khac",
    "quan ly va phong ngua",
    "tac nhan gay benh",
    "ten tai lieu",
    "trieu chung",
    "tu khoa",
}
_DISEASE_MATCH_STOPWORDS = {
    "benh",
    "cay",
    "disease",
    "of",
    "on",
    "plant",
    "the",
    "tree",
    "tren",
}


def _humanize_filename(path: Path) -> str:
    words = re.sub(r"[-_]+", " ", path.stem).strip()
    return words.title() or path.stem


def _normalize_for_match(value: str) -> str:
    """Chuẩn hóa chữ hoa, dấu và dấu câu để so khớp tên bệnh."""

    without_accents = "".join(
        character
        for character in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(character)
    ).replace("đ", "d")
    return " ".join(re.findall(r"[a-z0-9]+", without_accents))


def _meaningful_disease_tokens(value: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in _normalize_for_match(value).split()
        if token not in _DISEASE_MATCH_STOPWORDS
    )


def _next_non_empty_line(lines: Sequence[str], start: int) -> str | None:
    for line in lines[start:]:
        candidate = line.strip()
        if candidate:
            return candidate
    return None


def _content_title(lines: Sequence[str]) -> str | None:
    for index, line in enumerate(lines):
        if _normalize_for_match(line) == "ten tai lieu":
            return _next_non_empty_line(lines, index + 1)
    return _next_non_empty_line(lines, 0)


def _strip_parenthetical(value: str) -> str:
    outside = re.sub(r"\s*\([^)]*\)\s*", " ", value).strip()
    return re.sub(r"\s+", " ", outside) or value.strip()


def _labeled_value(line: str) -> tuple[str, str] | None:
    if ":" not in line:
        return None
    label, value = line.split(":", 1)
    normalized_label = _normalize_for_match(label)
    if normalized_label in _DISEASE_LABELS or normalized_label.startswith(
        "ten goi khac"
    ):
        return normalized_label, value.strip()
    return None


def _extract_disease_metadata(
    content: str,
    path: Path,
    title: str,
) -> tuple[str, str, str]:
    """Suy ra tên bệnh và alias từ template mềm, không bắt buộc đủ heading."""

    lines = content.splitlines()
    document_title = _content_title(lines)
    disease: str | None = None

    for index, line in enumerate(lines):
        labeled_heading = _labeled_value(line.strip())
        if labeled_heading and labeled_heading[0] == "ten benh":
            disease = labeled_heading[1] or None
            if disease:
                break
        if _normalize_for_match(line) != "ten benh":
            continue
        labeled_fallback: str | None = None
        for candidate in lines[index + 1 :]:
            candidate = candidate.strip()
            if not candidate:
                continue
            normalized_candidate = _normalize_for_match(candidate)
            if candidate.isupper() and any(
                normalized_candidate.startswith(heading)
                for heading in _SECTION_HEADINGS
            ):
                break
            labeled = _labeled_value(candidate)
            if labeled:
                label, value = labeled
                if value and labeled_fallback is None:
                    labeled_fallback = value
                if label in {"ten benh", "ten tieng viet", "vietnamese name"}:
                    disease = value or None
                    if disease:
                        break
                continue
            disease = candidate
            break
        disease = disease or labeled_fallback
        break

    if not disease:
        disease = _strip_parenthetical(document_title or title)

    aliases: list[str] = []
    seen_aliases: set[str] = set()

    def add_alias(value: str | None) -> None:
        if not value:
            return
        candidate = re.sub(r"\s+", " ", value.replace("|", "/")).strip(
            " ,;/-"
        )
        normalized = _normalize_for_match(candidate)
        if candidate and normalized and normalized not in seen_aliases:
            seen_aliases.add(normalized)
            aliases.append(candidate)

    add_alias(disease)
    add_alias(title)
    add_alias(document_title)
    add_alias(re.sub(r"(?i)^bệnh\s+", "", disease).strip())
    add_alias(
        re.sub(
            r"(?i)\s+trên\s+(?:cây\s+)?[^,;()]+$",
            "",
            disease,
        ).strip()
    )
    crop_words = path.parts[0].replace("_", " ").replace("-", " ").split()
    title_words = title.split()
    crop_prefix = " ".join(title_words[: len(crop_words)])
    if crop_words and _normalize_for_match(crop_prefix) == _normalize_for_match(
        " ".join(crop_words)
    ):
        add_alias(" ".join(title_words[len(crop_words) :]))
    if document_title:
        add_alias(_strip_parenthetical(document_title))
        for parenthetical in re.findall(r"\(([^)]+)\)", document_title):
            add_alias(parenthetical)

    for line in lines:
        labeled = _labeled_value(line.strip())
        if not labeled:
            continue
        _, value = labeled
        for alias in re.split(r"\s*[;,|]\s*", value):
            add_alias(alias)

    disease_tokens = _meaningful_disease_tokens(disease)
    crop_id = _normalize_for_match(path.parts[0]) if len(path.parts) > 1 else ""
    identity_tokens = disease_tokens or tuple(
        _normalize_for_match(path.stem).split()
    )
    disease_id_parts = tuple(filter(None, (crop_id, *identity_tokens)))
    disease_id = "-".join(disease_id_parts) or _normalize_for_match(path.stem)
    return disease, disease_id, _ALIAS_SEPARATOR.join(aliases)


def _identity_header(metadata: dict[str, Any]) -> str:
    fields = [
        f"Tài liệu: {metadata.get('title', 'không rõ')}",
        f"Bệnh: {metadata.get('disease', 'không rõ')}",
    ]
    aliases = str(metadata.get("disease_aliases", "")).strip()
    if aliases:
        fields.append(f"Tên gọi: {aliases}")
    return "\n".join(fields)


def load_documents(data_dir: Path | str = DATA_DIR) -> list[Document]:
    """Đọc mỗi file .txt UTF-8 bên dưới data_dir thành Document."""

    root = Path(data_dir).resolve()
    if not root.is_dir():
        raise RuntimeError(f"Không tìm thấy thư mục dữ liệu: {root}")

    paths = sorted(
        (path for path in root.rglob("*.txt") if path.is_file()),
        key=lambda path: path.as_posix().casefold(),
    )
    documents: list[Document] = []
    for path in paths:
        relative_path = path.relative_to(root).as_posix()
        try:
            content = canonical_text(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError as exc:
            raise RuntimeError(
                f"File không phải UTF-8 hoặc chứa ký tự lỗi: {relative_path}"
            ) from exc
        except OSError as exc:
            raise RuntimeError(f"Không thể đọc file: {relative_path}") from exc

        if not content.strip():
            warnings.warn(f"Bỏ qua file rỗng: {relative_path}", stacklevel=2)
            continue

        relative = Path(relative_path)
        crop = relative.parts[0] if len(relative.parts) > 1 else "unknown"
        title = _humanize_filename(path)
        disease, disease_id, disease_aliases = _extract_disease_metadata(
            without_heading_markers(content),
            relative,
            title,
        )
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": relative_path,
                    "crop": crop,
                    "title": title,
                    "disease": disease,
                    "disease_id": disease_id,
                    "disease_aliases": disease_aliases,
                },
            )
        )

    if not documents:
        raise RuntimeError(
            f"Không có file .txt UTF-8, không rỗng trong: {root}. "
            "Hãy thêm dữ liệu theo dạng data/<crop>/*.txt."
        )
    return documents


def split_documents(
    documents: Sequence[Document],
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    strategy: str | None = None,
    tokenizer: Any = None,
) -> list[Document]:
    """Env/override chọn structure hoặc đường recursive tương thích cách cũ."""

    settings = get_chunking_settings(strategy=strategy)
    if settings.strategy == "structure":
        records = structure_records(documents, settings, tokenizer=tokenizer)
        chunks = [Document(page_content=record["text"], metadata=record["metadata"])
                  for record in records if record["kind"] == "chunk"]
        if not chunks:
            raise RuntimeError("Không tạo được chunk nào từ dữ liệu đầu vào.")
        return chunks

    if chunk_size < 1:
        raise ValueError("chunk_size phải lớn hơn 0")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap phải nằm trong khoảng [0, chunk_size)")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
    )
    chunks = []
    for index, chunk in enumerate(splitter.split_documents(list(documents))):
        content = chunk.page_content.strip()
        if not content:
            continue
        chunks.append(
            Document(
                page_content=f"{_identity_header(chunk.metadata)}\n\n{content}",
                metadata={**chunk.metadata, "end_index": chunk.metadata["start_index"] + len(content),
                          "chunk_index": index, "chunking_strategy": "recursive",
                          "chunking_version": "recursive-v1"},
            )
        )
    if not chunks:
        raise RuntimeError("Không tạo được chunk nào từ dữ liệu đầu vào.")
    return chunks


def new_vector_store(
    embeddings: Embeddings | None,
    persist_directory: Path | str,
    collection_name: str,
) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(Path(persist_directory).resolve()),
        collection_metadata={"hnsw:space": "cosine"},
    )


def build_index(
    data_dir: Path | str = DATA_DIR,
    persist_directory: Path | str | None = None,
    *,
    embeddings: Embeddings | None = None,
    collection_name: str = COLLECTION_NAME,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    strategy: str | None = None,
    tokenizer: Any = None,
) -> tuple[int, int]:
    """Build lại toàn bộ Chroma collection và trả (số file, số chunk)."""

    settings = get_chunking_settings(strategy=strategy)
    persist_path = get_index_directory(persist_directory, strategy=settings.strategy)
    # Không vô tình ghi baseline bằng strategy khác khi CHROMA_DIR bị giữ cố định.
    path = manifest_path(persist_path, collection_name)
    if path.exists():
        import json
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous["chunking"]["strategy"] != settings.strategy:
            raise ValueError("Index đích thuộc strategy khác. Hãy chọn --index-dir/CHROMA_DIR riêng.")
    elif settings.strategy == "structure" and (persist_path / "chroma.sqlite3").exists():
        raise ValueError("Index đích chưa có manifest. Hãy dùng thư mục riêng cho structure để giữ baseline.")
    documents = load_documents(data_dir)
    chunks = split_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=settings.strategy,
        tokenizer=tokenizer,
    )
    embedding_model = embeddings or create_embeddings()
    persist_path.mkdir(parents=True, exist_ok=True)
    manifest = describe_manifest(
        documents, settings, collection=collection_name,
        embedding_model=_embedding_identity(embeddings), chunk_size=chunk_size, chunk_overlap=chunk_overlap,
    )
    vector_store = new_vector_store(
        embedding_model, persist_path, collection_name
    )
    # Nếu add_documents bị lỗi, query phải thấy trạng thái building thay vì đọc index dở dang.
    write_manifest(persist_path, collection_name, manifest)
    vector_store.reset_collection()
    vector_store.add_documents(chunks)
    manifest.update(status="ready", document_count=len(documents), chunk_count=len(chunks))
    write_manifest(persist_path, collection_name, manifest)
    return len(documents), len(chunks)


def _embedding_identity(embeddings: Embeddings | None) -> str:
    if embeddings is None:
        return EMBEDDING_MODEL
    # Caller inject model cần chịu trách nhiệm cấu hình weights; test offline cũng dùng nhánh này.
    return str(getattr(embeddings, "model_name", "") or
               f"injected:{type(embeddings).__module__}.{type(embeddings).__qualname__}")


def _load_vector_store(
    persist_directory: Path | str | None,
    embeddings: Embeddings | None,
    collection_name: str,
    *,
    load_embeddings: bool = True,
) -> Chroma:
    persist_path = get_index_directory(persist_directory)
    if not persist_path.is_dir():
        raise RuntimeError(
            f"Index chưa được tạo tại {persist_path}. Hãy chạy `python main.py index` với cùng config trước."
        )

    validate_query_manifest(
        persist_path, collection_name, _embedding_identity(embeddings) if load_embeddings else None,
    )

    embedding_model = None
    if load_embeddings:
        embedding_model = embeddings or create_embeddings()
    vector_store = new_vector_store(
        embedding_model,
        persist_path,
        collection_name,
    )
    if not vector_store.get(limit=1, include=[]).get("ids"):
        raise RuntimeError(
            "Index đang rỗng. Hãy chạy lại `python main.py index`."
        )
    return vector_store


def retrieve_with_debug(
    question: str,
    k: int = TOP_K,
    *,
    embeddings: Embeddings | None = None,
    persist_directory: Path | str | None = None,
    collection_name: str = COLLECTION_NAME,
    vector_store: Any | None = None,
    mode: str | None = None,
    rerank: bool | None = None,
    scope: MetadataScope | None = None,
) -> SearchResult:
    """Tìm chunk và giữ trace thứ hạng; không suy metadata filter từ query.

    `scope` chỉ đến từ nhãn có cấu trúc (xem taxonomy.py), không từ câu hỏi.
    """

    question = question.strip()
    if not question:
        raise ValueError("Câu hỏi không được để trống.")
    if k < 1:
        raise ValueError("k phải lớn hơn 0")

    settings = get_retrieval_settings(mode=mode, rerank=rerank)
    store = vector_store
    if store is None:
        store = _load_vector_store(
            persist_directory,
            embeddings,
            collection_name,
            load_embeddings=settings.mode != "bm25",
        )
    return search_store(question, store, k=k, settings=settings, scope=scope)


def retrieve(
    question: str,
    k: int = TOP_K,
    *,
    embeddings: Embeddings | None = None,
    persist_directory: Path | str | None = None,
    collection_name: str = COLLECTION_NAME,
    vector_store: Any | None = None,
    mode: str | None = None,
    rerank: bool | None = None,
    scope: MetadataScope | None = None,
) -> list[Document]:
    """API gọn cho caller chỉ cần các chunk; debug nằm ở retrieve_with_debug."""
    result = retrieve_with_debug(
        question, k,
        embeddings=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
        vector_store=vector_store,
        mode=mode,
        rerank=rerank,
        scope=scope,
    )
    return result.documents


def format_context(documents: Sequence[Document]) -> str:
    """Gắn nhãn nguồn cho từng chunk để LLM có thể trích dẫn."""

    blocks: list[str] = []
    for index, document in enumerate(documents, start=1):
        source = str(document.metadata.get("source", "không rõ nguồn"))
        blocks.append(
            f"[Nguồn {index}: {source}]\n{document.page_content.strip()}"
        )
    return "\n\n".join(blocks)


# Viết theo NFC để khớp cả khi LLM trả về "Nguồn" ở dạng tổ hợp dấu khác.
_CITATION_PATTERN = re.compile(
    unicodedata.normalize("NFC", r"\[\s*Nguồn\s+(\d+)"), re.IGNORECASE
)


def cited_source_numbers(answer: str) -> list[int]:
    """Các số nguồn LLM đã dẫn, theo thứ tự xuất hiện trong câu trả lời."""

    return [
        int(number)
        for number in _CITATION_PATTERN.findall(unicodedata.normalize("NFC", answer))
    ]


def invalid_citations(answer: str, document_count: int) -> list[int]:
    """Số nguồn ngoài [1, document_count]: nhãn trỏ ra ngoài NGỮ CẢNH lượt này."""

    return sorted(
        {
            number
            for number in cited_source_numbers(answer)
            if not 1 <= number <= document_count
        }
    )


def _unique_sources(documents: Sequence[Document]) -> list[str]:
    return list(
        dict.fromkeys(
            str(document.metadata.get("source", "không rõ nguồn"))
            for document in documents
        )
    )


def ask(
    question: str,
    *,
    k: int = TOP_K,
    llm: BaseChatModel | None = None,
    embeddings: Embeddings | None = None,
    persist_directory: Path | str | None = None,
    collection_name: str = COLLECTION_NAME,
    vector_store: Any | None = None,
    mode: str | None = None,
    rerank: bool | None = None,
) -> tuple[str, list[str]]:
    """Retrieve context, gọi LLM và trả ``(answer, source_paths)``."""

    question = question.strip()
    documents = retrieve(
        question,
        k,
        embeddings=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
        vector_store=vector_store,
        mode=mode,
        rerank=rerank,
    )
    return _answer_from_documents(question, documents, llm=llm)


def _answer_from_documents(
    question: str,
    documents: list[Document],
    *,
    llm: BaseChatModel | None = None,
    history: list[BaseMessage] | None = None,
    retrieval_query: str | None = None,
    subject: str | None = None,
    examples: Sequence[dict[str, str]] | None = None,
    callbacks: list[BaseCallbackHandler] | None = None,
) -> tuple[str, list[str]]:
    """Dùng chung bước generation cho lệnh ask một lần và phiên hỏi liên tục."""
    if not documents:
        return NO_CONTEXT_ANSWER, []

    context = format_context(documents)
    chain = RAG_PROMPT | (llm or create_chat_model()) | StrOutputParser()
    answer = chain.invoke({
        "context": context, "question": question, "history": history or [],
        "retrieval_query": retrieval_query or question,
        # Rỗng thì prompt giống hệt trước khi có tham số này (CLI không đổi).
        "subject_block": (
            f"{SUBJECT_HEADER}\n{subject.strip()}\n\n"
            if subject and subject.strip() else ""
        ),
        "examples_block": examples_block(examples),
    }, config={"callbacks": callbacks} if callbacks else None).strip()
    if not answer:
        raise RuntimeError("LLM trả về nội dung rỗng.")
    # Cảnh báo thay vì raise: câu trả lời vẫn tới người dùng, nhưng nhãn trỏ ra
    # ngoài NGỮ CẢNH là dấu hiệu trích dẫn bịa và cần thấy được khi chạy/đo.
    invalid = invalid_citations(answer, len(documents))
    if invalid:
        warnings.warn(
            f"Câu trả lời dẫn nguồn không có trong NGỮ CẢNH: {invalid}; "
            f"lượt này chỉ có {len(documents)} nguồn.",
            stacklevel=2,
        )
    return answer, _unique_sources(documents)


def generate_answer(
    question: str,
    documents: list[Document],
    *,
    llm: BaseChatModel | None = None,
    history: list[BaseMessage] | None = None,
    retrieval_query: str | None = None,
    subject: str | None = None,
    examples: Sequence[dict[str, str]] | None = None,
    callbacks: list[BaseCallbackHandler] | None = None,
) -> tuple[str, list[str]]:
    """Bước sinh câu trả lời dùng chung cho CLI, RAGSession và API server.

    `examples`: câu trả lời mẫu admin đã duyệt ({question, answer}), chỉ API gửi.
    `callbacks` để API đọc model/token/finish_reason mà LLM báo về.
    """
    return _answer_from_documents(
        question, documents, llm=llm, history=history,
        retrieval_query=retrieval_query, subject=subject, examples=examples, callbacks=callbacks,
    )


@dataclass
class ChatResult:
    """Kết quả một lượt; debug giúp thấy query thật đã gửi vào retrieval."""

    question: str
    retrieval_query: str
    answer: str
    sources: list[str]
    retrieval: SearchResult
    history_turns_used: int

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "retrieval_query": self.retrieval_query,
            "history_turns_used": self.history_turns_used,
            "retrieval": self.retrieval.to_debug_dict(),
        }


class RAGSession:
    """Giữ tài nguyên để xử lý nhiều câu hỏi tuần tự trong cùng process.

    Tạo một session bên ngoài vòng lặp xử lý câu hỏi. Embedding và reranker
    dùng cache model có sẵn; session giữ Chroma, LLM client và lịch sử giới hạn
    trong RAM. Mỗi session có lịch sử riêng, không lưu xuống đĩa.
    """

    def __init__(
        self,
        *,
        k: int = TOP_K,
        mode: str | None = None,
        rerank: bool | None = None,
        history_turns: int | None = None,
        embeddings: Embeddings | None = None,
        llm: BaseChatModel | None = None,
        vector_store: Any | None = None,
        persist_directory: Path | str | None = None,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        if k < 1:
            raise ValueError("k phải lớn hơn 0")
        # Chốt config cho cả phiên, tránh đổi mode nhưng dùng nhầm tài nguyên.
        self.k = k
        self.settings = get_retrieval_settings(mode=mode, rerank=rerank)
        self._memory = ConversationMemory(get_chat_settings(history_turns=history_turns))
        self._embeddings = embeddings
        self._llm = llm
        self._store = vector_store
        self._persist_directory = get_index_directory(persist_directory)
        self._collection_name = collection_name

    def warmup(self) -> None:
        """Mở index, nạp model local cần dùng; không gọi LLM hoặc sinh câu trả lời.

        Có thể gọi trước khi nhận câu hỏi. Gọi lại sẽ dùng tài nguyên đã nạp.
        BM25 không rerank chỉ mở Chroma, không nạp model nào.
        """
        if self._store is None:
            self._store = _load_vector_store(
                self._persist_directory,
                self._embeddings,
                self._collection_name,
                load_embeddings=self.settings.mode != "bm25",
            )
        if self.settings.reranker_enabled:
            load_reranker(self.settings.reranker_model, self.settings.reranker_device)

    def search(self, question: str) -> SearchResult:
        """Retrieval độc lập: không dùng/thay đổi lịch sử và không gọi LLM."""
        question = question.strip()
        if not question:
            raise ValueError("Câu hỏi không được để trống.")
        self.warmup()
        return search_store(question, self._store, k=self.k, settings=self.settings)

    def ask(self, question: str) -> tuple[str, list[str]]:
        """API gọn, cùng định dạng (answer, sources) như trước khi có history."""
        result = self.ask_with_debug(question)
        return result.answer, result.sources

    @property
    def history(self) -> tuple[ChatTurn, ...]:
        return self._memory.turns

    def clear_history(self) -> None:
        """Bắt đầu hội thoại mới, giữ nguyên model và kết nối đã nạp."""
        self._memory.clear()

    def _get_llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = create_chat_model()
        return self._llm

    def ask_with_debug(self, question: str) -> ChatResult:
        """Lịch sử → query độc lập → retrieval → trả lời → lưu lượt thành công."""
        question = question.strip()
        if not question:
            raise ValueError("Câu hỏi không được để trống.")
        history = self._memory.messages()
        history_turns_used = len(self.history)
        retrieval_query = (
            rewrite_question(question, history, self._get_llm()) if history else question
        )
        result = self.search(retrieval_query)
        # Câu đầu không có context vẫn không tạo client/gọi provider.
        llm = self._get_llm() if result.documents else None
        answer, sources = _answer_from_documents(
            question, result.documents, llm=llm, history=history, retrieval_query=retrieval_query,
        )
        # Chỉ lưu sau khi cả lượt thành công; lỗi rewrite/retrieval/provider
        # không thêm một câu hỏi dở dang vào lịch sử khi người dùng thử lại.
        self._memory.add(question, retrieval_query, answer)
        return ChatResult(question, retrieval_query, answer, sources, result, history_turns_used)
