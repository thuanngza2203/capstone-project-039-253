"""Pipeline RAG tối giản: load -> split -> embed -> retrieve -> generate."""

from __future__ import annotations

import re
import unicodedata
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    TOP_K,
    create_chat_model,
    create_embeddings,
)
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
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
"""

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            (
                "NGỮ CẢNH:\n{context}\n\nCÂU HỎI:\n{question}\n\n"
                "Hãy trả lời theo các quy tắc trên."
            ),
        ),
    ]
)

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
_NEGATION_MARKERS = (
    ("khong", "phai"),
    ("loai", "tru"),
    ("not",),
    ("exclude",),
)


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


def _contains_non_negated_alias(
    question_tokens: tuple[str, ...],
    alias_tokens: tuple[str, ...],
) -> bool:
    """Khớp alias theo cụm từ và bỏ qua các cụm bị phủ định rõ ràng."""

    alias_length = len(alias_tokens)
    for start in range(len(question_tokens) - alias_length + 1):
        if question_tokens[start : start + alias_length] != alias_tokens:
            continue
        prefix = question_tokens[max(0, start - 4) : start]
        is_negated = any(
            any(
                prefix[index : index + len(marker)] == marker
                for index in range(len(prefix) - len(marker) + 1)
            )
            for marker in _NEGATION_MARKERS
        )
        if not is_negated:
            return True
    return False


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
            content = path.read_text(encoding="utf-8")
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
            content,
            relative,
            title,
        )
        documents.append(
            Document(
                page_content=content.strip(),
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
) -> list[Document]:
    """Chia tài liệu theo độ dài ký tự và giữ nguyên metadata nguồn."""

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
    for chunk in splitter.split_documents(list(documents)):
        content = chunk.page_content.strip()
        if not content:
            continue
        chunks.append(
            Document(
                page_content=f"{_identity_header(chunk.metadata)}\n\n{content}",
                metadata=dict(chunk.metadata),
            )
        )
    if not chunks:
        raise RuntimeError("Không tạo được chunk nào từ dữ liệu đầu vào.")
    return chunks


def _new_vector_store(
    embeddings: Embeddings,
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
    persist_directory: Path | str = CHROMA_DIR,
    *,
    embeddings: Embeddings | None = None,
    collection_name: str = COLLECTION_NAME,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> tuple[int, int]:
    """Build lại toàn bộ Chroma collection và trả (số file, số chunk)."""

    documents = load_documents(data_dir)
    chunks = split_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    embedding_model = embeddings or create_embeddings()
    persist_path = Path(persist_directory).resolve()
    persist_path.mkdir(parents=True, exist_ok=True)
    vector_store = _new_vector_store(
        embedding_model, persist_path, collection_name
    )
    vector_store.reset_collection()
    vector_store.add_documents(chunks)
    return len(documents), len(chunks)


def _load_vector_store(
    persist_directory: Path | str,
    embeddings: Embeddings | None,
    collection_name: str,
) -> Chroma:
    persist_path = Path(persist_directory).resolve()
    if not persist_path.is_dir():
        raise RuntimeError(
            "Index chưa được tạo. Hãy chạy `python main.py index` trước."
        )

    vector_store = _new_vector_store(
        embeddings or create_embeddings(),
        persist_path,
        collection_name,
    )
    if not vector_store.get(limit=1, include=[]).get("ids"):
        raise RuntimeError(
            "Index đang rỗng. Hãy chạy lại `python main.py index`."
        )
    return vector_store


def _metadata_catalog(store: Any) -> list[dict[str, Any]]:
    """Lấy metadata duy nhất từ index; fake store cũ vẫn được hỗ trợ."""

    get_records = getattr(store, "get", None)
    if not callable(get_records):
        return []
    try:
        result = get_records(include=["metadatas"])
    except TypeError:
        # Vector store được inject có thể không triển khai API get của Chroma.
        return []

    catalog: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for metadata in result.get("metadatas") or []:
        if not isinstance(metadata, dict):
            continue
        disease_id = str(metadata.get("disease_id", "")).strip()
        source = str(metadata.get("source", "")).strip()
        key = (disease_id, source)
        if disease_id and key not in seen:
            seen.add(key)
            catalog.append(metadata)
    return catalog


def _detect_disease_id(
    question: str,
    catalog: Sequence[dict[str, Any]],
) -> str | None:
    """Trả disease_id khi câu hỏi gọi đích danh đúng một bệnh."""

    question_tokens = _meaningful_disease_tokens(question)
    matches: set[str] = set()
    for metadata in catalog:
        disease_id = str(metadata.get("disease_id", "")).strip()
        crop_tokens = _meaningful_disease_tokens(
            str(metadata.get("crop", ""))
        )
        aliases = str(metadata.get("disease_aliases", "")).split(
            _ALIAS_SEPARATOR
        )
        aliases.append(str(metadata.get("disease", "")))
        for alias in aliases:
            alias_tokens = _meaningful_disease_tokens(alias)
            if not alias_tokens or alias_tokens == crop_tokens:
                continue
            if _contains_non_negated_alias(question_tokens, alias_tokens):
                matches.add(disease_id)
                break
    return next(iter(matches)) if len(matches) == 1 else None


def retrieve(
    question: str,
    k: int = TOP_K,
    *,
    embeddings: Embeddings | None = None,
    persist_directory: Path | str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    vector_store: Any | None = None,
) -> list[Document]:
    """Tìm chunk, tự lọc khi câu hỏi gọi đích danh đúng một bệnh."""

    question = question.strip()
    if not question:
        raise ValueError("Câu hỏi không được để trống.")
    if k < 1:
        raise ValueError("k phải lớn hơn 0")

    store = vector_store
    if store is None:
        store = _load_vector_store(
            persist_directory,
            embeddings,
            collection_name,
        )
    disease_id = _detect_disease_id(question, _metadata_catalog(store))
    if disease_id:
        return list(
            store.similarity_search(
                question,
                k=k,
                filter={"disease_id": disease_id},
            )
        )
    return list(store.similarity_search(question, k=k))


def format_context(documents: Sequence[Document]) -> str:
    """Gắn nhãn nguồn cho từng chunk để Gemini có thể trích dẫn."""

    blocks: list[str] = []
    for index, document in enumerate(documents, start=1):
        source = str(document.metadata.get("source", "không rõ nguồn"))
        blocks.append(
            f"[Nguồn {index}: {source}]\n{document.page_content.strip()}"
        )
    return "\n\n".join(blocks)


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
    persist_directory: Path | str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    vector_store: Any | None = None,
) -> tuple[str, list[str]]:
    """Retrieve context, goi Gemini va tra ``(answer, source_paths)``."""

    question = question.strip()
    documents = retrieve(
        question,
        k,
        embeddings=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
        vector_store=vector_store,
    )
    if not documents:
        return (
            "Kho tài liệu hiện tại chưa có đủ thông tin liên quan để trả lời câu hỏi này.",
            [],
        )

    context = format_context(documents)
    chain = RAG_PROMPT | (llm or create_chat_model()) | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question}).strip()
    if not answer:
        raise RuntimeError("Gemini trả về nội dung rỗng.")
    return answer, _unique_sources(documents)
