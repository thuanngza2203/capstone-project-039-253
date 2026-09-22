import argparse
import shutil

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL,
)


def load_documents():
    docs = []
    for file_path in sorted(DATA_DIR.rglob("*.txt")):
        relative_source = file_path.relative_to(DATA_DIR).as_posix()
        docs.append(
            Document(
                page_content=file_path.read_text(encoding="utf-8"),
                metadata={"source": relative_source},
            )
        )
    return docs


def build_index(*, reset: bool = False):
    docs = load_documents()
    if not docs:
        raise RuntimeError(f"No .txt files found under {DATA_DIR}")

    if reset and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )
    Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )
    print(
        f"Indexed {len(docs)} files into {len(chunks)} chunks "
        f"({COLLECTION_NAME}, {CHROMA_DIR})"
    )

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Build the domain Chroma index")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing rag/chroma_db index before indexing",
    )
    args = parser.parse_args()
    build_index(reset=args.reset)
