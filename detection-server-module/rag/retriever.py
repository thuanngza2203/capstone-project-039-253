from functools import lru_cache

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from .config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL,
    TOP_K,
)

@lru_cache(maxsize=1)
def get_db():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

def retrieve(query, k=TOP_K):
    return [document for document, _score in retrieve_with_scores(query, k=k)]


def retrieve_with_scores(query, k=TOP_K):
    """Return Chroma documents with their raw distance scores.

    Chroma's score is a distance: lower values indicate a closer match.
    """
    return get_db().similarity_search_with_score(query, k=k)
