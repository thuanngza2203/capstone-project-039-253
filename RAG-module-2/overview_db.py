from collections import Counter

from config import CHROMA_DIR, COLLECTION_NAME, create_embeddings
from rag import _new_vector_store

store = _new_vector_store(
    create_embeddings(),
    CHROMA_DIR,
    COLLECTION_NAME,
)

data = store.get(
    include=["documents", "metadatas"]
)

documents = data["documents"]
metadatas = data["metadatas"]

print(f"Tổng số chunk: {len(documents)}")

sources = Counter(
    metadata.get("source", "unknown")
    for metadata in metadatas
)

crops = Counter(
    metadata.get("crop", "unknown")
    for metadata in metadatas
)

diseases = Counter(
    metadata.get("disease", "unknown")
    for metadata in metadatas
)

print("\n=== CHUNK THEO FILE ===")
for source, count in sources.items():
    print(f"{source}: {count}")

print("\n=== CHUNK THEO CÂY ===")
for crop, count in crops.items():
    print(f"{crop}: {count}")

print("\n=== CHUNK THEO BỆNH ===")
for disease, count in diseases.items():
    print(f"{disease}: {count}")