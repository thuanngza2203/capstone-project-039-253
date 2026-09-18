"""Thống kê nhanh chunk trong Chroma; chỉ đọc metadata, không nạp model."""

from collections import Counter

from config import COLLECTION_NAME, get_index_directory
from rag import new_vector_store


index_dir = get_index_directory()

print(f"Đang đọc Vector DB tại: {index_dir}")

# store.get() không cần embedding: bỏ qua việc nạp model để script chạy tức thì.
store = new_vector_store(
    None,
    index_dir,
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
