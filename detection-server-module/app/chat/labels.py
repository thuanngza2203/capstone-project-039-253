"""Đưa nhãn thô của detector về key trong prompt normalizer.

Chỉ dùng để dựng `retrieval_query` (câu tìm tài liệu): `Apple___Apple_scab`
-> `apple_scab`. Trường `plant_type`/`disease` gửi sang RAG vẫn giữ nguyên nhãn gốc.
"""

import re


# Cây: tên ConvNeXt/PlantVillage -> key normalizer.
_PLANT_KEYS = {
    "corn_maize": "corn",
    "maize": "corn",
    "pepper_bell": "pepper",
    "bell_pepper": "pepper",
    "cherry_including_sour": "cherry",
}

# Tiền tố cây có thể dính trước nhãn bệnh; dài trước ngắn sau.
_PLANT_PREFIXES = sorted(
    {
        "apple", "cherry", "cherry_including_sour", "corn", "corn_maize", "grape",
        "peach", "pepper", "pepper_bell", "potato", "squash", "strawberry", "tomato",
    },
    key=len,
    reverse=True,
)

# Key bệnh trong prompt normalizer. Đã là key chuẩn thì không cắt tiền tố
# (`apple_scab` không được thành `scab`).
NORMALIZER_DISEASE_KEYS = frozenset({
    "apple_scab", "black_rot", "cedar_apple_rust", "powdery_mildew",
    "cercospora_leaf_spot", "common_rust", "northern_leaf_blight",
    "esca_black_measles", "leaf_blight", "bacterial_spot", "early_blight",
    "late_blight", "leaf_scorch", "leaf_mold", "septoria_leaf_spot",
    "spider_mites", "target_spot", "tomato_yellow_leaf_curl_virus",
    "tomato_mosaic_virus", "healthy",
})

# class_names của IEViT khác key normalizer sau khi đã bỏ tiền tố cây.
_DISEASE_KEYS = {
    "cercospora_leaf_spot_gray_leaf_spot": "cercospora_leaf_spot",
    "esca": "esca_black_measles",
    "esca_black_measles": "esca_black_measles",
    "yellow_leaf_curl_virus": "tomato_yellow_leaf_curl_virus",
    "mosaic_virus": "tomato_mosaic_virus",
    "spider_mites_two_spotted_spider_mite": "spider_mites",
}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def canonical_plant(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    slug = _slug(value)
    return _PLANT_KEYS.get(slug, slug)


def canonical_disease(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    slug = _slug(value)
    if slug in NORMALIZER_DISEASE_KEYS:
        return slug
    for prefix in _PLANT_PREFIXES:
        if slug.startswith(prefix + "_"):
            slug = slug[len(prefix) + 1:]
            break
    return _DISEASE_KEYS.get(slug, slug)
