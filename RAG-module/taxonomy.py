"""Ánh xạ nhãn cây/bệnh của detection-server sang tài liệu trong data/.

Key chuẩn lấy theo taxonomy mà Query Normalizer của detection-server dùng
(`apple`, `black_rot`...). Nhãn từ detector có nhiều dạng khác (`Apple___Black_rot`,
`Scab`, `Esca_(Black_Measles)`), nên mọi đầu vào đều qua `label_key()` rồi tra bảng
alias tường minh. Không đoán tên bệnh từ câu hỏi tự do: câu hỏi vẫn đi vào
retrieval như cũ, bảng này chỉ dùng cho nhãn có cấu trúc.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from retrieval import MetadataScope


def label_key(value: str) -> str:
    """`Apple___Black_rot`, `apple black-rot`, `Esca_(Black_Measles)` → dạng snake_case."""
    folded = "".join(
        character for character in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(character)
    ).replace("đ", "d")
    return re.sub(r"[^a-z0-9]+", "_", folded).strip("_")


@dataclass(frozen=True)
class Plant:
    key: str
    crop: str  # Tên thư mục trong data/ và giá trị metadata `crop` của chunk.
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Disease:
    plant: str
    key: str
    source: str | None  # None: bệnh có trong taxonomy nhưng kho chưa có tài liệu.
    aliases: tuple[str, ...] = ()


PLANTS: tuple[Plant, ...] = (
    Plant("apple", "apple", ("tao",)),
    Plant("cherry", "cherry", ("cherry_including_sour", "anh_dao")),
    Plant("corn", "corn", ("corn_maize", "maize", "ngo", "bap")),
    Plant("grape", "grape", ("nho",)),
    Plant("peach", "peach", ("dao",)),
    Plant("pepper", "pepper_bell", ("pepper_bell", "bell_pepper", "ot", "ot_chuong")),
    Plant("potato", "potato", ("khoai_tay",)),
    Plant("strawberry", "strawberry", ("dau_tay",)),
    Plant("tomato", "tomato", ("ca_chua",)),
    # Detector chưa nhận diện bí, nhưng câu hỏi văn bản vẫn dùng được tài liệu này.
    Plant("squash", "squash", ("bi",)),
)

DISEASES: tuple[Disease, ...] = (
    Disease("apple", "apple_scab", "apple/apple_scab.txt", ("scab",)),
    Disease("apple", "black_rot", "apple/apple_black_rot.txt"),
    Disease("apple", "cedar_apple_rust", "apple/apple_cedar_rust.txt", ("rust", "cedar_rust")),
    Disease("cherry", "powdery_mildew", "cherry/cherry_powdery_mildew.txt"),
    # Cercospora zeae-maydis gây bệnh đốm xám lá: cùng một bệnh, hai tên.
    Disease("corn", "cercospora_leaf_spot", "corn/corn_gray_leaf_spot.txt",
            ("gray_leaf_spot", "grey_leaf_spot", "cercospora_leaf_spot_gray_leaf_spot")),
    Disease("corn", "common_rust", "corn/corn_common_rust.txt"),
    Disease("corn", "northern_leaf_blight", "corn/corn_northern_leaf_blight.txt", ("leaf_blight",)),
    Disease("grape", "black_rot", None),
    Disease("grape", "esca_black_measles", "grape/grape_esca.txt", ("esca",)),
    Disease("grape", "leaf_blight", None,
            ("leaf_blight_isariopsis_leaf_spot", "isariopsis_leaf_spot")),
    Disease("peach", "bacterial_spot", "peach/peach_bacterial_spot.txt"),
    Disease("pepper", "bacterial_spot", "pepper_bell/pepper_bell_bacterial_spot.txt"),
    Disease("potato", "early_blight", "potato/potato_early_blight.txt"),
    Disease("potato", "late_blight", None),
    Disease("strawberry", "leaf_scorch", "strawberry/strawberry_leaf_scorch.txt"),
    Disease("tomato", "bacterial_spot", "tomato/tomato_bacterial_spot.txt"),
    Disease("tomato", "early_blight", "tomato/tomato_early_blight.txt"),
    Disease("tomato", "late_blight", "tomato/tomato_late_blight.txt"),
    Disease("tomato", "leaf_mold", "tomato/tomato_leaf_mold.txt"),
    Disease("tomato", "septoria_leaf_spot", "tomato/tomato_septoria_leaf_spot.txt"),
    Disease("tomato", "spider_mites", "tomato/tomato_spider_mites.txt",
            ("spider_mites_two_spotted_spider_mite", "two_spotted_spider_mite")),
    Disease("tomato", "target_spot", "tomato/tomato_target_spot.txt"),
    Disease("tomato", "tomato_yellow_leaf_curl_virus", "tomato/tomato_yellow_leaf_curl_virus.txt",
            ("yellow_leaf_curl_virus", "yellowleaf_curl_virus", "tylcv")),
    Disease("tomato", "tomato_mosaic_virus", "tomato/tomato_mosaic_virus.txt",
            ("mosaic_virus", "tomv")),
    Disease("squash", "powdery_mildew", "squash/squash_powdery_mildew.txt"),
)

HEALTHY = "healthy"

# Trạng thái mà retrieval không được chạy: tìm theo cây lúc này dễ trả lời sai bệnh.
NOT_SEARCHABLE = frozenset({"unsupported_disease", "unknown_disease"})


def _plant_index() -> dict[str, Plant]:
    index: dict[str, Plant] = {}
    for plant in PLANTS:
        for name in (plant.key, plant.crop, *plant.aliases):
            index[label_key(name)] = plant
    return index


_PLANT_BY_KEY = _plant_index()


def find_plant(value: str | None) -> Plant | None:
    return _PLANT_BY_KEY.get(label_key(value)) if value and value.strip() else None


def _disease_keys(entry: Disease) -> set[str]:
    return {label_key(name) for name in (entry.key, *entry.aliases)}


def _candidate_keys(raw: str, plant: Plant) -> list[str]:
    """Nhãn detector hay kèm tên cây phía trước (`Apple___Black_rot`); thử cả hai."""
    key = label_key(raw)
    candidates = [key]
    for prefix in (plant.key, plant.crop, *plant.aliases):
        head = label_key(prefix) + "_"
        if key.startswith(head):
            candidates.append(key[len(head):])
    return candidates


def is_healthy(value: str, plant: Plant) -> bool:
    """IEViT trả `Apple___healthy`, `Corn_(maize)___healthy`: phải bỏ tiền tố cây trước."""
    return HEALTHY in _candidate_keys(value, plant)


def find_disease(plant: Plant, value: str) -> Disease | None:
    candidates = _candidate_keys(value, plant)
    for entry in DISEASES:
        if entry.plant == plant.key and _disease_keys(entry).intersection(candidates):
            return entry
    return None


@dataclass(frozen=True)
class ResolvedScope:
    """Kết quả tra nhãn: tìm ở đâu, và lý do bằng lời để hiển thị/ghi log."""

    status: str
    plant: str | None = None
    disease: str | None = None
    crop: str | None = None
    sources: tuple[str, ...] = ()
    message: str = ""
    received: dict[str, str | None] = field(default_factory=dict)

    @property
    def searchable(self) -> bool:
        return self.status not in NOT_SEARCHABLE

    def metadata_scope(self) -> MetadataScope | None:
        if self.sources:
            return MetadataScope(sources=self.sources)
        if self.crop:
            return MetadataScope(crop=self.crop)
        return None


def resolve_scope(plant_type: str | None, disease: str | None) -> ResolvedScope:
    """Chuyển nhãn detector/normalizer thành phạm vi tìm kiếm, theo bảng trong plan."""
    received = {"plant_type": plant_type, "disease": disease}
    plant_given = bool(plant_type and plant_type.strip())
    disease_given = bool(disease and disease.strip())
    if not plant_given and not disease_given:
        return ResolvedScope("none", message="Không có cây/bệnh; tìm trên toàn bộ tài liệu.",
                             received=received)

    plant = find_plant(plant_type) if plant_given else None
    if plant_given and plant is None:
        return ResolvedScope(
            "unknown_plant",
            message=f"Không nhận ra cây '{plant_type}'; tìm trên toàn bộ tài liệu.",
            received=received,
        )

    if plant is not None and not disease_given:
        return ResolvedScope("crop", plant=plant.key, crop=plant.crop,
                             message=f"Tìm trong các tài liệu của cây {plant.key}.",
                             received=received)

    if plant is not None and is_healthy(disease, plant):
        return ResolvedScope("healthy", plant=plant.key, disease=HEALTHY, crop=plant.crop,
                             message=f"Cây {plant.key} được báo là khỏe; tìm tài liệu phòng bệnh của cây này.",
                             received=received)

    if plant is not None:
        entry = find_disease(plant, disease)
        if entry is None:
            return ResolvedScope(
                "unknown_disease", plant=plant.key, crop=plant.crop,
                message=(f"Không nhận ra nhãn bệnh '{disease}' của cây {plant.key}. "
                         "Cần bổ sung alias trong taxonomy.py."),
                received=received,
            )
        if entry.source is None:
            return ResolvedScope(
                "unsupported_disease", plant=plant.key, disease=entry.key, crop=plant.crop,
                message=f"Kho tài liệu chưa có bệnh {entry.key} trên cây {plant.key}.",
                received=received,
            )
        return ResolvedScope("document", plant=plant.key, disease=entry.key, crop=plant.crop,
                             sources=(entry.source,),
                             message=f"Tìm trong tài liệu {entry.source}.", received=received)

    # `Apple___healthy` không kèm plant_type: tiền tố vẫn cho biết là cây nào.
    prefixed = [entry for entry in PLANTS if HEALTHY in _candidate_keys(disease, entry)[1:]]
    if prefixed:
        plant = prefixed[0]
        return ResolvedScope("healthy", plant=plant.key, disease=HEALTHY, crop=plant.crop,
                             message=f"Cây {plant.key} được báo là khỏe; tìm tài liệu phòng bệnh của cây này.",
                             received=received)
    if label_key(disease) == HEALTHY:
        return ResolvedScope("none", disease=HEALTHY,
                             message="Báo khỏe nhưng không rõ cây; tìm trên toàn bộ tài liệu.",
                             received=received)

    # Chỉ có bệnh: tìm mọi cây có nhãn đó, vì cùng tên bệnh có thể thuộc nhiều cây.
    matches = [entry for plant_entry in PLANTS
               if (entry := find_disease(plant_entry, disease)) is not None]
    if not matches:
        return ResolvedScope("unknown_disease",
                             message=f"Không nhận ra nhãn bệnh '{disease}'. Cần bổ sung alias trong taxonomy.py.",
                             received=received)
    sources = tuple(entry.source for entry in matches if entry.source)
    if not sources:
        plants = ", ".join(entry.plant for entry in matches)
        return ResolvedScope("unsupported_disease", disease=matches[0].key,
                             message=f"Kho tài liệu chưa có bệnh {matches[0].key} ({plants}).",
                             received=received)
    if len(matches) == 1:
        only = matches[0]
        plant_entry = find_plant(only.plant)
        return ResolvedScope("document", plant=only.plant, disease=only.key,
                             crop=plant_entry.crop if plant_entry else None, sources=sources,
                             message=f"Tìm trong tài liệu {only.source}.", received=received)
    plants = ", ".join(entry.plant for entry in matches)
    return ResolvedScope(
        "disease_multi_crop", disease=matches[0].key, sources=sources,
        message=f"Bệnh {matches[0].key} có trên nhiều cây ({plants}); tìm trong {len(sources)} tài liệu.",
        received=received,
    )


def taxonomy_table() -> list[dict]:
    """Dữ liệu cho GET /v1/taxonomy: app biết bệnh nào trả lời được."""
    return [
        {
            "plant": plant.key,
            "crop": plant.crop,
            "plant_aliases": list(plant.aliases),
            "diseases": [
                {"disease": entry.key, "source": entry.source,
                 "has_document": entry.source is not None, "aliases": list(entry.aliases)}
                for entry in DISEASES if entry.plant == plant.key
            ],
        }
        for plant in PLANTS
    ]
