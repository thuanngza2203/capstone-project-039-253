"""Bảng ánh xạ nhãn detection → tài liệu phải khớp data/ thật và taxonomy của normalizer."""

from __future__ import annotations

from pathlib import Path

import pytest

import taxonomy
from taxonomy import resolve_scope

DATA = Path(__file__).resolve().parents[1] / "data"

# Chép từ detection-server-module/app/prompts/query_normalizer_prompt.py.
NORMALIZER_TAXONOMY = {
    "apple": ["apple_scab", "black_rot", "cedar_apple_rust", "healthy"],
    "cherry": ["powdery_mildew", "healthy"],
    "corn": ["cercospora_leaf_spot", "common_rust", "northern_leaf_blight", "healthy"],
    "grape": ["black_rot", "esca_black_measles", "leaf_blight", "healthy"],
    "peach": ["bacterial_spot", "healthy"],
    "pepper": ["bacterial_spot", "healthy"],
    "potato": ["early_blight", "late_blight", "healthy"],
    "strawberry": ["leaf_scorch", "healthy"],
    "tomato": [
        "bacterial_spot", "early_blight", "late_blight", "leaf_mold", "septoria_leaf_spot",
        "spider_mites", "target_spot", "tomato_yellow_leaf_curl_virus", "tomato_mosaic_virus",
        "healthy",
    ],
}


def test_every_mapped_document_exists() -> None:
    """Đổi tên file trong data/ mà quên sửa bảng thì test này đỏ ngay (đã xảy ra 21/09)."""
    missing = [entry.source for entry in taxonomy.DISEASES
               if entry.source and not (DATA / entry.source).is_file()]
    assert missing == []


def test_every_document_in_corpus_is_mapped() -> None:
    """Thêm tài liệu mới mà không khai báo thì scope `document` không bao giờ tới được nó."""
    mapped = {entry.source for entry in taxonomy.DISEASES if entry.source}
    corpus = {path.relative_to(DATA).as_posix() for path in DATA.rglob("*.txt")}
    assert corpus - mapped == set()


def test_every_crop_folder_with_documents_has_a_plant() -> None:
    # Thư mục rỗng (như data/orange/) không có gì để tìm nên không cần khai báo.
    folders = {path.parent.name for path in DATA.rglob("*.txt")}
    assert folders <= {plant.crop for plant in taxonomy.PLANTS}


@pytest.mark.parametrize(("plant", "disease"), [
    (plant, disease) for plant, diseases in NORMALIZER_TAXONOMY.items() for disease in diseases
])
def test_every_normalizer_label_is_understood(plant: str, disease: str) -> None:
    assert resolve_scope(plant, disease).status in {"document", "healthy", "unsupported_disease"}


def test_coverage_gaps_are_exactly_the_ones_in_the_plan() -> None:
    gaps = sorted(
        (plant, disease) for plant, diseases in NORMALIZER_TAXONOMY.items() for disease in diseases
        if resolve_scope(plant, disease).status == "unsupported_disease"
    )
    assert gaps == [("grape", "black_rot"), ("grape", "leaf_blight"), ("potato", "late_blight")]


@pytest.mark.parametrize(("plant", "disease", "source"), [
    ("Apple", "Apple___Black_rot", "apple/apple_black_rot.txt"),
    ("apple", "Scab", "apple/apple_scab.txt"),
    ("Apple", "Apple_scab", "apple/apple_scab.txt"),
    ("Apple", "Cedar_apple_rust", "apple/apple_cedar_rust.txt"),
    ("Apple", "Rust", "apple/apple_cedar_rust.txt"),
    ("Grape", "Esca_(Black_Measles)", "grape/grape_esca.txt"),
    ("Tomato", "Tomato_Yellow_Leaf_Curl_Virus", "tomato/tomato_yellow_leaf_curl_virus.txt"),
    ("tomato", "YellowLeaf_Curl_Virus", "tomato/tomato_yellow_leaf_curl_virus.txt"),
    ("Tomato", "Tomato___Tomato_mosaic_virus", "tomato/tomato_mosaic_virus.txt"),
    ("Pepper,_bell", "Pepper,_bell___Bacterial_spot", "pepper_bell/pepper_bell_bacterial_spot.txt"),
    ("Corn_(maize)", "Cercospora_leaf_spot Gray_leaf_spot", "corn/corn_gray_leaf_spot.txt"),
    ("corn", "Common_rust_", "corn/corn_common_rust.txt"),
    ("Tomato", "Spider_mites Two-spotted_spider_mite", "tomato/tomato_spider_mites.txt"),
    ("cà chua", "leaf_mold", "tomato/tomato_leaf_mold.txt"),
])
def test_detector_label_variants_map_to_one_document(plant: str, disease: str, source: str) -> None:
    scope = resolve_scope(plant, disease)
    assert scope.status == "document"
    assert scope.sources == (source,)
    assert scope.metadata_scope().sources == (source,)


@pytest.mark.parametrize(("plant", "disease", "status", "searchable"), [
    (None, None, "none", True),
    ("  ", "", "none", True),
    (None, "healthy", "none", True),
    ("Soybean", None, "unknown_plant", True),
    ("pepper", None, "crop", True),
    ("Tomato", "healthy", "healthy", True),
    ("potato", "late_blight", "unsupported_disease", False),
    ("apple", "made_up_label", "unknown_disease", False),
    (None, "made_up_label", "unknown_disease", False),
    (None, "leaf_mold", "document", True),
    (None, "bacterial_spot", "disease_multi_crop", True),
    (None, "isariopsis_leaf_spot", "unsupported_disease", False),
])
def test_scope_status(plant, disease, status: str, searchable: bool) -> None:
    scope = resolve_scope(plant, disease)
    assert scope.status == status
    assert scope.searchable is searchable


def test_crop_scope_uses_corpus_folder_name() -> None:
    scope = resolve_scope("pepper", None)
    assert scope.crop == "pepper_bell"
    assert scope.metadata_scope().crop == "pepper_bell"


def test_multi_crop_disease_lists_every_existing_document() -> None:
    scope = resolve_scope(None, "bacterial_spot")
    assert set(scope.sources) == {
        "peach/peach_bacterial_spot.txt",
        "pepper_bell/pepper_bell_bacterial_spot.txt",
        "tomato/tomato_bacterial_spot.txt",
    }


def test_unknown_label_is_reported_back_for_alias_fixing() -> None:
    scope = resolve_scope("Apple", "Apple___Weird_Class")
    assert "Apple___Weird_Class" in scope.message
    assert scope.received == {"plant_type": "Apple", "disease": "Apple___Weird_Class"}
    assert scope.metadata_scope() is not None  # crop có sẵn, nhưng searchable=False chặn trước.
    assert not scope.searchable


# Tên class đọc từ checkpoint thật trên HF vietthien1404/plant-disease-models (23/09).
# Tên cây là class_names của ConvNeXt; tên bệnh là class_names của IEViT từng cây.
# Định dạng không nhất quán giữa các cây (có/không tiền tố), nên phải khóa bằng test.
DETECTOR_CLASSES = {
    "Apple": ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy"],
    "Cherry": ["Powdery_mildew", "healthy"],
    "Corn": ["Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
             "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy"],
    "Grape": ["Black_rot", "Esca", "Leaf_blight", "healthy"],
    "Peach": ["Bacterial_spot", "healthy"],
    "Pepper": ["Bacterial_spot", "healthy"],
    "Potato": ["Early_blight", "Late_blight", "healthy"],
    "Strawberry": ["Leaf_scorch", "healthy"],
    "Tomato": ["Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold", "Mosaic_virus",
               "Septoria_leaf_spot", "Spider_mites", "Target_Spot", "Yellow_Leaf_Curl_Virus", "healthy"],
}


@pytest.mark.parametrize(("plant", "disease"), [
    (plant, disease) for plant, classes in DETECTOR_CLASSES.items() for disease in classes
])
def test_every_real_detector_class_is_understood(plant: str, disease: str) -> None:
    scope = resolve_scope(plant, disease)
    expected = "healthy" if disease.lower().endswith("healthy") else {"document", "unsupported_disease"}
    assert scope.status in (expected if isinstance(expected, set) else {expected}), scope.message


def test_detector_gaps_match_normalizer_gaps() -> None:
    gaps = sorted(
        (plant, disease) for plant, classes in DETECTOR_CLASSES.items() for disease in classes
        if resolve_scope(plant, disease).status == "unsupported_disease"
    )
    assert gaps == [("Grape", "Black_rot"), ("Grape", "Leaf_blight"), ("Potato", "Late_blight")]


@pytest.mark.parametrize("plant", ["Orange", "Squash"])
def test_classifier_plants_without_ievit(plant: str) -> None:
    """ConvNeXt có 11 lớp, IEViT chỉ 9 cây. RAG vẫn phải xử lý được tên cây này."""
    assert resolve_scope(plant, None).status in {"unknown_plant", "crop"}
