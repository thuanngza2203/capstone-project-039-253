import re
from pathlib import Path

import pytest

from app.chat.labels import NORMALIZER_DISEASE_KEYS, canonical_disease, canonical_plant
from app.chat.query import RetrievalQueryBuilder
from app.schemas import Action, Intent, ResolvedQuery, RouteDecision

# class_names của checkpoint IEViT, đọc từ HF ngày 23/09.
IEVIT_CLASSES = {
    "Apple": ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy"],
    "Cherry": ["Powdery_mildew", "healthy"],
    "Corn": [
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
        "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    ],
    "Grape": ["Black_rot", "Esca", "Leaf_blight", "healthy"],
    "Peach": ["Bacterial_spot", "healthy"],
    "Pepper": ["Bacterial_spot", "healthy"],
    "Potato": ["Early_blight", "Late_blight", "healthy"],
    "Strawberry": ["Leaf_scorch", "healthy"],
    "Tomato": [
        "Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold", "Mosaic_virus",
        "Septoria_leaf_spot", "Spider_mites", "Target_Spot", "Yellow_Leaf_Curl_Virus", "healthy",
    ],
}

PROMPT = Path(__file__).resolve().parents[1] / "app" / "prompts" / "query_normalizer_prompt.py"


def normalizer_taxonomy() -> dict[str, set[str]]:
    text = PROMPT.read_text(encoding="utf-8")
    section = text.split("DANH SÁCH CÂY VÀ BỆNH HỆ THỐNG HỖ TRỢ", 1)[1].split("QUY TẮC CHUẨN HÓA", 1)[0]
    taxonomy: dict[str, set[str]] = {}
    plant = None
    for line in section.splitlines():
        line = line.strip()
        if re.fullmatch(r"[a-z_]+:", line):
            plant = line[:-1]
            taxonomy[plant] = set()
        elif line.startswith("- ") and plant:
            taxonomy[plant].add(line[2:])
    return taxonomy


def test_disease_key_set_matches_prompt():
    keys = set().union(*normalizer_taxonomy().values())
    assert keys == NORMALIZER_DISEASE_KEYS


@pytest.mark.parametrize(
    ("plant", "label"),
    [(plant, label) for plant, labels in IEVIT_CLASSES.items() for label in labels],
)
def test_every_ievit_class_maps_to_a_normalizer_key(plant, label):
    taxonomy = normalizer_taxonomy()
    assert canonical_plant(plant) in taxonomy
    assert canonical_disease(label) in taxonomy[canonical_plant(plant)]


def test_normalizer_keys_are_unchanged():
    for plant, diseases in normalizer_taxonomy().items():
        assert canonical_plant(plant) == plant
        for disease in diseases:
            assert canonical_disease(disease) == disease


def test_retrieval_query_uses_canonical_labels():
    query = RetrievalQueryBuilder().build(
        resolved=ResolvedQuery(plant="Apple", disease="Apple___Apple_scab", intent=Intent.TREATMENT),
        decision=RouteDecision(action=Action.ACCEPT_QUERY),
        fallback_normalized_query="x",
    )
    assert query == "Cách điều trị bệnh apple_scab trên cây apple"
