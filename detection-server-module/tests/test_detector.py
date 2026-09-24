import asyncio
import io

import pytest
from PIL import Image

from app.chat import detector as detector_module
from app.plant_ai.pipeline import prediction
from app.plant_ai.pipeline.segmentation import LeafSegmentation


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "green").save(buffer, format="PNG")
    return buffer.getvalue()


class FakeClassifier:
    def __init__(self, name):
        self.name = name
        self.calls = 0

    def predict(self, image):
        self.calls += 1
        return {"name": self.name, "confidence": 0.81}


def must_not_run(*args, **kwargs):
    raise AssertionError("Không được chạy bước này")


def leaves(count=1, confidence=0.9, area=0.5):
    return lambda image: LeafSegmentation(image, count, confidence, area)


def detect(image_bytes):
    return asyncio.run(detector_module.PlantAIDetector().detect(image_bytes))


@pytest.fixture(autouse=True)
def real_predict(monkeypatch):
    monkeypatch.setattr(detector_module, "predict_image", prediction.predict_image)


def test_plant_without_ievit_skips_disease_step(monkeypatch):
    monkeypatch.setattr(prediction, "segment_leaves", leaves())
    monkeypatch.setattr(prediction, "get_plant_classifier", lambda: FakeClassifier("Orange"))
    monkeypatch.setattr(prediction, "get_disease_service", must_not_run)

    result = detect(png_bytes())
    assert result.plant == "Orange"
    assert result.disease is None
    assert result.confidence == 0.81


@pytest.mark.parametrize(("count", "confidence", "area"), [
    (0, 0.0, 0.0),      # YOLO không thấy lá
    (1, 0.57, 0.19),    # ảnh COCO bé + bóng bay: lá nhầm, không chắc và nhỏ
    (1, 0.39, 0.01),    # hoa trong bình: vùng "lá" rất nhỏ
])
def test_non_leaf_image_is_rejected_before_classification(monkeypatch, count, confidence, area):
    classifier = FakeClassifier("Apple")
    monkeypatch.setattr(prediction, "segment_leaves", leaves(count, confidence, area))
    monkeypatch.setattr(prediction, "get_plant_classifier", lambda: classifier)
    monkeypatch.setattr(prediction, "get_disease_service", must_not_run)

    with pytest.raises(prediction.InvalidImageError, match="Ảnh không hợp lệ"):
        detect(png_bytes())
    assert classifier.calls == 0


@pytest.mark.parametrize(("confidence", "area"), [
    (0.94, 0.63),   # lá rõ
    (0.87, 0.11),   # lá chắc chắn nhưng nhỏ trong khung
    (0.29, 0.42),   # YOLO không chắc nhưng lá chiếm gần nửa ảnh (tomato bacterial spot)
])
def test_leaf_image_is_accepted(monkeypatch, confidence, area):
    monkeypatch.setattr(prediction, "segment_leaves", leaves(1, confidence, area))
    monkeypatch.setattr(prediction, "get_plant_classifier", lambda: FakeClassifier("Orange"))
    assert detect(png_bytes()).plant == "Orange"


def test_file_that_is_not_an_image_is_rejected(monkeypatch):
    monkeypatch.setattr(prediction, "segment_leaves", must_not_run)
    with pytest.raises(prediction.InvalidImageError, match="không mở được"):
        detect(b"%PDF-1.4 not an image")
