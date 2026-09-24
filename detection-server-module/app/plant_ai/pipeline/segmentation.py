import logging
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

from app.plant_ai.models.yolo_model import get_yolo_model


logger = logging.getLogger(__name__)


BLACK_BACKGROUND = (0, 0, 0)
MAX_LEAVES = int(os.getenv("YOLO_TOP_K", "2"))
MIN_MASK_AREA = int(os.getenv("YOLO_MIN_MASK_AREA", "100"))

# Ảnh được coi là ảnh lá khi GenYOLO thấy một lá chắc chắn (>= LEAF_MIN_CONFIDENCE),
# hoặc thấy lá (conf >= 0.25) chiếm ít nhất LEAF_MIN_AREA_RATIO diện tích ảnh.
# Đo ngày 23/09 trên 24 ảnh PlantVillage + 26 ảnh không phải lá (COCO, ảnh chụp
# màn hình, ảnh trơn): lá thật có conf >= 0.76 hoặc chiếm >= 39% ảnh; ảnh nhầm
# cao nhất là conf 0.57 / 19% ảnh. Chỉnh lại khi có thêm ảnh chụp ngoài vườn.
LEAF_MIN_CONFIDENCE = float(os.getenv("LEAF_MIN_CONFIDENCE", "0.6"))
LEAF_MIN_AREA_RATIO = float(os.getenv("LEAF_MIN_AREA_RATIO", "0.25"))


@dataclass
class LeafSegmentation:
    image: Image.Image          # lá giữ lại, nền tô đen (ảnh gốc nếu không thấy lá)
    leaf_count: int             # số mask lá qua ngưỡng conf 0.25 và diện tích tối thiểu
    best_confidence: float      # độ tin cậy cao nhất trong các lá
    leaf_area_ratio: float      # tỉ lệ diện tích ảnh mà các lá đó phủ

    @property
    def looks_like_leaf(self) -> bool:
        if self.leaf_count == 0:
            return False
        return (
            self.best_confidence >= LEAF_MIN_CONFIDENCE
            or self.leaf_area_ratio >= LEAF_MIN_AREA_RATIO
        )


def segment_leaves(image: Image.Image) -> LeafSegmentation:
    """Chạy GenYOLO một lần: giữ các lá tốt nhất và đo xem ảnh có phải ảnh lá không."""
    model = get_yolo_model()
    results = model.predict(
        source=image,
        conf=0.25,
        imgsz=640,
        retina_masks=True,
        verbose=False,
    )
    result = results[0]

    if result.masks is None:
        logger.info("YOLO segment detected=0 selected=0")
        return LeafSegmentation(image, 0, 0.0, 0.0)

    height, width = image.size[1], image.size[0]
    raw_masks = result.masks.data.detach().cpu().numpy()

    if getattr(result, "boxes", None) is not None:
        confidences = result.boxes.conf.detach().cpu().numpy()
    else:
        confidences = np.ones(len(raw_masks), dtype=np.float32)

    candidates: list[tuple[float, float, np.ndarray]] = []
    for index, raw_mask in enumerate(raw_masks):
        mask_image = Image.fromarray(
            ((raw_mask > 0.5) * 255).astype("uint8")
        ).resize(
            (width, height),
            Image.Resampling.NEAREST,
        )
        mask = np.asarray(mask_image) > 0
        area = int(mask.sum())
        if area < MIN_MASK_AREA:
            continue

        confidence = (
            float(confidences[index])
            if index < len(confidences)
            else 1.0
        )
        score = confidence * (area / (height * width))
        candidates.append((score, confidence, mask))

    if not candidates:
        logger.info("YOLO segment detected=0 selected=0")
        return LeafSegmentation(image, 0, 0.0, 0.0)

    best_confidence = max(confidence for _, confidence, _ in candidates)
    leaf_area_ratio = float(np.any(np.stack([m for _, _, m in candidates]), axis=0).mean())

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected_masks = [mask for _, _, mask in candidates[:MAX_LEAVES]]
    logger.info(
        "YOLO segment detected=%d selected=%d best_conf=%.2f area=%.2f",
        len(candidates), len(selected_masks), best_confidence, leaf_area_ratio,
    )
    combined_mask = np.any(np.stack(selected_masks), axis=0)

    source = np.asarray(image)
    segmented = np.empty_like(source)
    segmented[...] = BLACK_BACKGROUND
    segmented[combined_mask] = source[combined_mask]

    return LeafSegmentation(
        Image.fromarray(segmented.astype("uint8")),
        len(candidates),
        best_confidence,
        leaf_area_ratio,
    )


def segment_leaf(image: Image.Image) -> Image.Image:
    """Keep the highest-ranked leaf masks and paint everything else black."""
    return segment_leaves(image).image
