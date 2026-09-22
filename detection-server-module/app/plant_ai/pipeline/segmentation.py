import os

import numpy as np
from PIL import Image

from app.plant_ai.models.yolo_model import get_yolo_model


BLACK_BACKGROUND = (0, 0, 0)
MAX_LEAVES = int(os.getenv("YOLO_TOP_K", "2"))
MIN_MASK_AREA = int(os.getenv("YOLO_MIN_MASK_AREA", "100"))


def segment_leaf(image: Image.Image) -> Image.Image:
    """Keep the highest-ranked leaf masks and paint everything else black."""
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
        print("[YOLO SEGMENT] detected=0 selected=0")
        return image

    height, width = image.size[1], image.size[0]
    raw_masks = result.masks.data.detach().cpu().numpy()

    if getattr(result, "boxes", None) is not None:
        confidences = result.boxes.conf.detach().cpu().numpy()
    else:
        confidences = np.ones(len(raw_masks), dtype=np.float32)

    candidates: list[tuple[float, np.ndarray]] = []
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
        candidates.append((score, mask))

    if not candidates:
        print("[YOLO SEGMENT] detected=0 selected=0")
        return image

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected_masks = [mask for _, mask in candidates[:MAX_LEAVES]]
    print(
        f"[YOLO SEGMENT] detected={len(candidates)} "
        f"selected={len(selected_masks)}"
    )
    combined_mask = np.any(np.stack(selected_masks), axis=0)

    source = np.asarray(image)
    segmented = np.empty_like(source)
    segmented[...] = BLACK_BACKGROUND
    segmented[combined_mask] = source[combined_mask]

    return Image.fromarray(segmented.astype("uint8"))
