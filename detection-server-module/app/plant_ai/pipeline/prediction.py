from PIL import Image, UnidentifiedImageError

import io
import base64
import logging
import os
from datetime import datetime
from pathlib import Path


from app.plant_ai.model_manager import has_ievit_model
from app.plant_ai.models.plant_classifier import PlantClassifier


from app.plant_ai.pipeline.segmentation import (
    segment_leaves
)


from app.plant_ai.pipeline.disease import (
    DiseaseService
)



logger = logging.getLogger(__name__)


class InvalidImageError(ValueError):
    """Ảnh không dùng được để chẩn đoán (không mở được, hoặc không thấy lá cây).

    Là ValueError nên route /api/chat trả HTTP 400 kèm thông báo này.
    """


NOT_AN_IMAGE = "Ảnh không hợp lệ: không mở được file này. Hãy gửi ảnh JPG hoặc PNG chụp lá cây."
NOT_A_LEAF = (
    "Ảnh không hợp lệ: mình không thấy lá cây nào trong ảnh. "
    "Hãy chụp rõ một chiếc lá, đủ sáng, lá chiếm phần lớn khung hình."
)


# ======================================================
# MODEL CACHE
# ======================================================

plant_classifier = None

disease_service = None

SEGMENTATION_DEBUG_ENABLED = os.getenv(
    "SAVE_SEGMENTATION_PREVIEW",
    "true",
).lower() in {"1", "true", "yes", "on"}
SEGMENTATION_DEBUG_DIR = (
    Path(__file__).resolve().parents[3] / "debug" / "segmentation"
)


def save_segmentation_preview(image: Image.Image) -> Path | None:
    """Save the latest YOLO output for local inspection without logging pixels."""
    if not SEGMENTATION_DEBUG_ENABLED:
        return None

    SEGMENTATION_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    filename = datetime.now().strftime("segmented_%Y%m%d_%H%M%S_%f.png")
    output_path = SEGMENTATION_DEBUG_DIR / filename
    image.save(output_path, format="PNG")
    logger.info("YOLO segment preview saved: %s", output_path)
    return output_path




def get_plant_classifier():

    """
    Load ConvNeXt plant classifier
    chỉ khi cần dùng
    """

    global plant_classifier


    if plant_classifier is None:


        plant_classifier = PlantClassifier()



    return plant_classifier





def get_disease_service():


    """
    Load IEViT service
    """

    global disease_service


    if disease_service is None:


        disease_service = DiseaseService()



    return disease_service





# ======================================================
# MAIN PIPELINE
# ======================================================


async def predict_image(

    image_bytes: bytes

):


    # ==================================================
    # Decode image
    # ==================================================

    try:

        image = Image.open(

            io.BytesIO(

                image_bytes

            )

        ).convert(

            "RGB"

        )

    except (UnidentifiedImageError, OSError) as exc:

        raise InvalidImageError(NOT_AN_IMAGE) from exc




    # ==================================================
    # STEP 0
    # GenYOLO: có lá cây trong ảnh không? Không có thì dừng ngay,
    # không chạy ConvNeXt/IEViT và không gọi LLM.
    # ==================================================

    leaves = segment_leaves(image)

    if not leaves.looks_like_leaf:

        logger.info(
            "Rejected non-leaf image: leaves=%d best_conf=%.2f area=%.2f",
            leaves.leaf_count, leaves.best_confidence, leaves.leaf_area_ratio,
        )

        raise InvalidImageError(NOT_A_LEAF)




    # ==================================================
    # STEP 1
    # Detect plant with ConvNeXt
    # ==================================================

    classifier = get_plant_classifier()



    plant_result = classifier.predict(

        image

    )



    plant_name = plant_result["name"]



    # Cây không có checkpoint bệnh (Orange, Squash): trả kết quả nhận diện cây,
    # disease=None, để lượt chat đi tiếp như câu hỏi chưa rõ bệnh.
    if not has_ievit_model(plant_name):

        logger.info("No IEViT model for %s; skip disease step", plant_name)

        return {
            "success": True,
            "plant": plant_result,
            "segmentation": None,
            "disease": None,
            "message": f"No disease model for {plant_name}",
        }



    # ==================================================
    # STEP 2
    # GenYOLO segmentation (đã chạy ở STEP 0)
    # ==================================================

    segmented_image = leaves.image

    save_segmentation_preview(segmented_image)




    # ==================================================
    # STEP 3
    # IEViT disease classification
    # ==================================================

    disease = get_disease_service()



    disease_result = disease.predict(

        plant_name,

        segmented_image

    )





    # ==================================================
    # Encode YOLO output image
    # ==================================================

    buffer = io.BytesIO()



    segmented_image.save(

        buffer,

        format="PNG"

    )



    segmented_base64 = base64.b64encode(

        buffer.getvalue()

    ).decode(

        "utf-8"

    )





    # ==================================================
    # FINAL RESPONSE
    # ==================================================

    return {


        "success":

            True,



        "plant":

            plant_result,



        "segmentation":

        {

            "image":

                segmented_base64

        },



        "disease":

            disease_result,



        "message":

            "Prediction completed"

    }
