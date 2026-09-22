from PIL import Image

import io
import base64
import os
from datetime import datetime
from pathlib import Path


from app.plant_ai.models.plant_classifier import PlantClassifier


from app.plant_ai.pipeline.segmentation import (
    segment_leaf
)


from app.plant_ai.pipeline.disease import (
    DiseaseService
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
    print(f"[YOLO SEGMENT] preview saved: {output_path}")
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

    image = Image.open(

        io.BytesIO(

            image_bytes

        )

    ).convert(

        "RGB"

    )




    # ==================================================
    # STEP 1
    # Detect plant with ConvNeXt
    # ==================================================

    classifier = get_plant_classifier()



    plant_result = classifier.predict(

        image

    )



    plant_name = plant_result["name"]



    # ==================================================
    # STEP 2
    # GenYOLO segmentation
    # ==================================================

    segmented_image = segment_leaf(

        image

    )

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
