from abc import ABC, abstractmethod

from app.schemas import DetectionResult

from app.plant_ai.pipeline.prediction import (
    predict_image
)



class DiseaseDetector(ABC):

    @abstractmethod
    async def detect(
        self,
        image_bytes: bytes,
        filename: str | None = None
    ):
        raise NotImplementedError





class PlantAIDetector(DiseaseDetector):

    def __init__(self):
        pass


    async def detect(
        self,
        image_bytes: bytes,
        filename: str | None = None
    ):

        result = await predict_image(image_bytes)

        plant = result["plant"]
        disease = result["disease"]

        return DetectionResult(
            plant=plant["name"],
            disease=disease["name"],
            confidence=float(disease["confidence"]),
            source="plant_ai_pipeline"
        )
