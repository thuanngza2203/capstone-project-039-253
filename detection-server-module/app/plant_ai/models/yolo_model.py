from ultralytics import YOLO

from app.plant_ai.model_manager import download_yolo_model


yolo_model = None


def get_yolo_model():

    global yolo_model


    if yolo_model is None:

        path = download_yolo_model()

        yolo_model = YOLO(path)


    return yolo_model
