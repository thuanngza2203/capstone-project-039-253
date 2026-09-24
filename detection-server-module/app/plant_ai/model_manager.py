import logging
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download


logger = logging.getLogger(__name__)


# ======================================================
# PATH
# Tính theo thư mục module để chạy từ đâu cũng dùng chung cache checkpoint.
# ======================================================

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


# ======================================================
# 1. GENYOLO MODEL
# ======================================================

YOLO_MODEL_URL = (
    "https://github.com/aaslihanyildirim/GenYOLO-Leaf/"
    "releases/download/shared_best_yolov8_models/"
    "genyolo_leaf_yolov8s.pt"
)

YOLO_MODEL_PATH = MODEL_DIR / "genyolo_leaf_yolov8s.pt"


def download_yolo_model() -> str:
    if YOLO_MODEL_PATH.exists():
        logger.info("YOLO model exists")
        return str(YOLO_MODEL_PATH)

    logger.info("Downloading GenYOLO model...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    response = requests.get(YOLO_MODEL_URL, stream=True, timeout=60)
    response.raise_for_status()

    # Ghi file tạm rồi đổi tên: tải hỏng giữa chừng không để lại checkpoint dở.
    partial = YOLO_MODEL_PATH.with_suffix(".part")
    with open(partial, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    partial.replace(YOLO_MODEL_PATH)

    logger.info("YOLO downloaded: %s", YOLO_MODEL_PATH)
    return str(YOLO_MODEL_PATH)


# ======================================================
# 2. HUGGING FACE CONFIG
# ======================================================

HF_REPO_ID = "vietthien1404/plant-disease-models"


# ======================================================
# 3. PLANT CLASSIFIER MODEL
# ConvNeXt Tiny
# ======================================================

PLANT_MODEL_NAME = "best_stage2_convnext_tiny_focus_tomato.pth"

PLANT_MODEL_PATH = MODEL_DIR / PLANT_MODEL_NAME


def download_plant_classifier_model() -> str:
    if PLANT_MODEL_PATH.exists():
        logger.info("Plant classifier model exists")
        return str(PLANT_MODEL_PATH)

    logger.info("Downloading Plant classifier model...")
    downloaded_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=PLANT_MODEL_NAME,
        local_dir=MODEL_DIR,
    )
    logger.info("Plant classifier downloaded: %s", downloaded_path)
    return downloaded_path


# ======================================================
# 4. IEViT MODELS
# Chỉ 9 cây có checkpoint bệnh. ConvNeXt còn nhận ra Orange, Squash:
# các cây đó đi qua `has_ievit_model()` = False, không gọi download.
# ======================================================

IEVIT_MODELS = {
    "Apple": "best_ievit_stage3_apple_yolo.pth",
    "Cherry": "best_ievit_stage3_cherry_yolo.pth",
    "Corn": "best_ievit_stage3_corn_yolo.pth",
    "Grape": "best_ievit_stage3_grape_yolo.pth",
    "Peach": "best_ievit_stage3_peach_yolo.pth",
    "Pepper": "best_ievit_stage3_pepper_yolo.pth",
    "Potato": "best_ievit_stage3_potato_yolo.pth",
    "Strawberry": "best_ievit_stage3_strawberry_yolo.pth",
    "Tomato": "best_ievit_stage3_tomato_yolo.pth",
}


def has_ievit_model(plant_name: str) -> bool:
    return plant_name in IEVIT_MODELS


def download_ievit_model(plant_name: str) -> str:
    if plant_name not in IEVIT_MODELS:
        raise ValueError(f"Không có model IEViT cho {plant_name}")

    filename = IEVIT_MODELS[plant_name]
    local_path = MODEL_DIR / filename

    if local_path.exists():
        logger.info("%s model exists", plant_name)
        return str(local_path)

    logger.info("Downloading IEViT model: %s", plant_name)
    downloaded_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        local_dir=MODEL_DIR,
    )
    logger.info("Downloaded: %s", downloaded_path)
    return downloaded_path
