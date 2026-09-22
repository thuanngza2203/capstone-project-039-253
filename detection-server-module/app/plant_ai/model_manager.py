import os
import requests

from huggingface_hub import hf_hub_download


# ======================================================
# PATH
# ======================================================

MODEL_DIR = "models"

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)



# ======================================================
# 1. GENYOLO MODEL
# ======================================================

YOLO_MODEL_URL = (

    "https://github.com/aaslihanyildirim/GenYOLO-Leaf/"
    "releases/download/shared_best_yolov8_models/"
    "genyolo_leaf_yolov8s.pt"

)


YOLO_MODEL_PATH = (

    f"{MODEL_DIR}/genyolo_leaf_yolov8s.pt"

)



def download_yolo_model():


    if os.path.exists(YOLO_MODEL_PATH):

        print(
            "YOLO model exists"
        )

        return YOLO_MODEL_PATH



    print(
        "Downloading GenYOLO model..."
    )



    r = requests.get(

        YOLO_MODEL_URL,

        stream=True

    )


    r.raise_for_status()



    with open(

        YOLO_MODEL_PATH,

        "wb"

    ) as f:


        for chunk in r.iter_content(

            chunk_size=8192

        ):

            f.write(chunk)



    print(

        "YOLO downloaded:",

        YOLO_MODEL_PATH

    )


    return YOLO_MODEL_PATH




# ======================================================
# 2. HUGGING FACE CONFIG
# ======================================================


HF_REPO_ID = (

    "vietthien1404/plant-disease-models"

)




# ======================================================
# 3. PLANT CLASSIFIER MODEL
# ConvNeXt Tiny
# ======================================================


PLANT_MODEL_NAME = (

    "best_stage2_convnext_tiny_focus_tomato.pth"

)



PLANT_MODEL_PATH = (

    f"{MODEL_DIR}/{PLANT_MODEL_NAME}"

)



def download_plant_classifier_model():


    if os.path.exists(
        PLANT_MODEL_PATH
    ):

        print(
            "Plant classifier model exists"
        )


        return PLANT_MODEL_PATH



    print(
        "Downloading Plant classifier model..."
    )



    downloaded_path = hf_hub_download(

        repo_id=HF_REPO_ID,

        filename=PLANT_MODEL_NAME,

        local_dir=MODEL_DIR

    )



    print(

        "Plant classifier downloaded:",

        downloaded_path

    )



    return downloaded_path





# ======================================================
# 4. IEViT MODELS
# ======================================================


IEVIT_MODELS = {


    "Apple":

    "best_ievit_stage3_apple_yolo.pth",



    "Cherry":

    "best_ievit_stage3_cherry_yolo.pth",



    "Corn":

    "best_ievit_stage3_corn_yolo.pth",



    "Grape":

    "best_ievit_stage3_grape_yolo.pth",



    "Peach":

    "best_ievit_stage3_peach_yolo.pth",



    "Pepper":

    "best_ievit_stage3_pepper_yolo.pth",



    "Potato":

    "best_ievit_stage3_potato_yolo.pth",



    "Strawberry":

    "best_ievit_stage3_strawberry_yolo.pth",



    "Tomato":

    "best_ievit_stage3_tomato_yolo.pth"

}




def download_ievit_model(

    plant_name

):


    if plant_name not in IEVIT_MODELS:


        raise ValueError(

            f"Không có model IEViT cho {plant_name}"

        )



    filename = IEVIT_MODELS[plant_name]



    local_path = (

        f"{MODEL_DIR}/{filename}"

    )



    if os.path.exists(local_path):


        print(

            f"{plant_name} model exists"

        )


        return local_path




    print(

        f"Downloading IEViT model: {plant_name}"

    )



    downloaded_path = hf_hub_download(

        repo_id=HF_REPO_ID,

        filename=filename,

        local_dir=MODEL_DIR

    )



    print(

        "Downloaded:",

        downloaded_path

    )


    return downloaded_path