import torch
from PIL import Image
from torchvision import transforms

from app.plant_ai.model_manager import download_ievit_model
from app.plant_ai.models.ievit import IEViT


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DiseaseService:


    def __init__(self):

        self.model_cache = {}

        self.transform = transforms.Compose([

            transforms.Resize(
                (224,224)
            ),

            transforms.ToTensor(),

            transforms.Normalize(

                mean=[
                    0.485,
                    0.456,
                    0.406
                ],

                std=[
                    0.229,
                    0.224,
                    0.225
                ]

            )

        ])



    def load_model(
        self,
        plant_name
    ):


        if plant_name in self.model_cache:

            return self.model_cache[plant_name]



        model_path = download_ievit_model(
            plant_name
        )


        checkpoint = torch.load(
            model_path,
            map_location=DEVICE
        )


        class_names = checkpoint[
            "class_names"
        ]


        model = IEViT(
            num_classes=len(class_names)
        )


        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )


        model.to(
            DEVICE
        )


        model.eval()



        self.model_cache[plant_name] = (
            model,
            class_names
        )


        return (
            model,
            class_names
        )




    def predict(
        self,
        plant_name,
        image: Image.Image
    ):


        model, class_names = self.load_model(
            plant_name
        )


        tensor = self.transform(
            image
        )


        tensor = tensor.unsqueeze(
            0
        )


        tensor = tensor.to(
            DEVICE
        )


        with torch.no_grad():

            output = model(
                tensor
            )


            probs = torch.softmax(
                output,
                dim=1
            )


            idx = torch.argmax(
                probs,
                dim=1
            ).item()



        return {

            "name":
                class_names[idx],

            "confidence":
                float(
                    probs[0][idx]
                ),

            "probabilities":
                {
                    class_names[i]:
                    float(probs[0][i])

                    for i in range(
                        len(class_names)
                    )
                }

        }
