import torch
import timm

from torchvision import transforms

from app.plant_ai.model_manager import download_plant_classifier_model


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)




PLANT_CLASSES = [

    "Apple",
    "Cherry",
    "Corn",
    "Grape",
    "Orange",
    "Peach",
    "Pepper",
    "Potato",
    "Squash",
    "Strawberry",
    "Tomato"

]


EXCLUDED_CLASSES = {
    "Orange",
    "Squash"
}



class PlantClassifier:


    def __init__(self):

        self.model = self.load_model()


        self.transform = transforms.Compose([

            transforms.Resize(
                (256,256)
            ),

            transforms.CenterCrop(
                224
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




    def clean_state_dict(self,state_dict):

        new_state = {}

        for k,v in state_dict.items():

            if k.startswith("module."):

                new_state[k[7:]]=v

            else:

                new_state[k]=v

        return new_state




    def load_model(self):

        checkpoint = torch.load(

            download_plant_classifier_model(),

            map_location=DEVICE

        )


        if "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        else:

            state_dict = checkpoint



        state_dict = self.clean_state_dict(
            state_dict
        )


        num_classes = (
            state_dict[
                "head.fc.weight"
            ].shape[0]
        )


        model = timm.create_model(

            "convnext_tiny",

            pretrained=False,

            num_classes=num_classes

        )


        model.load_state_dict(
            state_dict,
            strict=True
        )


        model.to(
            DEVICE
        )


        model.eval()


        return model




    @torch.no_grad()

    def predict(self,image):


        x = self.transform(
            image
        )


        x = x.unsqueeze(
            0
        ).to(
            DEVICE
        )


        output = self.model(
            x
        )


        probs = torch.softmax(

            output,

            dim=1

        )[0]



        probs = probs.cpu().numpy()



        for i,name in enumerate(PLANT_CLASSES):

            if name in EXCLUDED_CLASSES:

                probs[i]=0



        probs = probs / probs.sum()



        idx = probs.argmax()



        return {


            "name":
                PLANT_CLASSES[idx],


            "confidence":
                float(probs[idx])


        }