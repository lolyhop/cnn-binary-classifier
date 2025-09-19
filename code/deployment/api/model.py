import os
import typing as tp

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

MODEL_PATH: str = os.getenv("MODEL_PATH", "/app/models/cats_dogs_model.pth")
DEVICE: str = os.getenv("DEVICE", "cpu")


class CatsDogsModel(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super(CatsDogsModel, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


model: tp.Optional[CatsDogsModel] = None
transform: transforms.Compose = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

def load_model() -> tp.Optional[CatsDogsModel]:
    global model
    try:
        model = CatsDogsModel(num_classes=2)
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        model.to(DEVICE)
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
    
    return model



def predict_image(image: Image.Image) -> tp.Dict[str, tp.Any]:
    if model is None:
        raise Exception("Model not loaded")

    image_tensor: torch.Tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs: torch.Tensor = model(image_tensor)
        probabilities: torch.Tensor = torch.softmax(outputs, dim=1)

    class_names: tp.List[str] = ["cat", "dog"]
    probs_dict: tp.Dict[str, float] = {
        class_names[i]: probabilities[0][i].item() for i in range(len(class_names))
    }

    prediction: str = max(probs_dict, key=probs_dict.get)

    return {"prediction": prediction, "probabilities": probs_dict}
