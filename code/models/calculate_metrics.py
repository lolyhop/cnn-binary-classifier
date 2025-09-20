import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import os
import typing as tp

DATA_PATH: str = os.getenv("DATA_PATH", "data/processed")
MODEL_PATH: str = os.getenv("MODEL_PATH", "models/cats_dogs_model.pth")
TENSORBOARD_LOG_DIR: str = os.getenv("TENSORBOARD_LOG_DIR", "runs/metrics")
METRICS_SAVE_PATH: str = os.getenv("METRICS_SAVE_PATH", "models")
DEVICE: str = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
NUM_WORKERS: int = int(os.getenv("NUM_WORKERS", "4"))


class CatsDogsModel(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super(CatsDogsModel, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def get_test_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_device(device_config: str) -> torch.device:
    if device_config == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_config)


def create_test_dataloader() -> DataLoader:
    test_dataset = datasets.ImageFolder(
        os.path.join(DATA_PATH, "test"), get_test_transform()
    )

    return DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )


def load_model() -> CatsDogsModel:
    model = CatsDogsModel(num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def get_predictions(
    model: CatsDogsModel, dataloader: DataLoader
) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> pd.DataFrame:
    class_names = ["cats", "dogs"]

    # Overall metrics
    overall_metrics = {
        "metric": ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"],
        "value": [
            accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred, average="weighted"),
            recall_score(y_true, y_pred, average="weighted"),
            f1_score(y_true, y_pred, average="weighted"),
        ],
        "class": ["overall"] * 4,
    }

    # Per-class metrics
    per_class_metrics = {"metric": [], "value": [], "class": []}

    for i, class_name in enumerate(class_names):
        class_mask = y_true == i
        if np.sum(class_mask) > 0:
            metrics = [
                ("precision", precision_score(y_true == i, y_pred == i)),
                ("recall", recall_score(y_true == i, y_pred == i)),
                ("f1_score", f1_score(y_true == i, y_pred == i)),
                ("support", np.sum(class_mask)),
            ]

            for metric_name, metric_value in metrics:
                per_class_metrics["metric"].append(metric_name)
                per_class_metrics["value"].append(metric_value)
                per_class_metrics["class"].append(class_name)

    # Confusion matrix as flattened metrics
    cm = confusion_matrix(y_true, y_pred)
    cm_metrics = {"metric": [], "value": [], "class": []}

    for i, true_class in enumerate(class_names):
        for j, pred_class in enumerate(class_names):
            cm_metrics["metric"].append(
                f"confusion_matrix_{true_class}_predicted_as_{pred_class}"
            )
            cm_metrics["value"].append(cm[i, j])
            cm_metrics["class"].append("confusion_matrix")

    # Combine all metrics
    all_metrics = {
        "metric": overall_metrics["metric"]
        + per_class_metrics["metric"]
        + cm_metrics["metric"],
        "value": overall_metrics["value"]
        + per_class_metrics["value"]
        + cm_metrics["value"],
        "class": overall_metrics["class"]
        + per_class_metrics["class"]
        + cm_metrics["class"],
    }

    return pd.DataFrame(all_metrics)


def save_metrics(df_metrics: pd.DataFrame) -> None:
    os.makedirs(METRICS_SAVE_PATH, exist_ok=True)

    metrics_path = os.path.join(METRICS_SAVE_PATH, "metrics.csv")
    df_metrics.to_csv(metrics_path, index=False)


def main() -> None:
    if not os.path.exists(MODEL_PATH):
        return

    os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)

    test_dataloader = create_test_dataloader()
    model = load_model()

    y_pred, y_true, y_prob = get_predictions(model, test_dataloader)
    metrics_df = calculate_metrics(y_true, y_pred, y_prob)

    save_metrics(metrics_df)


if __name__ == "__main__":
    main()
