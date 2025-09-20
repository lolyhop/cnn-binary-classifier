import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torch.utils.tensorboard import SummaryWriter
import os
import typing as tp
import json
from code.models.config_loader import load_config, Config


def get_device(device_config: str) -> torch.device:
    if device_config == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_config)


class CatsDogsModel(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super(CatsDogsModel, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def get_transforms() -> tp.Dict[str, transforms.Compose]:
    return {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    }


def create_dataloaders(config: Config) -> tp.Dict[str, DataLoader]:
    data_transforms = get_transforms()
    image_datasets = {
        phase: datasets.ImageFolder(
            os.path.join(config.data.data_path, phase), data_transforms[phase]
        )
        for phase in ["train", "val", "test"]
    }

    dataloaders = {
        phase: DataLoader(
            image_datasets[phase],
            batch_size=config.data.batch_size,
            shuffle=(phase == "train"),
            num_workers=config.data.num_workers,
        )
        for phase in ["train", "val", "test"]
    }

    return dataloaders


def train_model(
    model: CatsDogsModel,
    dataloaders: tp.Dict[str, DataLoader],
    writer: SummaryWriter,
    config: Config,
    device: torch.device,
) -> tp.Tuple[CatsDogsModel, tp.Dict[str, tp.List[float]]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=config.training.step_size, gamma=config.training.gamma
    )

    model.to(device)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(config.training.num_epochs):
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss: float = 0.0
            running_corrects: int = 0

            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                step = epoch * len(dataloaders[phase]) + batch_idx
                writer.add_scalar(f"{phase}/batch_loss", loss.item(), step)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)

            writer.add_scalar(f"{phase}/epoch_loss", epoch_loss, epoch)
            writer.add_scalar(f"{phase}/epoch_accuracy", epoch_acc, epoch)

            if phase == "train":
                current_lr = scheduler.get_last_lr()[0]
                writer.add_scalar("learning_rate", current_lr, epoch)

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(float(epoch_acc))

    return model, history


def save_model_and_history(
    model: CatsDogsModel, history: tp.Dict[str, tp.List[float]], config: Config
) -> None:
    os.makedirs(config.paths.model_save_path, exist_ok=True)

    model_path = os.path.join(config.paths.model_save_path, "cats_dogs_model.pth")
    torch.save(model.state_dict(), model_path)

    history_path = os.path.join(config.paths.model_save_path, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)


def main() -> None:
    config = load_config()
    device = get_device(config.device)

    os.makedirs(config.paths.tensorboard_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=config.paths.tensorboard_log_dir)

    dataloaders = create_dataloaders(config)
    model = CatsDogsModel(num_classes=config.model.num_classes)

    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    writer.add_graph(model.to(device), dummy_input)

    hparams = {
        "batch_size": config.data.batch_size,
        "learning_rate": config.training.learning_rate,
        "num_epochs": config.training.num_epochs,
        "device": str(device),
    }
    writer.add_hparams(hparams, {})
    print(f"Starting model training with:\n{hparams}")
    trained_model, history = train_model(model, dataloaders, writer, config, device)
    save_model_and_history(trained_model, history, config)
    writer.close()


main()
