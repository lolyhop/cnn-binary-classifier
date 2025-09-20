import json
import os
import typing as tp
from dataclasses import dataclass


@dataclass
class DataConfig:
    data_path: str
    batch_size: int
    num_workers: int


@dataclass
class ModelConfig:
    num_classes: int
    backbone: str


@dataclass
class TrainingConfig:
    learning_rate: float
    num_epochs: int
    step_size: int
    gamma: float


@dataclass
class PathsConfig:
    model_save_path: str
    tensorboard_log_dir: str
    metrics_save_path: str


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    paths: PathsConfig
    device: str


def load_config(config_path: str = "code/models/config.json") -> Config:
    config_path = os.getenv("TRAINING_CONFIG_PATH", config_path)
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    return Config(
        data=DataConfig(**config_dict["data"]),
        model=ModelConfig(**config_dict["model"]),
        training=TrainingConfig(**config_dict["training"]),
        paths=PathsConfig(**config_dict["paths"]),
        device=config_dict["device"],
    )
