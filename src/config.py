# config.py
""" Configuration file. """
import logging
import pathlib
from typing import Literal  # pylint: disable=unused-import

import torch
import yaml
from torch import Tensor  # pylint: disable=unused-import


def load_config(config_file_path: str = "config.yaml"):
    """Load configuration from a YAML file."""
    with open(config_file_path, "r", encoding="utf-8") as file:
        config_loaded = yaml.safe_load(file)
    return config_loaded


def log_config(config: dict, logger: logging.Logger):
    """Log the configuration."""
    for key, value in config.items():
        logger.info(f"{key}: {value}")


# Load config with path given by main
CONFIG = load_config()

# Paths
DATA_ROOT: pathlib.Path = pathlib.Path(CONFIG["paths"]["data_root"])
RUNS_ROOT: pathlib.Path = pathlib.Path(CONFIG["paths"]["runs_root"])
TENSORBOARD_WRITER_FOLDER: str = CONFIG["paths"]["tensorboard_writer_folder"]

# Preprocessing
IMAGE_SIZE: tuple[int] = tuple(CONFIG["preprocessing"]["image_size"])
MEAN = torch.tensor(CONFIG["preprocessing"]["mean"])
STD = torch.tensor(CONFIG["preprocessing"]["std"])
P_HORIZONTAL_FLIP: float = CONFIG["preprocessing"]["p_horizontal_flip"]
ROTATION_DEGREES: int = CONFIG["preprocessing"]["rotation_degrees"]
COLOR_JITTER: float = CONFIG["preprocessing"]["color_jitter"]
SCALE: tuple[float] = tuple(CONFIG["preprocessing"]["scale"])
RATIO: tuple[float] = tuple(CONFIG["preprocessing"]["ratio"])
CROP_PROB: float = CONFIG["preprocessing"]["crop_prob"]

# Hyperparameters
HYPERPARAMETERS = CONFIG["hyperparameters"]

BATCH_SIZE: int = CONFIG["hyperparameters"]["batch_size"]
NUM_CLASSES: int = CONFIG["hyperparameters"]["num_classes"]
NUM_WORKERS: int = CONFIG["hyperparameters"]["num_workers"]
EPOCHS: int = CONFIG["hyperparameters"]["epochs"]
LEARNING_RATE: float = CONFIG["hyperparameters"]["learning_rate"]
WEIGHT_DECAY: float = CONFIG["hyperparameters"]["weight_decay"]

# Model-specific configurations
MODEL_CONFIG = CONFIG["model"]

# Profiler
PROFILER = CONFIG["profiler"]

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.set_float32_matmul_precision("medium")
