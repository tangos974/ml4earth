# data_loaders.py
import time

import torch
from torch.utils.data import DataLoader, Dataset


def get_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
    num_workers: int,
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    return train_loader, val_loader, test_loader


def setup_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def measure_loader_time(loader: DataLoader, num_samples: int) -> dict:

    start_time = time.time()
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_samples:
            break
    time1 = time.time() - start_time

    start_time = time.time()
    sample = next(iter(loader))
    time2 = time.time() - start_time

    return {"time_multi_sample": time1, "time_single_sample": time2}
