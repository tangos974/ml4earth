# viz.py
import logging
import os
import random
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from torch import Tensor
from torch.utils.data import Dataset

from config import MEAN, STD

classes = ("Background", "Building", "Woodland", "Water", "Road")
cmap = ListedColormap(
    [
        (0 / 255, 0 / 255, 0 / 255),  # Background
        (97 / 255, 74 / 255, 74 / 255),  # Building
        (38 / 255, 115 / 255, 0 / 255),  # Woodland
        (0 / 255, 197 / 255, 255 / 255),  # Water
        (207 / 255, 207 / 255, 207 / 255),  # Road
    ]
)


def visualize_sample(sample: dict[str, Tensor]) -> None:
    image = sample["image"]
    mask = sample["mask"]

    num_panels = 2

    fig, axs = plt.subplots(1, num_panels, figsize=(num_panels * 4, 5))

    # Handle the case when there's only one panel
    if num_panels == 1:
        axs = [axs]

    denormalized_image = image * STD[:, None, None] + MEAN[:, None, None]
    # denormalized_image = denormalized_image.clamp(
    #     0, 1
    # )  # Ensure values are within [0,1]

    image_np = denormalized_image.cpu().numpy().transpose(1, 2, 0)

    # Plot Denormalized Image
    axs[0].imshow(image_np)
    axs[0].axis("off")
    axs[0].set_title("Image (Denormalized)")

    # Plot Mask
    mask_np = mask.cpu().numpy()  # Shape: HxW
    axs[1].imshow(mask_np, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    axs[1].axis("off")
    axs[1].set_title("Mask")

    plt.tight_layout()
    plt.show()
    plt.close()


def visualize_samples(train_dataset, num_samples=3):
    indices = random.sample(range(len(train_dataset)), num_samples)
    for idx in indices:
        sample = train_dataset[idx]
        visualize_sample(sample)


def visualize_predictions(
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    output_folder: str,
    epoch: int,
    num_samples: int = 2,
    logger: logging.Logger = None,
):
    """
    Visualize the predictions of the model on a subset of the dataset.

    Args:
        model: The trained neural network model.
        dataset: A dataset containing 'image' and 'mask' fields.
        device: The device on which to run the model ('cuda' or 'cpu').
        output_folder: Folder to save the visualized images.
        epoch: Current epoch number.
        num_samples: Number of samples to visualize from the dataset (default: 2).

    Description:
        The function selects a random subset of images with non-empty masks from the dataset,
        generates predictions for the masks, and visualizes the input image,
        the ground truth mask, and the predicted mask side by side using matplotlib.
        The images are denormalized for proper visualization.

    Note:
        This function is meant for segmentation tasks, where the model predicts pixel-wise masks.
        Ensure the dataset has valid 'image' and 'mask' fields.

    Returns:
        A plot showing the original image, the ground truth mask, and the predicted mask for each sample.
    """
    model.eval()

    selected_indices = []
    attempts = 0

    # Get random non-empty samples
    while (
        len(selected_indices) <= num_samples and attempts < 25
    ):  # Limit attempts to avoid infinite loops
        idx = random.randint(0, len(dataset) - 1)
        sample = dataset[idx]
        mask = np.asarray(sample["mask"])  # Ensure mask is a NumPy array
        if np.any(mask):  # Check if there are any non-zero elements in the mask
            selected_indices.append(idx)
        attempts += 1

    if len(selected_indices) == 0:
        logger.warning(
            "Warning: Only %d samples with non-empty masks found after %d attempts.",
            len(selected_indices),
            attempts,
        )
        return

    # Ensure the "plots" directory exists
    plots_dir = os.path.join(output_folder, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    for idx in selected_indices:
        sample = dataset[idx]
        image = sample["image"].to(device).unsqueeze(0)
        mask = sample["mask"].numpy()

        with torch.no_grad():
            output = model(image)["out"]
            pred_mask = torch.argmax(output, dim=1).cpu().squeeze(0).numpy()

        # Denormalize image
        mean = np.array(MEAN)
        std = np.array(STD)
        image_np = image.cpu().squeeze(0).numpy()
        image_np = (image_np * std[:, None, None]) + mean[:, None, None]
        image_np = np.clip(image_np.transpose(1, 2, 0), 0, 1)

        # Plotting
        _, axs = plt.subplots(1, 3, figsize=(18, 6))
        axs[0].imshow(image_np)
        axs[0].set_title("Image")
        axs[0].axis("off")

        axs[1].imshow(mask, cmap="gray")
        axs[1].set_title("Ground Truth Mask")
        axs[1].axis("off")

        axs[2].imshow(pred_mask, cmap="gray")
        axs[2].set_title("Predicted Mask")
        axs[2].axis("off")

        # Save the plot
        plt.savefig(os.path.join(plots_dir, f"epoch_{epoch}_sample_{idx}.png"))

        # Close the plot
        plt.close()
