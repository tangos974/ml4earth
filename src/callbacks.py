# callbacks.py

import random
import time

import pytorch_lightning as pl
import torch

from config import MEAN, STD


class BaseVisualizer(pl.Callback):
    """
    Base class for visualization callbacks that provides shared methods.
    """

    def mask_to_rgb(self, mask):
        """
        Convert a mask to an RGB image.

        Args:
            mask (Tensor): Mask tensor with shape [1, H, W].

        Returns:
            Tensor: RGB image tensor with shape [3, H, W].
        """
        # Define colors for each class (Customize as needed)
        colors = torch.tensor(
            [
                [30, 30, 30],  # Background - soft dark grey
                [200, 120, 120],  # Building - softer red
                [100, 180, 100],  # Woodland - softer green
                [100, 200, 255],  # Water - softer blue
                [180, 180, 180],  # Road - soft grey
            ],
            dtype=torch.uint8,
        )

        mask = mask.squeeze(0).to(torch.int64)
        mask_rgb = colors[mask].permute(2, 0, 1) / 255.0  # Normalize to [0,1]

        return mask_rgb

    def create_grid(self, images, masks, preds=None, padding=2):
        """
        Create a grid of images, masks, and optionally predictions.

        Args:
            images (list of Tensors): List of image tensors.
            masks (list of Tensors): List of mask tensors.
            preds (list of Tensors, optional): List of prediction tensors.
            padding (int): Padding size.

        Returns:
            Tensor: A grid image tensor.
        """
        num_samples = len(images)
        combined_images = []

        if preds is not None:
            # We have predictions; create image-mask-pred grids
            # Calculate the total width after concatenating img, mask, and pred
            pad_height = padding
            pad_width = images[0].size(2) * 3  # Width for img, mask, pred
            pad_tensor = torch.zeros(3, pad_height, pad_width)  # 3 for RGB channels

            for i in range(num_samples):
                img = images[i]
                mask = masks[i].unsqueeze(0)
                pred = preds[i].unsqueeze(0)

                # Convert masks and predictions to RGB images
                mask_rgb = self.mask_to_rgb(mask)
                pred_rgb = self.mask_to_rgb(pred)

                # Concatenate image, mask, and prediction horizontally
                img_mask_pred = torch.cat([img, mask_rgb, pred_rgb], dim=2)
                combined_images.append(img_mask_pred)

                # Add padding except after the last sample
                if i < num_samples - 1:
                    combined_images.append(pad_tensor)

            # Stack all samples vertically
            grid = torch.cat(combined_images, dim=1)
        else:
            # No predictions; create image-mask grids
            pad_height = padding
            pad_width = images[0].size(2) * 2  # Width for img and mask
            pad_tensor = torch.zeros(3, pad_height, pad_width)

            for i in range(num_samples):
                img = images[i]
                mask = masks[i].unsqueeze(0)

                # Convert mask to RGB image
                mask_rgb = self.mask_to_rgb(mask)

                # Concatenate image and mask horizontally
                img_mask = torch.cat([img, mask_rgb], dim=2)
                combined_images.append(img_mask)

                if i < num_samples - 1:
                    combined_images.append(pad_tensor)

            grid = torch.cat(combined_images, dim=1)

        return grid


class PredictionVisualizer(BaseVisualizer):
    """
    Callback to visualize predictions during validation.

    Args:
        dataloader (DataLoader): Validation data loader.
        num_samples (int): Number of samples to visualize per epoch.
    """

    def __init__(self, dataloader, num_samples=2):
        super().__init__()
        self.dataloader = dataloader
        self.num_samples = num_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        self.visualize_samples(trainer, pl_module, stage="Validation")

    def visualize_samples(self, trainer, pl_module, stage="Validation"):
        imgs = []
        masks = []
        preds = []

        # Collect random samples
        dataset = self.dataloader.dataset
        indices = random.sample(range(len(dataset)), self.num_samples)

        pl_module.eval()
        with torch.no_grad():
            for idx in indices:
                sample = dataset[idx]
                image = sample["image"].unsqueeze(0).to(pl_module.device)
                mask = sample["mask"].to(pl_module.device)

                # Make prediction
                output = pl_module(image)
                pred = torch.argmax(output, dim=1).squeeze(0).cpu()

                # Move data to CPU and append
                imgs.append(image.squeeze(0).cpu())
                masks.append(mask.cpu())
                preds.append(pred)

        # Denormalize images
        imgs = [img * STD[:, None, None] + MEAN[:, None, None] for img in imgs]
        imgs = [torch.clamp(img, 0, 1) for img in imgs]

        # Create a grid of images, masks, and predictions
        grid = self.create_grid(imgs, masks, preds)

        # Log the grid to TensorBoard
        trainer.logger.experiment.add_image(
            f"{stage} Predictions", grid, global_step=trainer.current_epoch
        )


class TrainingVisualizer(BaseVisualizer):
    """
    Callback to visualize augmented training samples.

    Args:
        dataloader (DataLoader): Training data loader.
        num_samples (int): Number of samples to visualize per epoch.
    """

    def __init__(self, dataloader, num_samples=2):
        super().__init__()
        self.dataloader = dataloader
        self.num_samples = num_samples

    def on_train_epoch_end(self, trainer, pl_module):
        self.visualize_samples(trainer, pl_module, stage="Training")

    def visualize_samples(self, trainer, pl_module, stage="Training"):
        imgs = []
        masks = []

        # Collect random samples
        dataset = self.dataloader.dataset
        indices = random.sample(range(len(dataset)), self.num_samples)

        for idx in indices:
            sample = dataset[idx]
            image = sample["image"]
            mask = sample["mask"]

            # Move data to CPU and append
            imgs.append(image.cpu())
            masks.append(mask.cpu())

        # Denormalize images
        imgs = [img * STD[:, None, None] + MEAN[:, None, None] for img in imgs]
        imgs = [torch.clamp(img, 0, 1) for img in imgs]

        # Create a grid of images and masks
        grid = self.create_grid(imgs, masks)

        # Log the grid to TensorBoard
        trainer.logger.experiment.add_image(
            f"{stage} Augmentations", grid, global_step=trainer.current_epoch
        )


class TimingCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.epoch_start_time = None
        self.batch_start_time = None

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.epoch_start_time
        trainer.logger.log_metrics(
            {"train/epoch_time": epoch_time}, step=trainer.current_epoch
        )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, unused=0):
        self.batch_start_time = time.time()

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, unused=0
    ):
        batch_time = time.time() - self.batch_start_time
        global_step = trainer.global_step
        trainer.logger.log_metrics({"train/batch_time": batch_time}, step=global_step)
