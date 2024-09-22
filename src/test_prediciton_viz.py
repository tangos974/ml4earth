# test_prediction_viz.py

from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch

# Import necessary PyTorch Lightning components
from pytorch_lightning.loggers import TensorBoardLogger

from callbacks import PredictionVisualizer
from config import (
    BATCH_SIZE,
    DATA_ROOT,
    MEAN,
    MODEL_CONFIG,
    NUM_CLASSES,
    NUM_WORKERS,
    STD,
)
from data_loaders import get_data_loaders, setup_device

# Import your dataset and data loader functions
from dataset import TrainTransform, ValTestTransform, get_datasets
from model import SegmentationModel, load_pretrained_swin_transformer

if __name__ == "__main__":
    # Set up device
    device = setup_device()

    # Define your data transforms
    train_transform = TrainTransform()
    val_transform = ValTestTransform()

    # Get datasets
    train_dataset, val_dataset, test_dataset = get_datasets(
        data_root=DATA_ROOT,
        train_transform=train_transform,
        val_transform=val_transform,
    )

    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # Load pre-trained backbone
    model_path = MODEL_CONFIG["pretrained_model_path"]
    backbone = load_pretrained_swin_transformer(model_path, device)

    # Create feature extractor
    from torchvision.models.feature_extraction import create_feature_extractor

    return_nodes = MODEL_CONFIG["return_nodes"]
    backbone = create_feature_extractor(backbone, return_nodes=return_nodes)
    backbone.to(device)

    # Instantiate your segmentation model
    num_classes = NUM_CLASSES
    model = SegmentationModel(backbone, num_classes)
    model.to(device)
    model.eval()

    prediction_visualizer = PredictionVisualizer(
        val_dataloader=val_loader, num_samples=2
    )

    logger = TensorBoardLogger(save_dir="logs", name="test_visualizer")
    trainer = MagicMock()
    trainer.logger = logger
    trainer.current_epoch = 0

    # Modify the on_validation_epoch_end method to display the grid
    def on_validation_epoch_end(self, trainer, pl_module, device):
        val_imgs = []
        val_masks = []
        val_preds = []

        pl_module.eval()
        with torch.no_grad():
            # Get the device from the model's parameters
            device = device
            for batch in self.val_dataloader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                # Make predictions on the batch
                outputs = pl_module(images)
                preds = torch.argmax(outputs, dim=1)

                # Collect samples
                for img, mask, pred in zip(images.cpu(), masks.cpu(), preds.cpu()):
                    val_imgs.append(img)
                    val_masks.append(mask)
                    val_preds.append(pred)
                    if len(val_imgs) >= self.num_samples:
                        break
                if len(val_imgs) >= self.num_samples:
                    break  # Stop after enough samples

        # Denormalize images
        val_imgs = [img * STD[:, None, None] + MEAN[:, None, None] for img in val_imgs]
        val_imgs = [torch.clamp(img, 0, 1) for img in val_imgs]

        # Create a grid of images, masks, and predictions
        grid = self.create_grid(val_imgs, val_masks, val_preds)

        # Display the grid
        grid_np = grid.permute(1, 2, 0).numpy()
        plt.figure(figsize=(10, 10))
        plt.imshow(grid_np)
        plt.axis("off")
        plt.show()

    # Replace the method in the PredictionVisualizer instance
    import types

    prediction_visualizer.on_validation_epoch_end = types.MethodType(
        on_validation_epoch_end, prediction_visualizer
    )

    prediction_visualizer.on_validation_epoch_end(trainer, model, device)
    prediction_visualizer.on_validation_epoch_end(trainer, model, device)
