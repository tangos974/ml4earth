# models.py

import time

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.ops as ops
from torchmetrics import Accuracy, F1Score, JaccardIndex, Precision, Recall
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.swin_transformer import swin_v2_b

from config import CONFIG


def freeze_backbone_layers(model, num_layers_to_freeze):
    """
    Freezes the first `num_layers_to_freeze` layers of the backbone.
    Assumes that the backbone's layers are accessible via named_modules.
    """
    layers = list(model.named_modules())
    # Exclude the top-level module
    layers = layers[1:]
    for name, layer in layers[:num_layers_to_freeze]:
        for param in layer.parameters():
            param.requires_grad = False


def load_pretrained_swin_transformer(model_path: str, device: torch.device):
    backbone = swin_v2_b()
    full_state_dict = torch.load(model_path, map_location=device)
    # Extract just the Swin backbone parameters from the full state dict.
    swin_prefix = "backbone.backbone."
    swin_state_dict = {
        k[len(swin_prefix) :]: v
        for k, v in full_state_dict.items()
        if k.startswith(swin_prefix)
    }
    backbone.load_state_dict(swin_state_dict)
    backbone.head = nn.Identity()  # Remove the classification head
    backbone.to(device)

    # Define the layers from which to extract features
    return_nodes = CONFIG["model"]["return_nodes"]
    backbone = create_feature_extractor(backbone, return_nodes=return_nodes)

    # Freeze layers after feature extractor is created
    if CONFIG["model"]["freeze_backbone"]:
        freeze_backbone_layers(
            backbone, num_layers_to_freeze=CONFIG["model"]["num_layers_to_freeze"]
        )

    return backbone


class SegmentationModel(nn.Module):
    def __init__(self, backbone, num_classes):
        super(SegmentationModel, self).__init__()
        self.backbone = backbone

        self.fpn = ops.FeaturePyramidNetwork(
            in_channels_list=CONFIG["model"]["in_channels_list"],
            out_channels=CONFIG["model"]["fpn_out_channels"],
        )
        self.seg_head = nn.Sequential(
            nn.Conv2d(
                CONFIG["model"]["fpn_out_channels"], 128, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )

    def forward(self, x):
        features = self.backbone(x)
        # Ensure feature maps have correct shape
        expected_channels = CONFIG["model"]["expected_channels"]
        for name, feature in features.items():
            # Check if feature has shape [batch_size, height, width, channels]
            if feature.shape[1] != expected_channels.get(name, -1):
                # Assume shape is [batch_size, height, width, channels], permute to [batch_size, channels, height, width]
                features[name] = feature.permute(0, 3, 1, 2).contiguous()
        fpn_out = self.fpn(features)
        x = fpn_out["stage4"]  # Use the highest-level feature map
        x = self.seg_head(x)
        # Upsample to match the input spatial dimensions
        x = nn.functional.interpolate(
            x,
            scale_factor=CONFIG["model"]["upsample_scale"],
            mode="bilinear",
            align_corners=False,
        )
        return x


class SegmentationLightningModule(pl.LightningModule):
    def __init__(
        self,
        model,
        num_classes,
        learning_rate,
        weight_decay,
        class_weights,
        label_smoothing,
    ):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

        self.criterion = torch.nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

        # Metrics
        self.train_accuracy = Accuracy(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.train_iou = JaccardIndex(task="multiclass", num_classes=num_classes)
        self.train_precision = Precision(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.train_recall = Recall(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.train_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )

        self.val_accuracy = Accuracy(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_iou = JaccardIndex(task="multiclass", num_classes=num_classes)
        self.val_precision = Precision(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_recall = Recall(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        masks = batch["mask"]

        outputs = self.forward(images)
        loss = self.criterion(outputs, masks)

        preds = torch.argmax(outputs, dim=1)
        acc = self.train_accuracy(preds, masks)
        iou = self.train_iou(preds, masks)
        precision = self.train_precision(preds, masks)
        recall = self.train_recall(preds, masks)
        f1 = self.train_f1(preds, masks)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_iou", iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train_precision", precision, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("train_recall", recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_f1", f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        masks = batch["mask"]

        start_time = time.perf_counter()
        outputs = self.forward(images)
        inference_time = time.perf_counter() - start_time

        loss = self.criterion(outputs, masks)

        preds = torch.argmax(outputs, dim=1)
        acc = self.val_accuracy(preds, masks)
        iou = self.val_iou(preds, masks)
        precision = self.val_precision(preds, masks)
        recall = self.val_recall(preds, masks)
        f1 = self.val_f1(preds, masks)

        self.log(
            "inference_time",
            inference_time,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_iou", iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val_precision", precision, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("val_recall", recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=CONFIG["hyperparameters"]["scheduler_mode"],
            factor=CONFIG["hyperparameters"]["scheduler_factor"],
            patience=CONFIG["hyperparameters"]["scheduler_patience"],
            min_lr=1e-8,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": CONFIG["hyperparameters"]["monitor"],
        }
