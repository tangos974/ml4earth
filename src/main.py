# main.py
import os
import sys

# Suppress kineto and PyTorch C++ logs
os.environ["KINETO_LOG_LEVEL"] = "5"  # pylint: disable=wrong-import-position
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"  # pylint: disable=wrong-import-position

import pytorch_lightning as pl
import torch
import torch.profiler
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from torchvision.models.feature_extraction import create_feature_extractor

from callbacks import PredictionVisualizer, TimingCallback, TrainingVisualizer
from config import (
    BATCH_SIZE,
    CONFIG,
    DATA_ROOT,
    MODEL_CONFIG,
    NUM_CLASSES,
    NUM_WORKERS,
    PROFILER,
    RUNS_ROOT,
    log_config,
)
from data_loaders import get_data_loaders, measure_loader_time, setup_device
from dataset import TrainTransform, ValTestTransform, get_datasets
from fix_tb_decode_bug import replace_invalid_utf8
from logger import setup_logger
from model import (
    SegmentationLightningModule,
    SegmentationModel,
    load_pretrained_swin_transformer,
)
from viz import visualize_samples

MODEL_NAME = CONFIG["model"]["name"]

# Ensure runs root exists by creating it if it doesn't
os.makedirs(RUNS_ROOT, exist_ok=True)


if __name__ == "__main__":

    if "--run_number" in sys.argv:
        # Get run number from command line
        run_number = int(sys.argv[sys.argv.index("--run_number") + 1].split("_")[-1])

    else:
        # List already present run numbers for model name
        run_names: list[str] = [
            f.split("_")[-1] for f in os.listdir(RUNS_ROOT) if f.startswith(MODEL_NAME)
        ]

        if len(run_names) == 0:
            RUN_NUMBER = 0

        else:
            RUN_NUMBER = int(max(run_names)) + 1

    # Generate run_id
    run_id = f"{MODEL_NAME}_{RUN_NUMBER}"

    # Generate this run's folder in the runs root
    os.makedirs(RUNS_ROOT / run_id, exist_ok=True)
    RUN_ROOT = RUNS_ROOT / run_id

    # Logger
    LOGGER = setup_logger(RUN_ROOT, run_id)

    # Set up TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=RUN_ROOT,
        name="tensorboard_logs",
        version=RUN_NUMBER,
    )

    # Setup profiler
    if PROFILER["enabled"]:
        profiler = PyTorchProfiler(
            dirpath=str(RUN_ROOT / "profiler_logs"),
            filename="profile",
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                (
                    torch.profiler.ProfilerActivity.CUDA
                    if torch.cuda.is_available()
                    else None
                ),
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(RUN_ROOT / "profiler_logs")
            ),
            record_shapes=True,
            with_stack=True,
            with_flops=True,
            profile_memory=True,
        )
    else:
        profiler = None

    # Set seed for reproducibility
    seed = RUN_NUMBER
    torch.manual_seed(seed)
    LOGGER.info("Seed: %d", seed)

    # Device
    device = setup_device()

    # Log run infos
    LOGGER.info("Run %d", RUN_NUMBER)
    log_config(config=CONFIG, logger=LOGGER)

    LOGGER.info("Using device: %s", device)
    if torch.cuda.is_available():
        LOGGER.debug("Device infos: %s", torch.cuda.get_device_properties(device))
        device_props = torch.cuda.get_device_properties(device)
        LOGGER.debug(
            "Compute capability: %d.%d", device_props.major, device_props.minor
        )
        LOGGER.debug(
            "Number of CUDA cores: %d", device_props.multi_processor_count * 128
        )

    # Set up data transforms
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

    # If the command is given flag --dataload_test
    if "--dataload_test" in sys.argv:
        # Test transforms
        visualize_samples(train_dataset)
        # Test data loaders
        num_samples = 3
        time_dict = measure_loader_time(train_loader, num_samples)
        LOGGER.info(
            "Time taken for single sample: %.4f", time_dict["time_single_sample"]
        )
        LOGGER.info(
            "Time taken for %d samples: %.4f",
            num_samples,
            time_dict["time_multi_sample"],
        )

        sys.exit()

    else:
        # Load pre-trained backbone
        model_path = MODEL_CONFIG["pretrained_model_path"]
        backbone = load_pretrained_swin_transformer(model_path, device)

        # Create feature extractor
        return_nodes = MODEL_CONFIG["return_nodes"]
        backbone = create_feature_extractor(backbone, return_nodes=return_nodes)
        backbone.to(device)

        # Instantiate the segmentation model
        num_classes = NUM_CLASSES
        model = SegmentationModel(backbone, num_classes)
        model.to(device)

        # Save model architecture to a file
        model_architecture_file = RUN_ROOT / "model_architecture.log"
        with open(model_architecture_file, "w", encoding="utf-8") as f:
            f.write(str(model))
        LOGGER.debug("Model architecture saved to %s", model_architecture_file)

        # Define class weights and label smoothing
        class_weights = torch.tensor(CONFIG["model"]["class_weights"]).to(device)
        label_smoothing = CONFIG["model"]["label_smoothing"]

        # Instantiate the LightningModule
        learning_rate = CONFIG["hyperparameters"]["learning_rate"]
        weight_decay = CONFIG["hyperparameters"]["weight_decay"]

        lightning_model = SegmentationLightningModule(
            model=model,
            num_classes=num_classes,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        )

        # Instantiate the progress bar
        progress_bar = RichProgressBar()

        # Callbacks for checkpointing and early stopping
        checkpoint_callback = ModelCheckpoint(
            dirpath=RUN_ROOT,
            filename="best_model",
            save_top_k=1,
            verbose=True,
            monitor=CONFIG["hyperparameters"]["monitor"],
            mode=CONFIG["hyperparameters"]["scheduler_mode"],
        )

        early_stopping_callback = EarlyStopping(
            monitor=CONFIG["hyperparameters"]["monitor"],
            patience=CONFIG["hyperparameters"]["early_stopping_patience"],
            verbose=True,
            mode=CONFIG["hyperparameters"]["scheduler_mode"],
        )

        # Callback for prediction visualization
        prediction_callback = PredictionVisualizer(
            dataloader=val_loader,
            num_samples=2,  # Set the desired number of samples
        )

        training_visualizer = TrainingVisualizer(
            dataloader=train_loader,
            num_samples=2,  # Set the desired number of samples
        )

        # Define the trainer
        trainer = pl.Trainer(
            max_epochs=CONFIG["hyperparameters"]["epochs"],
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=tb_logger,
            profiler=profiler if PROFILER["enabled"] else None,
            log_every_n_steps=20,
            callbacks=[
                checkpoint_callback,
                early_stopping_callback,
                prediction_callback,
                TimingCallback(),
                training_visualizer,
                progress_bar,
            ],
            precision=CONFIG["hyperparameters"]["precision"],
            gradient_clip_val=CONFIG["hyperparameters"]["gradient_clip_val"],
        )

    # Start training
    trainer.fit(
        lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    # When done training, fix decode problems in tensorboard/kineto - generated traces
    # see   # https://github.com/pytorch/pytorch/issues/121219

    if PROFILER["enabled"]:
        # List json files in profiler_logs dir
        json_files = [
            f for f in os.listdir(RUN_ROOT / "profiler_logs") if f.endswith(".json")
        ]

        for json_file in json_files:
            # Split filename and extension correctly even if there are multiple dots
            file_stem, file_ext = os.path.splitext(json_file)

            # Correctly replace invalid utf8 characters in the json file
            replace_invalid_utf8(
                RUN_ROOT / "profiler_logs" / json_file,
                RUN_ROOT / "profiler_logs" / f"{file_stem}-fixed{file_ext}",
            )

            # Rename old json file
            os.rename(
                RUN_ROOT / "profiler_logs" / json_file,
                RUN_ROOT / "profiler_logs" / f"{file_stem}-old.backup{file_ext}",
            )
