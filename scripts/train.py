import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger
import torch
import os
from pathlib import Path

from src.data.datasets import SquatFormDatamodule
from src.train.trainer import SquatFormTrainer

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:
    # Print configuration
    print(OmegaConf.to_yaml(cfg))

    # Set random seed for reproducibility
    pl.seed_everything(cfg.seed)

    # --- Data Module ---
    datamodule = SquatFormDatamodule(cfg)
    datamodule.setup()

    # Get class weights from datamodule for loss function
    class_weights = datamodule.class_weights

    # --- Model ---
    model = SquatFormTrainer(cfg, class_weights=class_weights)

    # --- Callbacks ---
    callbacks = []

    # Early Stopping
    if cfg.train.early_stopping.enabled:
        early_stopping_callback = EarlyStopping(
            monitor=cfg.train.early_stopping.monitor,
            patience=cfg.train.early_stopping.patience,
            mode=cfg.train.early_stopping.mode,
            verbose=True
        )
        callbacks.append(early_stopping_callback)

    # Model Checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.paths.checkpoint_dir,
        filename=cfg.train.checkpointing.filename,
        monitor=cfg.train.checkpointing.monitor,
        mode=cfg.train.checkpointing.mode,
        save_top_k=1, # Save only the best model
        save_last=True, # Save the last model
        verbose=True
    )
    callbacks.append(checkpoint_callback)

    # Learning Rate Monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)

    # --- Logger ---
    if cfg.experiment_tracker == "mlflow":
        # Use the original working directory (before Hydra changes it)
        original_cwd = hydra.utils.get_original_cwd()
        mlflow_tracking_dir = Path(original_cwd) / "mlruns"
        mlflow_tracking_dir.mkdir(exist_ok=True)

        # Convert to URI using pathlib's as_uri() method (works cross-platform)
        mlflow_tracking_uri = mlflow_tracking_dir.as_uri()

        mlflow_logger = MLFlowLogger(
            experiment_name=cfg.project_name,
            run_name=cfg.run_name,
            tracking_uri=mlflow_tracking_uri,
            log_model=cfg.train.tracker.log_model,
        )
        print(f"MLflow tracking URI: {mlflow_tracking_uri}")
    else:
        mlflow_logger = None # Or configure other loggers like WandbLogger

    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="gpu" if cfg.cuda and torch.cuda.is_available() else "cpu",
        devices=cfg.gpu_ids if cfg.cuda and torch.cuda.is_available() else 1,
        callbacks=callbacks,
        logger=mlflow_logger,
        log_every_n_steps=cfg.log_interval,
        precision=cfg.model.mixed_precision.dtype if cfg.model.mixed_precision.enabled else 32,
        gradient_clip_val=cfg.train.gradient_clipping.clip_value if cfg.train.gradient_clipping.enabled else 0,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches, # Integrate gradient accumulation
        # For DDP
        strategy="ddp" if cfg.train.ddp.enabled and len(cfg.gpu_ids) > 1 else "auto",
        # Resume from checkpoint
        # resume_from_checkpoint=cfg.train.resume_from_checkpoint, # Uncomment to enable resume
    )

    # --- Training ---
    print("Starting model training...")
    trainer.fit(model, datamodule=datamodule)
    print("Model training complete.")

    # --- Test Best Model ---
    # Load the best model checkpoint for testing
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"Testing best model from {best_model_path}...")
        trainer.test(ckpt_path=best_model_path, datamodule=datamodule)
        print("Testing complete.")
    else:
        print("No best model checkpoint found for testing.")


if __name__ == "__main__":
    train()
