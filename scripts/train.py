import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
import torch

from src.data.datasets import SquatFormDatamodule
from src.train.trainer import SquatFormTrainer

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def train(cfg: DictConfig):
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

    # --- Logger ---
    if cfg.experiment_tracker == "mlflow":
        mlflow_logger = MLFlowLogger(
            experiment_name=cfg.project_name, # Assuming project_name is defined in config
            run_name=cfg.run_name, # Assuming run_name is defined in config
            save_dir=cfg.paths.output_dir,
            log_model=cfg.train.tracker.log_model
        )
        # Log hyperparameters to MLflow
        mlflow_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
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
        # For DDP
        strategy="ddp" if cfg.train.ddp.enabled and len(cfg.gpu_ids) > 1 else "auto",
        # Resume from checkpoint
        # resume_from_checkpoint=cfg.train.resume_from_checkpoint, # Uncomment to enable resume
    )

    # --- Training ---
    trainer.fit(model, datamodule=datamodule)

    # --- Test Best Model ---
    # Load the best model checkpoint for testing
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"Loading best model from {best_model_path} for testing.")
        # trainer.test(ckpt_path=best_model_path, datamodule=datamodule)
        # For now, we'll just log that it's done. Full evaluation will be a separate script.
    else:
        print("No best model checkpoint found.")


if __name__ == "__main__":
    train()
