import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, Precision, Recall, F1Score, MetricCollection
from typing import Dict, Any

from src.models.form_classifier import FormClassifier

class SquatFormTrainer(pl.LightningModule):
    """
    PyTorch Lightning module for training the Squat Form Classifier.
    Handles model definition, loss, optimizer, metrics, and training loop.
    """
    def __init__(self, config: Dict[str, Any], class_weights: torch.Tensor = None):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        # Model
        self.model = FormClassifier(
            classifier=config.model.classifier,
            num_classes=config.model.num_classes,
            input_dim=config.model.classifier.input_dim
        )

        # Loss Function
        if class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        # Metrics
        metrics = MetricCollection({
            'accuracy': Accuracy(task="multiclass", num_classes=config.model.num_classes),
            'precision': Precision(task="multiclass", num_classes=config.model.num_classes, average='macro'),
            'recall': Recall(task="multiclass", num_classes=config.model.num_classes, average='macro'),
            'f1_score': F1Score(task="multiclass", num_classes=config.model.num_classes, average='macro'),
        })
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

        # Store confidence threshold for later evaluation (not directly used in these metrics)
        self.confidence_threshold = config.eval.confidence_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _step(self, batch: Any, batch_idx: int, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)

        if stage == 'train':
            self.train_metrics.update(preds, y)
            self.log(f'{stage}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log_dict(self.train_metrics, on_step=False, on_epoch=True, logger=True)
        elif stage == 'val':
            self.val_metrics.update(preds, y)
            self.log(f'{stage}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log_dict(self.val_metrics, on_step=False, on_epoch=True, logger=True)
        elif stage == 'test':
            self.test_metrics.update(preds, y)
            self.log(f'{stage}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log_dict(self.test_metrics, on_step=False, on_epoch=True, logger=True)

        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._step(batch, batch_idx, 'train')

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._step(batch, batch_idx, 'val')

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        optimizer_cfg = self.config.model.optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=optimizer_cfg.lr,
            weight_decay=optimizer_cfg.weight_decay
        )

        scheduler_cfg = self.config.model.scheduler
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=scheduler_cfg.mode,
            factor=scheduler_cfg.factor,
            patience=scheduler_cfg.patience,
            verbose=scheduler_cfg.verbose
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss", # Monitor validation loss for LR reduction
                "interval": "epoch",
                "frequency": 1,
            },
        }
