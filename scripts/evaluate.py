import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch
import os
import numpy as np
import pandas as pd
import mlflow

from src.data.datasets import SquatFormDatamodule
from src.train.trainer import SquatFormTrainer
from src.eval.metrics import compute_classification_metrics, plot_confusion_matrix, generate_html_report

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def evaluate(cfg: DictConfig):
    print("Starting model evaluation...")
    print(OmegaConf.to_yaml(cfg))

    # Set random seed for reproducibility
    pl.seed_everything(cfg.seed)

    # --- Data Module ---
    datamodule = SquatFormDatamodule(cfg)
    datamodule.setup()

    # --- Load Model ---
    # Expecting a checkpoint path to be provided in the config or as CLI arg
    if not cfg.checkpoint_path:
        raise ValueError("Please provide a 'checkpoint_path' in the config or as a CLI argument.")

    print(f"Loading model from checkpoint: {cfg.checkpoint_path}")
    model = SquatFormTrainer.load_from_checkpoint(cfg.checkpoint_path, config=cfg)
    model.eval() # Set model to evaluation mode
    model.freeze() # Freeze model parameters

    # --- Trainer for testing ---
    trainer = pl.Trainer(
        accelerator="gpu" if cfg.cuda and torch.cuda.is_available() else "cpu",
        devices=cfg.gpu_ids if cfg.cuda and torch.cuda.is_available() else 1,
        logger=False # No need for a logger here, we'll log manually to MLflow
    )

    # Run inference on the test set
    print("Running inference on the test set...")
    trainer.test(model, datamodule=datamodule)

    # Retrieve predictions, labels, and metadata from the model
    all_preds = model.all_test_preds.cpu().numpy()
    all_labels = model.all_test_labels.cpu().numpy()
    # all_logits = model.all_test_logits.cpu().numpy() # Not directly used for metrics, but available

    # Collect all metadata
    all_metadata = []
    for batch_idx, batch in enumerate(datamodule.test_dataloader()):
        # batch contains (keypoints_tensor, label_tensor, item_metadata)
        # We need to convert item_metadata (dict of lists) to list of dicts
        batch_metadata = batch[2]
        # Assuming all metadata fields are lists of same length as batch_size
        num_items_in_batch = len(batch_metadata[list(batch_metadata.keys())[0]])
        for i in range(num_items_in_batch):
            item_dict = {k: v[i] for k, v in batch_metadata.items()}
            all_metadata.append(item_dict)
    
    # Convert list of dicts to DataFrame for easier slicing
    metadata_df = pd.DataFrame(all_metadata)

    class_names = [datamodule.idx_to_class[i] for i in range(datamodule.num_classes)]

    # --- MLflow Logging Setup ---
    if cfg.experiment_tracker == "mlflow":
        # Ensure the MLflow tracking URI is set if not already
        if "MLFLOW_TRACKING_URI" not in os.environ:
            mlflow_tracking_uri = os.path.join(cfg.paths.output_dir, "mlruns")
            os.makedirs(mlflow_tracking_uri, exist_ok=True)
            os.environ["MLFLOW_TRACKING_URI"] = f"file://{mlflow_tracking_uri}"
            print(f"MLFLOW_TRACKING_URI set to: {os.environ['MLFLOW_TRACKING_URI']}")

        # Start an MLflow run for evaluation
        with mlflow.start_run(run_name=f"evaluation_{cfg.run_name}", nested=True) as run:
            mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

            # --- Compute Overall Metrics ---
            print("Computing overall metrics...")
            overall_metrics = compute_classification_metrics(all_preds, all_labels, class_names)
            
            # Log overall metrics to MLflow
            for metric_name, value in overall_metrics.items():
                if isinstance(value, (float, int)):
                    mlflow.log_metric(f"overall_{metric_name}", value)
                elif isinstance(value, dict) and metric_name == "classification_report":
                    # Log per-class metrics from classification report
                    for class_name, class_metrics in value.items():
                        if isinstance(class_metrics, dict): # Skip 'accuracy', 'macro avg', 'weighted avg'
                            for sub_metric, sub_value in class_metrics.items():
                                if isinstance(sub_value, (float, int)):
                                    mlflow.log_metric(f"overall_{class_name}_{sub_metric}", sub_value)

            # --- Plot Confusion Matrix ---
            print("Plotting confusion matrix...")
            cm_save_path = os.path.join(cfg.paths.output_dir, "confusion_matrix.png")
            plot_confusion_matrix(all_preds, all_labels, class_names, save_path=cm_save_path)
            mlflow.log_artifact(cm_save_path)

            # --- Sliced Evaluation ---
            sliced_metrics_results = {}
            if cfg.eval.sliced_evaluation.enabled and not metadata_df.empty:
                print("Performing sliced evaluation...")
                for slice_config in cfg.eval.sliced_evaluation.slices:
                    slice_attribute = slice_config.attribute
                    if slice_attribute in metadata_df.columns:
                        sliced_metrics_results[slice_attribute] = {}
                        unique_categories = metadata_df[slice_attribute].unique()
                        for category in unique_categories:
                            print(f"  Evaluating slice: {slice_attribute}={category}")
                            slice_indices = metadata_df[metadata_df[slice_attribute] == category].index.values
                            
                            if len(slice_indices) > 0:
                                slice_preds = all_preds[slice_indices]
                                slice_labels = all_labels[slice_indices]
                                
                                slice_metrics = compute_classification_metrics(slice_preds, slice_labels, class_names)
                                sliced_metrics_results[slice_attribute][category] = slice_metrics

                                # Log sliced metrics to MLflow
                                for metric_name, value in slice_metrics.items():
                                    if isinstance(value, (float, int)):
                                        mlflow.log_metric(f"slice/{slice_attribute}/{category}/{metric_name}", value)
                            else:
                                print(f"    No data for slice: {slice_attribute}={category}")
                    else:
                        print(f"Warning: Slice attribute '{slice_attribute}' not found in metadata.")

            # --- Generate HTML Report ---
            print("Generating HTML report...")
            html_report_path = os.path.join(cfg.paths.output_dir, f"evaluation_report.{cfg.eval.report.format}")
            html_content = generate_html_report(
                overall_metrics,
                class_names,
                confusion_matrix_path=cm_save_path,
                sliced_metrics=sliced_metrics_results
            )
            with open(html_report_path, "w") as f:
                f.write(html_content)
            mlflow.log_artifact(html_report_path)

            print(f"Evaluation report saved to {html_report_path}")
            print("Model evaluation complete and results logged to MLflow.")
    else:
        print("MLflow not enabled. Evaluation results will not be logged.")
        # If MLflow is not enabled, still compute and print metrics, and save report
        overall_metrics = compute_classification_metrics(all_preds, all_labels, class_names)
        print("\n--- Overall Metrics ---")
        print(overall_metrics)

        cm_save_path = os.path.join(cfg.paths.output_dir, "confusion_matrix.png")
        plot_confusion_matrix(all_preds, all_labels, class_names, save_path=cm_save_path)

        sliced_metrics_results = {}
        if cfg.eval.sliced_evaluation.enabled and not metadata_df.empty:
            for slice_config in cfg.eval.sliced_evaluation.slices:
                slice_attribute = slice_config.attribute
                if slice_attribute in metadata_df.columns:
                    sliced_metrics_results[slice_attribute] = {}
                    unique_categories = metadata_df[slice_attribute].unique()
                    for category in unique_categories:
                        slice_indices = metadata_df[metadata_df[slice_attribute] == category].index.values
                        if len(slice_indices) > 0:
                            slice_preds = all_preds[slice_indices]
                            slice_labels = all_labels[slice_indices]
                            slice_metrics = compute_classification_metrics(slice_preds, slice_labels, class_names)
                            sliced_metrics_results[slice_attribute][category] = slice_metrics
        
        html_report_path = os.path.join(cfg.paths.output_dir, f"evaluation_report.{cfg.eval.report.format}")
        html_content = generate_html_report(
            overall_metrics,
            class_names,
            confusion_matrix_path=cm_save_path,
            sliced_metrics=sliced_metrics_results
        )
        with open(html_report_path, "w") as f:
            f.write(html_content)
        print(f"Evaluation report saved to {html_report_path}")


if __name__ == "__main__":
    evaluate()
