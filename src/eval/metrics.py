import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

def compute_classification_metrics(
    preds: np.ndarray, labels: np.ndarray, class_names: List[str]
) -> Dict[str, Any]:
    """
    Computes standard classification metrics.
    Args:
        preds (np.ndarray): Predicted class labels.
        labels (np.ndarray): True class labels.
        class_names (List[str]): List of class names.
    Returns:
        Dict[str, Any]: A dictionary containing computed metrics.
    """
    metrics = {}

    metrics["accuracy"] = accuracy_score(labels, preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    metrics["precision_macro"] = precision
    metrics["recall_macro"] = recall
    metrics["f1_macro"] = f1

    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=np.arange(len(class_names)), zero_division=0
    )
    for i, class_name in enumerate(class_names):
        metrics[f"precision_{class_name}"] = precision_per_class[i]
        metrics[f"recall_{class_name}"] = recall_per_class[i]
        metrics[f"f1_{class_name}"] = f1_per_class[i]

    metrics["classification_report"] = classification_report(
        labels, preds, target_names=class_names, zero_division=0, output_dict=True
    )

    return metrics

def plot_confusion_matrix(
    preds: np.ndarray, labels: np.ndarray, class_names: List[str], save_path: str = None
) -> plt.Figure:
    """
    Plots the confusion matrix.
    Args:
        preds (np.ndarray): Predicted class labels.
        labels (np.ndarray): True class labels.
        class_names (List[str]): List of class names.
        save_path (str, optional): Path to save the plot. Defaults to None.
    Returns:
        plt.Figure: The matplotlib figure object.
    """
    cm = confusion_matrix(labels, preds, labels=np.arange(len(class_names)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    return plt.gcf()

def generate_html_report(
    metrics: Dict[str, Any],
    class_names: List[str],
    confusion_matrix_path: str = None,
    sliced_metrics: Dict[str, Any] = None,
) -> str:
    """
    Generates an HTML report for evaluation metrics.
    Args:
        metrics (Dict[str, Any]): Dictionary of overall metrics.
        class_names (List[str]): List of class names.
        confusion_matrix_path (str, optional): Path to the confusion matrix image.
        sliced_metrics (Dict[str, Any], optional): Dictionary of metrics for sliced evaluation.
    Returns:
        str: The HTML content of the report.
    """
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Model Evaluation Report</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ font-family: sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            .metric-card {{ border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 15px; }}
            .metric-card h4 {{ color: #007bff; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            pre {{ background-color: #f8f9fa; padding: 10px; border-radius: 5px; overflow-x: auto; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="mb-4">Model Evaluation Report</h1>

            <div class="row">
                <div class="col-md-6">
                    <div class="metric-card">
                        <h4>Overall Accuracy</h4>
                        <p>{metrics.get('accuracy', 'N/A'):.4f}</p>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="metric-card">
                        <h4>Macro-averaged Metrics</h4>
                        <p>Precision: {metrics.get('precision_macro', 'N/A'):.4f}</p>
                        <p>Recall: {metrics.get('recall_macro', 'N/A'):.4f}</p>
                        <p>F1-Score: {metrics.get('f1_macro', 'N/A'):.4f}</p>
                    </div>
                </div>
            </div>

            <h2 class="mt-4">Per-Class Metrics</h2>
            <table class="table table-striped">
                <thead>
                    <tr>
                        <th>Class</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                    </tr>
                </thead>
                <tbody>
    """
    for class_name in class_names:
        html_content += f"""
                    <tr>
                        <td>{class_name}</td>
                        <td>{metrics.get(f'precision_{class_name}', 'N/A'):.4f}</td>
                        <td>{metrics.get(f'recall_{class_name}', 'N/A'):.4f}</td>
                        <td>{metrics.get(f'f1_{class_name}', 'N/A'):.4f}</td>
                    </tr>
        """
    html_content += f"""
                </tbody>
            </table>

            <h2 class="mt-4">Classification Report</h2>
            <pre>{json.dumps(metrics.get('classification_report', {}), indent=2)}</pre>
    """

    if confusion_matrix_path:
        html_content += f"""
            <h2 class="mt-4">Confusion Matrix</h2>
            <img src="{os.path.basename(confusion_matrix_path)}" alt="Confusion Matrix" class="img-fluid mb-4">
        """

    if sliced_metrics:
        html_content += f"""
            <h2 class="mt-4">Sliced Evaluation</h2>
        """
        for slice_name, slice_data in sliced_metrics.items():
            html_content += f"""
            <h3>Slice: {slice_name}</h3>
            <table class="table table-striped">
                <thead>
                    <tr>
                        <th>Category</th>
                        <th>Accuracy</th>
                        <th>Precision (Macro)</th>
                        <th>Recall (Macro)</th>
                        <th>F1-Score (Macro)</th>
                    </tr>
                </thead>
                <tbody>
            """
            for category, category_metrics in slice_data.items():
                html_content += f"""
                    <tr>
                        <td>{category}</td>
                        <td>{category_metrics.get('accuracy', 'N/A'):.4f}</td>
                        <td>{category_metrics.get('precision_macro', 'N/A'):.4f}</td>
                        <td>{category_metrics.get('recall_macro', 'N/A'):.4f}</td>
                        <td>{category_metrics.get('f1_macro', 'N/A'):.4f}</td>
                    </tr>
                """
            html_content += f"""
                </tbody>
            </table>
            """

    html_content += f"""
        </div>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """
    return html_content

import json # Import json for classification report pretty printing
