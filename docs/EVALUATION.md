# Evaluation Plan: Iron Utility v1 (Squat)

This document outlines the evaluation strategy for the Iron Utility squat form analysis model, covering both the pose estimation and form classification components.

## 1. Evaluation Goals

*   **Quantify Performance**: Measure the accuracy, precision, recall, and other relevant metrics for both pose estimation and form classification.
*   **Identify Biases**: Systematically assess performance across different demographic groups, lighting conditions, clothing, and camera angles to identify and mitigate biases.
*   **Ensure Robustness**: Evaluate model performance under various real-world conditions and distribution shifts.
*   **Validate Latency**: Confirm that the end-to-end pipeline meets the real-time inference requirements.
*   **Support Model Card**: Provide data and insights for the `MODEL_CARD.md`.

## 2. Metrics

### 2.1. Pose Estimation Model (Internal)

The pose estimation model's performance will be evaluated on public datasets (COCO, MPII) and a subset of our proprietary data with ground-truth keypoint annotations.

*   **Percentage of Correct Keypoints (PCK@0.5)**: The primary metric. A keypoint is considered correct if its predicted location is within 50% of the bounding box size (or torso diameter) of the ground truth.
*   **Object Keypoint Similarity (OKS)**: Used for COCO dataset evaluation, similar to IoU for bounding boxes.
*   **Mean Average Precision (mAP)**: Standard metric for object detection and pose estimation.

### 2.2. Form Classification Model

The form classification model will be evaluated on the held-out test set of proprietary data, ensuring no subject overlap with training or validation sets.

*   **Primary Business Metrics (Confidence-Thresholded)**:
    *   **Precision**: For each error class (e.g., `knees_caving`), precision will be calculated only for predictions where the model's confidence (softmax probability) is above a threshold (e.g., 0.90). This aligns with the business goal of avoiding false positives.
    *   **Recall**: For each error class, recall will be calculated for predictions above the confidence threshold.
    *   **F1-Score**: The harmonic mean of precision and recall, providing a balanced view.
*   **Confusion Matrix**: To visualize true positives, true negatives, false positives, and false negatives for all classes.
*   **ROC Curves & AUC**: To assess the model's ability to discriminate between classes across various thresholds.
*   **Calibration Curves**: To evaluate how well the predicted probabilities align with the true likelihood of an event. This is crucial given the confidence-thresholding strategy.

## 3. Test Plan

### 3.1. Standard Evaluation

*   **Held-out Test Set**: Performance will be reported on the 15% held-out test set, stratified by subject.
*   **Overall Metrics**: Report aggregate precision, recall, and F1-score for all error classes.

### 3.2. Sliced Evaluation (Bias and Robustness)

Performance will be explicitly measured and reported across various data slices to identify and address potential biases and robustness issues.

*   **Demographic Slices**:
    *   **Skin Tone**: Performance evaluated across different categories of the Monk Skin Tone Scale (or similar).
    *   **Body Type**: Performance evaluated for different body mass index (BMI) ranges or body shapes.
*   **Environmental Slices**:
    *   **Lighting Conditions**: Low-light, bright-light, backlit scenarios.
    *   **Clothing**: Baggy vs. tight clothing.
    *   **Camera Angle**: Frontal, side, slightly angled views.
    *   **Occlusion**: Scenarios with partial body occlusion.
*   **Error Type Slices**: Detailed performance for each specific squat error (e.g., `knees_caving`, `not_deep_enough`).

### 3.3. Latency Benchmarking

*   **On-Device Simulation**: Measure the end-to-end pipeline latency (frame -> pose -> classification) on various simulated consumer hardware profiles (e.g., average laptop CPU, integrated GPU).
*   **Target**: p95 latency < 50ms.

### 3.4. Qualitative Analysis

*   **Error Analysis**: Review false positives and false negatives from the test set to understand common failure modes and inform model improvements.
*   **Adversarial Examples**: Explore cases where the model performs poorly or gives counter-intuitive feedback.

## 4. Reporting

*   **HTML/Markdown Report**: Automated generation of an evaluation report summarizing all metrics, sliced performance, and key findings.
*   **MLflow/W&B Integration**: All evaluation metrics, plots, and artifacts will be logged to MLflow or Weights & Biases for experiment tracking and reproducibility.

## 5. Versioning

*   **Model Versioning**: Each model release will have a corresponding evaluation report.
*   **Data Versioning**: The specific version of the test set used for evaluation will be recorded (e.g., via DVC hashes).
