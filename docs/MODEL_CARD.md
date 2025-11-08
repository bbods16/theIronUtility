# Model Card: Iron Utility v1 (Squat)

This model card provides information about the squat form analysis model, its performance characteristics, and its intended use.

## 1. Model Details

*   **Model Name**: Iron Utility Form Analysis (v1: Squat)
*   **Version**: 1.0
*   **Type**: Pipeline model consisting of:
    1.  **Pose Estimator**: A pre-trained, lightweight CNN (MediaPipe BlazePose) that extracts 33 body keypoints from each video frame.
    2.  **Form Classifier**: A Temporal Convolutional Network (TCN) that analyzes the time-series of keypoints to classify squat form.
*   **Objective**: To provide real-time, corrective feedback on a user's squat form by analyzing their webcam video feed directly in their browser.

## 2. Intended Use

*   **Primary Use**: To be integrated into the "The Iron Utility" web application to help users improve their squat form during exercise.
*   **Target Users**: Fitness enthusiasts, from beginners to intermediate, who are working out at home without a personal trainer.
*   **Out-of-Scope Use**:
    *   This model is **not a medical device** and should not be used for diagnostic or therapeutic purposes.
    *   It is not a substitute for professional medical advice or physical therapy.
    *   It is not designed to analyze movements other than the squat.

## 3. Training Data

The form classifier was trained on a proprietary dataset of ~1,000 video clips of squats, labeled by certified personal trainers.

*   **Data Diversity**: The dataset includes 100+ volunteers with diverse body types, skin tones, and clothing.
*   **Splitting**: Data was split by subject (person) to prevent data leakage.
*   **Bias**: The dataset has known limitations and potential biases. Please refer to the `DATA_CARD.md` for a full description.

## 4. Performance Metrics

Performance is measured on a held-out test set where no subjects appeared in the training data.

*   **Primary Business Metric**: High precision on corrective feedback to avoid giving bad advice.
*   **Feedback Threshold**: The model only provides feedback if its confidence (softmax probability) for a specific error class is **> 0.90**.

| Class             | Precision | Recall | Notes                                                 |
| ----------------- | --------- | ------ | ----------------------------------------------------- |
| `knees_caving`    | > 0.95    | > 0.85 | Optimized for high precision.                         |
| `not_deep_enough` | > 0.95    | > 0.85 | Optimized for high precision.                         |
| `butt_wink`       | > 0.95    | > 0.85 | Optimized for high precision.                         |
| `spinal_flexion`  | > 0.95    | > 0.85 | Optimized for high precision, especially for this high-risk error. |
| `good_rep`        | -         | -      | The absence of a correction implies a good rep.       |

*   **Pose Model Performance**: The underlying pose estimator (BlazePose) achieves >90% PCK@0.5 on public benchmarks. Performance on our internal, diversity-focused test set is tracked separately. See the `EVALUATION.md` report for details.
*   **Latency**: The end-to-end pipeline latency (frame -> pose -> classification) is under 50ms on most modern consumer laptops, enabling real-time feedback.

## 5. Bias, Risks, and Limitations

*   **Performance Gaps**: The model's accuracy may vary across different subpopulations due to factors like skin tone, body type, clothing, and lighting. We have an active program to measure and mitigate these gaps. See `EVALUATION.md` for sliced performance metrics.
*   **False Positives**: While precision is high, the model can still make mistakes and "correct" a good squat. The high confidence threshold is designed to minimize this, but the risk is not zero.
*   **False Negatives**: The model may miss subtle form errors, especially if they are not well-represented in the training data or if the user's body is partially occluded.
*   **Disclaimer**: The application must display a clear disclaimer to the user stating that the feedback is not medical advice and that improper form can lead to injury.

## 6. Export & Deployment

*   **Format**: The model is exported to ONNX and/or TensorFlow.js format.
*   **Size**: The total model bundle size is < 10MB.
*   **Platform**: Inference is performed 100% on-device in the user's web browser using WebGL or WebGPU. **No user video data is ever sent to a server.**

This model card is a living document and will be updated with subsequent model releases.
