# Data Card: Iron Utility v1 (Squat)

This document provides a summary of the datasets used to train and evaluate the Iron Utility squat form analysis model.

## 1. Datasets

The project utilizes two types of datasets:

1.  **Public Pose Estimation Datasets (for Pre-training)**: Used to train the underlying keypoint detection model.
2.  **Proprietary Form Classification Dataset**: Used to train the sequence model that classifies squat form quality.

### 1.1. Public Pose Estimation Datasets

*   **COCO (Common Objects in Context)**
    *   **Source**: [http://cocodataset.org](http://cocodataset.org)
    *   **License**: Creative Commons Attribution 4.0 License. Permissive for commercial use.
    *   **Usage**: The `person_keypoints` subset is used to pre-train the pose estimation model on a large, diverse set of human poses.
*   **MPII Human Pose Dataset**
    *   **Source**: [http://human-pose.mpi-inf.mpg.de/](http://human-pose.mpi-inf.mpg.de/)
    *   **License**: Permissive for research and academic purposes. We will use a model pre-trained on this (like MediaPipe BlazePose), which typically has a permissive license (e.g., Apache 2.0).
    *   **Usage**: Supplements COCO for pre-training, offering a wider range of activities and poses.

### 1.2. Proprietary Form Classification Dataset

*   **Source**: Internally recorded videos.
*   **Size**: Target of ~1,000 labeled video clips.
*   **Content**: Videos of 100+ volunteers performing squats. Data was collected to be diverse across:
    *   Body types and proportions.
    *   Skin tones (effort made to align with scales like Monk Skin Tone Scale).
    *   Clothing (from athletic wear to baggy clothes).
    *   Camera angles and lighting conditions.
*   **Labels**: Each repetition within a clip is labeled by certified personal trainers for one of the following classes:
    *   `good_rep`
    *   `knees_caving`
    *   `not_deep_enough`
    *   `butt_wink` (posterior pelvic tilt)
    *   `spinal_flexion` (upper or lower back rounding)
*   **License**: Proprietary. All rights reserved by The Iron Utility.
*   **Consent**: All participating volunteers have signed consent forms allowing their anonymized data (in the form of keypoint sequences) to be used for model training. Raw video is stored securely and is not part of the primary training data pipeline.

## 2. Data Splitting and Leakage Prevention

*   **Policy**: 70% Train / 15% Validation / 15% Test.
*   **Leakage Constraint**: **Splits are stratified by subject (person).** All video clips from a single volunteer are strictly confined to one split. This is critical to ensure the model generalizes to new, unseen individuals rather than memorizing the movement patterns of people in the training set.

## 3. PII, Privacy, and Governance

*   **PII Concern**: Video of a person's body and face is considered sensitive Personally Identifiable Information (PII).
*   **Primary Mitigation**: The entire inference pipeline runs **on-device** within the user's browser. User video data is never transmitted to a server, which is the cornerstone of our privacy and compliance strategy (GDPR, HIPAA, etc.).
*   **Training Data**: The proprietary video data is stored in a secure, access-controlled environment. Only derived, anonymized keypoint sequences are used in the main training pipeline.

## 4. Known Biases and Limitations

*   **Pose Estimation Accuracy**: The underlying pose model (e.g., MediaPipe BlazePose) may have performance variations across different skin tones, lighting conditions, and clothing types. We must explicitly test for this.
*   **Body Type Diversity**: While we aim for diversity, the training data may not capture all possible human body types, which could affect model accuracy for underrepresented groups.
*   **Camera Angles**: The model may be sensitive to camera placement. The training data includes varied angles, but extreme or unusual angles may degrade performance.
*   **Occlusion**: Objects between the user and the camera (e.g., furniture, pets) can occlude key body parts and disrupt pose tracking. The model is trained with simulated occlusion but has limits.

This data card will be updated as new data is collected or new biases are discovered.
