    dvc add data/raw_simulated_keypoints
    dvc add data/processed/squat_form
    dvc add data/splits
    git add data/.dvcignore data/raw_simulated_keypoints.dvc data/processed/squat_form.dvc data/splits.dvc dvc.yaml
    git commit -m "feat: Finalize data pipeline and DVC tracking"
    python scripts/train.py
        # Replace <PATH_TO_BEST_CHECKPOINT> with the actual path
    python scripts/evaluate.py checkpoint_path=<PATH_TO_BEST_CHECKPOINT>
      # Replace <PATH_TO_BEST_CHECKPOINT> with the actual path
    python scripts/export.py checkpoint_path=<PATH_TO_BEST_CHECKPOINT> export.format=onnx

