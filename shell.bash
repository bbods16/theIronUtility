    dvc add data/raw_simulated_keypoints
    dvc add data/processed/squat_form
    dvc add data/splits
    git add data/.dvcignore data/raw_simulated_keypoints.dvc data/processed/squat_form.dvc data/splits.dvc dvc.yaml
    git commit -m "feat: Implement data processing and subject-stratified split"
