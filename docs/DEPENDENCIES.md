# Project Dependencies

This document lists the core and development dependencies for The Iron Utility project, along with their versions and rationale.

## Core Dependencies (`dependencies` in `pyproject.toml`)

These libraries are essential for running the model training, evaluation, and export pipelines.

| Package                 | Version   | License     | Rationale                                                              |
| :---------------------- | :-------- | :---------- | :--------------------------------------------------------------------- |
| `torch`                 | `>=2.0.0` | BSD-3-Clause | Core deep learning framework.                                          |
| `torchvision`           | `>=0.15.0`| BSD-3-Clause | Utilities for computer vision tasks with PyTorch.                      |
| `numpy`                 | `>=1.24.0`| BSD-3-Clause | Fundamental package for numerical computing in Python.                 |
| `pandas`                | `>=2.0.0` | BSD-3-Clause | Data manipulation and analysis. Used for handling metadata and labels. |
| `scikit-learn`          | `>=1.2.0` | BSD-3-Clause | Machine learning utilities, especially for metrics and data splitting. |
| `hydra-core`            | `>=1.3.0` | MIT         | Configuration management for complex applications.                     |
| `mlflow`                | `>=2.0.0` | Apache-2.0  | Experiment tracking, model registry, and artifact management.          |
| `uv`                    | `>=0.1.0` | Apache-2.0  | Fast Python package installer and resolver.                            |
| `optuna`                | `>=3.0.0` | MIT         | Hyperparameter optimization framework.                                 |
| `hydra-optuna-sweeper`  | `>=1.2.0` | MIT         | Integrates Optuna with Hydra for hyperparameter sweeps.                |
| `onnx`                  |           | Apache-2.0  | Used for validating ONNX exported models.                              |
| `onnxruntime`           |           | MIT         | Used for verifying ONNX model inference.                               |
| `matplotlib`            |           | PSF         | Used for plotting (e.g., confusion matrix).                            |
| `seaborn`               |           | BSD-3-Clause | Statistical data visualization library (builds on matplotlib).         |

## Development Dependencies (`dev` in `pyproject.toml`)

These libraries are used for development, testing, linting, and type checking. They are not required for running the core application.

| Package       | Version   | License     | Rationale                                        |
| :------------ | :-------- | :---------- | :----------------------------------------------- |
| `pytest`      | `>=7.0.0` | MIT         | Testing framework.                               |
| `ruff`        | `>=0.1.0` | MIT         | Extremely fast Python linter and formatter.      |
| `mypy`        | `>=1.0.0` | MIT         | Static type checker for Python.                  |
| `black`       | `>=23.0.0`| MIT         | Uncompromising Python code formatter.            |
| `isort`       | `>=5.0.0` | MIT         | Python utility to sort imports alphabetically.   |

## License Hygiene

All listed dependencies are under permissive licenses (Apache-2.0, MIT, BSD-3-Clause, PSF) compatible with the project's Apache-2.0 license, avoiding restrictive licenses like GPL/AGPL.
