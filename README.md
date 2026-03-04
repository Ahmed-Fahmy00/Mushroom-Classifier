# mushroom-classifier-project

Mushroom classification data mining project (edible vs poisonous) using categorical features.

## Dataset Note

- This project and `project B` use the same mushroom dataset content.
- Only filename/path may differ.
- If needed, pass your file path with `--dataset`.

## Ordered Workflow (Execution Sequence)

These files are organized by execution order:

1. `src/step_00_io.py` (run folders, saving plots)
2. `src/step_01_preprocessing.py` (cleaning + imputation + encoding helpers)
3. `src/step_02_outlier_detection.py` (rarity + isolation forest)
4. `src/step_03_eda_analysis.py` (Cramer's V + Theil's U + EDA plots)
5. `src/step_04_modeling.py` (training, evaluation, confusion matrices)
6. `src/step_05_reporting.py` (milestone/final report text)
7. `src/project_pipeline.py` (orchestrates all steps)

The `step_*.py` files are the real implementation files (not wrappers), ordered by execution flow.

## Detailed Runtime Sequence

When running `src/project_pipeline.py`, the practical order is:

1. Create timestamped output folders (`step_00`).
2. Load dataset and basic profile tables (`step_03`).
3. Apply preprocessing and class-wise preprocessing artifacts (`step_01`).
4. Detect outliers and save summaries (`step_02`).
5. Generate EDA/association plots and matrices (`step_03`).
6. Split train/test data.
7. Build selected model pipelines and train/evaluate (`step_04`).
8. Save comparison plot, confusion matrices, and feature importance (`step_04`).
9. Build milestone/final reports (`step_05`).

## Models Folder

All model definitions are directly inside:

- `src/step_04_modeling.py`

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Full Pipeline

```bash
python src/project_pipeline.py
```

Optional:

```bash
python src/project_pipeline.py --dataset "Mushroom Dataset/mushrooms 2.csv" --output outputs --outlier-contamination 0.01
```

## Select One Model (Comment/Uncomment One Line)

Use one-line model selection at the top of:

- `src/step_04_modeling.py`

Pattern used:

```python
# DEFAULT_SELECTED_MODELS = ["LogisticRegression"]
# DEFAULT_SELECTED_MODELS = ["DecisionTree"]
DEFAULT_SELECTED_MODELS = ["RandomForest"]
# DEFAULT_SELECTED_MODELS = ["GradientBoosting"]
```

Only keep one active line uncommented (or use `DEFAULT_SELECTED_MODELS = ["ALL"]` to run all).

`project_pipeline.py` and the notebook read this setting directly.

## Outputs

Each run creates:

- `outputs/<timestamp>/tables`
- `outputs/<timestamp>/plots`
- `outputs/<timestamp>/reports`

Includes preprocessing tables, outlier summaries, EDA plots, model metrics, confusion matrices, and markdown reports.
