# Mushroom Classifier

Data mining project for classifying mushrooms as **edible** or **poisonous** using fully categorical features from the UCI-style mushroom dataset.

This repository includes exploratory and final-phase notebooks for:

- Supervised classification
- Unsupervised clustering
- Association rule mining

## Project Objectives

- Build and compare multiple machine learning models for mushroom class prediction.
- Analyze preprocessing strategies for categorical data.
- Benchmark model performance using common classification metrics.
- Explore clustering and pattern-discovery techniques beyond supervised learning.

## Repository Structure

```text
.
|-- README.md
|-- requirements.txt
|-- Mushroom Dataset/
|   |-- feature_value_reference.csv
|   `-- mushrooms 2.csv
|-- reports/
`-- src/
	|-- Final Phase 1.ipynb
	`-- model_notebooks/
		|-- categorical_NB_classifier.ipynb
		|-- dbscan_clustering.ipynb
		|-- decision_tree_classifier.ipynb
		|-- kmeans_clustering.ipynb
		|-- knn_classifier.ipynb
		|-- logistic_regression_classifier.ipynb
		|-- model_feature_rules.py
		|-- random_forest_classifier.ipynb
		|-- svm_classifier.ipynb
		`-- test.ipynb
```

## Models and Analyses Included

### Supervised Classification

- Categorical Naive Bayes
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

### Unsupervised Learning

- K-Means clustering
- DBSCAN clustering

### Pattern Mining

- Association rule mining comparisons (`apyori`, `pyfpgrowth`, `pyECLAT`)

### Reusable Model Utilities

- `src/model_notebooks/model_feature_rules.py` provides association-inspired lift scoring to rank useful feature columns for classification notebooks.

## Requirements

Main dependencies are listed in `requirements.txt`:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scipy`
- `scikit-learn`
- `scikit-learn-extra`
- `apyori`
- `pyfpgrowth`
- `pyECLAT`

## Setup

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies.

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to Run

1. Start Jupyter in the project root:

```bash
jupyter notebook
```

2. Open notebooks from `src/` or `src/model_notebooks/`.
3. Run cells in order from top to bottom.

## Recommended Notebook Flow

If you are new to the project, use this order:

1. Classifier notebooks in `src/model_notebooks/` (e.g., KNN, Random Forest, SVM).
2. Clustering notebooks (`kmeans_clustering.ipynb`, `dbscan_clustering.ipynb`).
3. Use `src/model_notebooks/model_feature_rules.py` in any classifier notebook when you want data-driven feature subset suggestions.

## Dataset Notes

- Primary dataset file: `Mushroom Dataset/mushrooms 2.csv`
- Value dictionary/reference: `Mushroom Dataset/feature_value_reference.csv`
- Target task: binary class label prediction (edible vs poisonous)
