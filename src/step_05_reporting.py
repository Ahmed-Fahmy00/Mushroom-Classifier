from __future__ import annotations

import pandas as pd


def build_milestone_1_report(
	rows_count: int,
	columns_count: int,
	duplicate_rows: int,
	edible_count: int,
	poisonous_count: int,
	outlier_contamination: float,
) -> str:
	return f"""# Milestone 1 Report

## 1) Dataset Understanding
- Rows: {rows_count}
- Columns: {columns_count}
- Duplicate rows: {duplicate_rows}
- Class distribution: edible={edible_count}, poisonous={poisonous_count}

## 2) Preprocessing Applied
- Replaced '?' with missing values.
- Filled missing categorical values using most frequent value per feature.
- Added class-wise mode imputation artifact for comparison (`preprocessed_dataset_class_mode.csv`).
- Applied one-hot encoding to categorical features.
- Applied StandardScaler normalization for KNN and SVM models.
- Produced cleaned dataset table for analysis.

## 3) Outlier Detection
- Rarity-based outlier detection (top {int(outlier_contamination * 100)}% rarity scores).
- Isolation Forest outlier detection (contamination={outlier_contamination}).
- Combined outliers using union of both methods.

## 4) Visual Analysis
- Class distribution plot.
- Top feature associations with class (Cramer's V).
- Categorical association heatmap.
- Theil's U association heatmap.
- Top feature countplots by class.

## 5) Important Milestone-1 Files
- Tables in outputs/tables.
- Plots in outputs/plots.
"""


def build_final_project_report(
	original_rows: int,
	modeling_rows: int,
	feature_count: int,
	comparison_df: pd.DataFrame,
	best_model_name: str,
	best_model_metrics: dict,
	random_forest_available: bool = True,
) -> str:
	comparison_text = comparison_df.round(4).to_string(index=False)
	if random_forest_available:
		feature_importance_text = """Top encoded features from Random Forest importance are saved in:
- outputs/tables/random_forest_feature_importance.csv
- outputs/plots/random_forest_feature_importance.png"""
	else:
		feature_importance_text = """Random Forest was not selected in this run, so feature-importance plotting was skipped.
The file below explains this in the run artifacts:
- outputs/tables/random_forest_feature_importance.csv"""

	return f"""# Final Project Summary

## Project
Mushroom Classification (Edible vs Poisonous)

## Dataset and Processing
- Original rows: {original_rows}
- Rows used for modeling after outlier removal: {modeling_rows}
- Features used: {feature_count}
- One-hot encoding applied for categorical features.
- StandardScaler normalization applied for KNN and SVM.

## Models Evaluated
```text
{comparison_text}
```

## Best Model
- Best model by poisonous F1: {best_model_name}
- Accuracy: {best_model_metrics['accuracy']:.4f}
- Precision (poisonous): {best_model_metrics['precision_poisonous']:.4f}
- Recall (poisonous): {best_model_metrics['recall_poisonous']:.4f}
- F1 (poisonous): {best_model_metrics['f1_poisonous']:.4f}

## Most Prominent Factors
{feature_importance_text}

## Deliverables Checklist
- Data understanding, preprocessing, outliers, EDA: completed.
- Multiple classifiers with metrics and comparison: completed.
- Feature prominence discussion material generated: completed.
"""

__all__ = ["build_milestone_1_report", "build_final_project_report"]
