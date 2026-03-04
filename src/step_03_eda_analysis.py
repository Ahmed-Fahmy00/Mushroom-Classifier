from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

from step_00_io import save_plot


def cramers_v(contingency_table: pd.DataFrame) -> float:
    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        return 0.0

    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.to_numpy().sum()
    if n <= 1:
        return 0.0

    phi2 = chi2 / n
    r, k = contingency_table.shape

    phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)

    denominator = min((k_corr - 1), (r_corr - 1))
    if denominator <= 0:
        return 0.0

    return float(np.sqrt(phi2_corr / denominator))


def _entropy_from_counts(counts: np.ndarray) -> float:
    probabilities = counts / counts.sum()
    probabilities = probabilities[probabilities > 0]
    if probabilities.size == 0:
        return 0.0
    return float(-(probabilities * np.log2(probabilities)).sum())


def theils_u(x: pd.Series, y: pd.Series) -> float:
    contingency = pd.crosstab(x, y)
    if contingency.empty:
        return 0.0

    x_counts = contingency.sum(axis=1).to_numpy(dtype=float)
    total = contingency.to_numpy(dtype=float).sum()
    if total <= 0:
        return 0.0

    hx = _entropy_from_counts(x_counts)
    if hx == 0:
        return 1.0

    conditional_entropy = 0.0
    for y_value in contingency.columns:
        joint_counts = contingency[y_value].to_numpy(dtype=float)
        y_total = joint_counts.sum()
        if y_total <= 0:
            continue
        conditional_entropy += (y_total / total) * _entropy_from_counts(joint_counts)

    value = (hx - conditional_entropy) / hx
    return float(max(0.0, min(1.0, value)))


def build_dataset_profile(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": [
                "rows",
                "columns",
                "duplicate_rows",
                "class_edible_count",
                "class_poisonous_count",
            ],
            "value": [
                len(df),
                len(df.columns),
                int(df.duplicated().sum()),
                int((df[target_col] == "e").sum()),
                int((df[target_col] == "p").sum()),
            ],
        }
    )


def build_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    replaced_df = df.replace("?", np.nan)
    return pd.DataFrame(
        {
            "column": df.columns,
            "missing_count_after_question_mark_replace": replaced_df.isna().sum().values,
            "unknown_question_mark_count": (df == "?").sum().values,
            "unique_values": [df[c].nunique(dropna=False) for c in df.columns],
        }
    )


def compute_feature_class_association(
    processed_df: pd.DataFrame, target_col: str, feature_cols: list[str]
) -> pd.DataFrame:
    class_associations = []
    for col in feature_cols:
        table = pd.crosstab(processed_df[col], processed_df[target_col])
        class_associations.append({"feature": col, "cramers_v_with_class": cramers_v(table)})

    return pd.DataFrame(class_associations).sort_values(by="cramers_v_with_class", ascending=False)


def compute_association_matrix(processed_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    assoc_matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)
    for col_a in columns:
        for col_b in columns:
            crosstab = pd.crosstab(processed_df[col_a], processed_df[col_b])
            assoc_matrix.loc[col_a, col_b] = cramers_v(crosstab)
    return assoc_matrix


def compute_theils_u_matrix(processed_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    theils_matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)
    for col_a in columns:
        for col_b in columns:
            theils_matrix.loc[col_a, col_b] = theils_u(processed_df[col_a], processed_df[col_b])
    return theils_matrix


def create_eda_artifacts(
    processed_df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    plots_dir: Path,
    tables_dir: Path,
) -> pd.DataFrame:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=processed_df, x=target_col, order=["e", "p"])
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    save_plot(plots_dir / "class_distribution.png")

    assoc_df = compute_feature_class_association(processed_df, target_col, feature_cols)
    assoc_df.to_csv(tables_dir / "feature_class_association.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=assoc_df.head(10), x="cramers_v_with_class", y="feature", orient="h")
    plt.title("Top Features Associated with Class (Cramer's V)")
    plt.xlabel("Cramer's V")
    plt.ylabel("Feature")
    save_plot(plots_dir / "top_feature_associations.png")

    top_features = assoc_df["feature"].head(4).tolist()
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, axis in enumerate(axes.flatten()):
        if idx < len(top_features):
            feature = top_features[idx]
            sns.countplot(data=processed_df, x=feature, hue=target_col, ax=axis)
            axis.set_title(f"{feature} by Class")
            axis.tick_params(axis="x", rotation=45)
        else:
            axis.axis("off")
    save_plot(plots_dir / "top_feature_countplots.png")

    all_columns = [target_col] + feature_cols
    assoc_matrix = compute_association_matrix(processed_df, all_columns)
    assoc_matrix.to_csv(tables_dir / "categorical_association_matrix.csv")

    plt.figure(figsize=(14, 12))
    sns.heatmap(assoc_matrix, cmap="viridis", linewidths=0.2)
    plt.title("Categorical Association Heatmap (Cramer's V)")
    save_plot(plots_dir / "categorical_association_heatmap.png")

    theils_matrix = compute_theils_u_matrix(processed_df, all_columns)
    theils_matrix.to_csv(tables_dir / "theils_u_association_matrix.csv")

    plt.figure(figsize=(14, 12))
    sns.heatmap(theils_matrix, cmap="coolwarm", linewidths=0.2)
    plt.title("Categorical Association Heatmap (Theil's U)")
    save_plot(plots_dir / "theils_u_heatmap.png")

    return assoc_df

__all__ = [
    "build_dataset_profile",
    "build_missing_summary",
    "create_eda_artifacts",
    "compute_association_matrix",
    "compute_feature_class_association",
    "compute_theils_u_matrix",
]
