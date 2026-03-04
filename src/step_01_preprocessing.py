from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def preprocess_dataframe(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    processed = df.replace("?", np.nan).copy()
    for col in processed.columns:
        if col == target_col:
            continue
        mode_values = processed[col].mode(dropna=True)
        fill_value = mode_values.iloc[0] if not mode_values.empty else "missing"
        processed[col] = processed[col].fillna(fill_value)
    return processed


def preprocess_dataframe_class_mode(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    processed = df.replace("?", np.nan).copy()

    for col in processed.columns:
        if col == target_col:
            continue

        for class_label in processed[target_col].dropna().unique():
            class_mask = processed[target_col] == class_label
            missing_mask = processed[col].isna()
            mode_values = processed.loc[class_mask, col].mode(dropna=True)

            if not mode_values.empty:
                processed.loc[class_mask & missing_mask, col] = mode_values.iloc[0]

        global_mode = processed[col].mode(dropna=True)
        global_fill = global_mode.iloc[0] if not global_mode.empty else "missing"
        processed[col] = processed[col].fillna(global_fill)

    return processed


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[("cat", categorical_pipeline, feature_columns)],
        remainder="drop",
    )

__all__ = [
    "preprocess_dataframe",
    "preprocess_dataframe_class_mode",
    "build_preprocessor",
]
