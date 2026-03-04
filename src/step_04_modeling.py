from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB, CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from step_00_io import save_plot
from step_01_preprocessing import build_preprocessor


def build_model_registry(random_state: int = 42) -> dict[str, object]:
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=random_state),
        "DecisionTree": DecisionTreeClassifier(criterion="entropy", random_state=random_state),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=random_state),
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "BernoulliNB": BernoulliNB(),
        "CategoricalNB": CategoricalNB(),
        "SVM": SVC(kernel="rbf", random_state=random_state),
    }


# Selected model configuration (single source of truth).
# Keep one line uncommented.
# DEFAULT_SELECTED_MODELS = ["LogisticRegression"]
# DEFAULT_SELECTED_MODELS = ["DecisionTree"]
DEFAULT_SELECTED_MODELS = ["RandomForest"]
# DEFAULT_SELECTED_MODELS = ["GradientBoosting"]
# DEFAULT_SELECTED_MODELS = ["KNN"]
# DEFAULT_SELECTED_MODELS = ["BernoulliNB"]
# DEFAULT_SELECTED_MODELS = ["CategoricalNB"]
# DEFAULT_SELECTED_MODELS = ["SVM"]
# DEFAULT_SELECTED_MODELS = ["ALL"]  # optional: run all models


def get_selected_model_names(random_state: int = 42) -> list[str]:
    available_model_names = list(build_model_registry(random_state=random_state).keys())

    if DEFAULT_SELECTED_MODELS == ["ALL"]:
        return available_model_names

    invalid_model_names = [name for name in DEFAULT_SELECTED_MODELS if name not in available_model_names]
    if invalid_model_names:
        raise ValueError(
            f"Unknown model(s) in DEFAULT_SELECTED_MODELS: {invalid_model_names}. "
            f"Available: {available_model_names}"
        )

    return DEFAULT_SELECTED_MODELS.copy()


def build_pipeline_for_model(model_name: str, model: object, feature_columns: list[str]) -> Pipeline:
    if model_name == "CategoricalNB":
        categorical_nb_preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "encoder",
                                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                            ),
                            (
                                "non_negative",
                                FunctionTransformer(lambda values: np.where(values < 0, 0, values)),
                            ),
                        ]
                    ),
                    feature_columns,
                )
            ],
            remainder="drop",
        )
        return Pipeline(
            steps=[
                ("preprocess", categorical_nb_preprocessor),
                ("classifier", model),
            ]
        )

    steps = [("preprocess", build_preprocessor(feature_columns))]
    if model_name in {"KNN", "SVM"}:
        steps.append(("scaler", StandardScaler(with_mean=False)))
    steps.append(("classifier", model))
    return Pipeline(steps=steps)


def train_selected_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    full_X: pd.DataFrame,
    full_y: pd.Series,
    feature_columns: list[str],
    model_names: list[str],
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    all_models = build_model_registry(random_state=random_state)
    selected_models = {name: all_models[name] for name in model_names if name in all_models}

    if not selected_models:
        raise ValueError("No valid model names were provided.")

    rows = []
    fitted_pipelines: dict[str, Pipeline] = {}

    for model_name, model in selected_models.items():
        pipeline = build_pipeline_for_model(model_name=model_name, model=model, feature_columns=feature_columns)

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        cv_scores = cross_val_score(pipeline, full_X, full_y, cv=5, scoring="accuracy")

        rows.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision_poisonous": precision_score(y_test, y_pred, pos_label="p", zero_division=0),
                "recall_poisonous": recall_score(y_test, y_pred, pos_label="p", zero_division=0),
                "f1_poisonous": f1_score(y_test, y_pred, pos_label="p", zero_division=0),
                "cv_accuracy_mean": cv_scores.mean(),
                "cv_accuracy_std": cv_scores.std(),
            }
        )

        fitted_pipelines[model_name] = pipeline

    results_df = pd.DataFrame(rows).sort_values(by="f1_poisonous", ascending=False).reset_index(drop=True)
    return results_df, fitted_pipelines


def train_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    full_X: pd.DataFrame,
    full_y: pd.Series,
    feature_columns: list[str],
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    model_names = list(build_model_registry(random_state=random_state).keys())
    return train_selected_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        full_X=full_X,
        full_y=full_y,
        feature_columns=feature_columns,
        model_names=model_names,
        random_state=random_state,
    )


def save_confusion_matrix_plot(y_true: pd.Series, y_pred: np.ndarray, model_name: str, path: Path) -> None:
    labels = ["e", "p"]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    save_plot(path)


def save_all_confusion_matrices(
    pipelines: dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series, plots_dir: Path
) -> None:
    for model_name, pipeline in pipelines.items():
        y_pred = pipeline.predict(X_test)
        save_confusion_matrix_plot(
            y_true=y_test,
            y_pred=y_pred,
            model_name=model_name,
            path=plots_dir / f"confusion_matrix_{model_name.lower()}.png",
        )


def save_model_comparison_plot(comparison_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=comparison_df, x="f1_poisonous", y="model", orient="h")
    plt.title("Model Comparison by F1 (Poisonous Class)")
    plt.xlabel("F1 Score")
    plt.ylabel("Model")
    save_plot(output_path)


def extract_random_forest_feature_importance(
    pipelines: dict[str, Pipeline], feature_cols: list[str]
) -> pd.DataFrame:
    random_forest_pipeline = pipelines["RandomForest"]
    preprocessor = random_forest_pipeline.named_steps["preprocess"]
    encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    rf_model = random_forest_pipeline.named_steps["classifier"]

    feature_names = encoder.get_feature_names_out(feature_cols)
    importances = rf_model.feature_importances_

    return pd.DataFrame(
        {"encoded_feature": feature_names, "importance": importances}
    ).sort_values(by="importance", ascending=False)


def save_feature_importance_plot(importance_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df.head(15), x="importance", y="encoded_feature", orient="h")
    plt.title("Top 15 Feature Importances (Random Forest)")
    plt.xlabel("Importance")
    plt.ylabel("Encoded Feature")
    save_plot(output_path)

__all__ = [
    "build_model_registry",
    "get_selected_model_names",
    "DEFAULT_SELECTED_MODELS",
    "train_models",
    "train_selected_models",
    "save_model_comparison_plot",
    "save_all_confusion_matrices",
    "extract_random_forest_feature_importance",
    "save_feature_importance_plot",
]
