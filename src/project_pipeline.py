from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from step_00_io import ensure_output_dirs
from step_01_preprocessing import preprocess_dataframe, preprocess_dataframe_class_mode
from step_02_outlier_detection import detect_outliers
from step_03_eda_analysis import build_dataset_profile, build_missing_summary, create_eda_artifacts
from step_04_modeling import (
    extract_random_forest_feature_importance,
    get_selected_model_names,
    save_all_confusion_matrices,
    save_feature_importance_plot,
    save_model_comparison_plot,
    train_selected_models,
)
from step_05_reporting import build_final_project_report, build_milestone_1_report

SEED = 42

def main(dataset_path: Path, output_path: Path, outlier_contamination: float) -> None:
    sns.set_theme(style="whitegrid")

    dirs = ensure_output_dirs(output_path)
    df = pd.read_csv(dataset_path)

    if "class" not in df.columns:
        raise ValueError("The dataset must contain a 'class' column with edible/poisonous labels.")

    target_col = "class"
    feature_cols = [col for col in df.columns if col != target_col]

    dataset_profile = build_dataset_profile(df, target_col=target_col)
    dataset_profile.to_csv(dirs["tables"] / "dataset_profile.csv", index=False)

    missing_table = build_missing_summary(df)
    missing_table.to_csv(dirs["tables"] / "missing_and_unique_summary.csv", index=False)

    processed_df = preprocess_dataframe(df, target_col=target_col)
    processed_df.to_csv(dirs["tables"] / "preprocessed_dataset.csv", index=False)

    processed_df_class_mode = preprocess_dataframe_class_mode(df, target_col=target_col)
    processed_df_class_mode.to_csv(dirs["tables"] / "preprocessed_dataset_class_mode.csv", index=False)

    outlier_df = detect_outliers(
        processed_df[feature_cols],
        contamination=outlier_contamination,
        random_state=SEED,
    )
    outlier_df.to_csv(dirs["tables"] / "outlier_flags.csv", index=True)

    outlier_summary = pd.DataFrame(
        {
            "method": ["rarity", "isolation_forest", "combined_union"],
            "outlier_count": [
                int(outlier_df["rarity_flag"].sum()),
                int(outlier_df["isolation_forest_flag"].sum()),
                int(outlier_df["combined_outlier_flag"].sum()),
            ],
        }
    )
    outlier_summary["outlier_percent"] = (outlier_summary["outlier_count"] / len(outlier_df) * 100).round(2)
    outlier_summary.to_csv(dirs["tables"] / "outlier_summary.csv", index=False)

    create_eda_artifacts(
        processed_df=processed_df,
        target_col=target_col,
        feature_cols=feature_cols,
        plots_dir=dirs["plots"],
        tables_dir=dirs["tables"],
    )

    model_df = processed_df.loc[~outlier_df["combined_outlier_flag"]].copy()
    X = model_df[feature_cols]
    y = model_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    selected_models = get_selected_model_names(random_state=SEED)
    print(f"Selected models: {selected_models}")

    comparison_df, pipelines = train_selected_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        full_X=X,
        full_y=y,
        feature_columns=feature_cols,
        model_names=selected_models,
        random_state=SEED,
    )
    comparison_df.to_csv(dirs["tables"] / "model_comparison.csv", index=False)

    save_model_comparison_plot(
        comparison_df=comparison_df,
        output_path=dirs["plots"] / "model_comparison_f1.png",
    )

    save_all_confusion_matrices(
        pipelines=pipelines,
        X_test=X_test,
        y_test=y_test,
        plots_dir=dirs["plots"],
    )

    if "RandomForest" in pipelines:
        importance_df = extract_random_forest_feature_importance(
            pipelines=pipelines,
            feature_cols=feature_cols,
        )
        importance_df.to_csv(dirs["tables"] / "random_forest_feature_importance.csv", index=False)

        save_feature_importance_plot(
            importance_df=importance_df,
            output_path=dirs["plots"] / "random_forest_feature_importance.png",
        )
    else:
        importance_df = pd.DataFrame(
            {
                "note": [
                    "RandomForest feature importance was skipped because RandomForest was not selected in this run."
                ]
            }
        )
        importance_df.to_csv(dirs["tables"] / "random_forest_feature_importance.csv", index=False)

    best_model_name = comparison_df.iloc[0]["model"]
    best_model_metrics = comparison_df.iloc[0].to_dict()

    milestone_1_report = build_milestone_1_report(
        rows_count=len(df),
        columns_count=len(df.columns),
        duplicate_rows=int(df.duplicated().sum()),
        edible_count=int((df[target_col] == "e").sum()),
        poisonous_count=int((df[target_col] == "p").sum()),
        outlier_contamination=outlier_contamination,
    )

    final_report = build_final_project_report(
        original_rows=len(df),
        modeling_rows=len(model_df),
        feature_count=len(feature_cols),
        comparison_df=comparison_df,
        best_model_name=best_model_name,
        best_model_metrics=best_model_metrics,
        random_forest_available="RandomForest" in pipelines,
    )

    (dirs["reports"] / "milestone_1_report.md").write_text(milestone_1_report, encoding="utf-8")
    (dirs["reports"] / "final_project_summary.md").write_text(final_report, encoding="utf-8")

    print("Pipeline completed.")
    print(f"Dataset used: {dataset_path}")
    print(f"Artifacts generated under: {dirs['run_root']}")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_dataset = repo_root / "Mushroom Dataset" / "mushrooms 2.csv"
    default_output = repo_root / "outputs"

    parser = argparse.ArgumentParser(description="Mushroom classification project pipeline")
    parser.add_argument("--dataset", type=Path, default=default_dataset, help="Path to dataset CSV")
    parser.add_argument("--output", type=Path, default=default_output, help="Path to output folder")
    parser.add_argument(
        "--outlier-contamination",
        type=float,
        default=0.01,
        help="Outlier contamination ratio between 0 and 0.5",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not 0.0 < args.outlier_contamination < 0.5:
        raise ValueError("--outlier-contamination must be between 0 and 0.5")
    main(
        dataset_path=args.dataset,
        output_path=args.output,
        outlier_contamination=args.outlier_contamination,
    )
