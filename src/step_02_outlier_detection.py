from __future__ import annotations

import pandas as pd
from scipy import sparse
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def detect_outliers(X: pd.DataFrame, contamination: float, random_state: int = 42) -> pd.DataFrame:
	rarity_parts = []
	for col in X.columns:
		frequencies = X[col].value_counts(normalize=True, dropna=False)
		rarity_col = 1.0 - X[col].map(frequencies).fillna(1.0)
		rarity_parts.append(rarity_col)

	rarity_scores = pd.concat(rarity_parts, axis=1).mean(axis=1)
	rarity_threshold = rarity_scores.quantile(1 - contamination)
	rarity_flag = rarity_scores >= rarity_threshold

	encoder = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="most_frequent")),
			("onehot", OneHotEncoder(handle_unknown="ignore")),
		]
	)

	X_encoded = encoder.fit_transform(X)
	if sparse.issparse(X_encoded):
		X_encoded = X_encoded.toarray()

	isolation_forest = IsolationForest(contamination=contamination, random_state=random_state)
	isolation_flag = isolation_forest.fit_predict(X_encoded) == -1

	outlier_df = pd.DataFrame(
		{
			"rarity_score": rarity_scores,
			"rarity_flag": rarity_flag,
			"isolation_forest_flag": isolation_flag,
		},
		index=X.index,
	)
	outlier_df["combined_outlier_flag"] = outlier_df["rarity_flag"] | outlier_df["isolation_forest_flag"]
	return outlier_df

__all__ = ["detect_outliers"]
