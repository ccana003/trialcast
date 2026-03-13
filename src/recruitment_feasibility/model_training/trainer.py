"""Model training utilities for recruitment feasibility prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class ModelArtifacts:
    """Container for trained model objects and metadata."""

    model: Pipeline
    feature_columns: List[str]
    target_column: str


class RecruitmentModelTrainer:
    """Train interpretable baseline models for recruitment outcome prediction."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    @staticmethod
    def add_target_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Create derived recruitment targets used by downstream models."""
        out = df.copy()
        out["enrollment_probability"] = np.where(
            out["patients_contacted"].fillna(0) > 0,
            out["participants_enrolled"] / out["patients_contacted"],
            np.nan,
        )

        duration_months = (
            pd.to_datetime(out["recruitment_end_date"], errors="coerce")
            - pd.to_datetime(out["recruitment_start_date"], errors="coerce")
        ).dt.days / 30.44

        out["recruitment_duration_months"] = duration_months
        out["expected_accrual_rate"] = np.where(
            out["recruitment_duration_months"].fillna(0) > 0,
            out["participants_enrolled"] / out["recruitment_duration_months"],
            np.nan,
        )

        return out

    def build_pipeline(
        self,
        numeric_features: List[str],
        categorical_features: List[str],
    ) -> Pipeline:
        """Build a model pipeline supporting missing values and mixed types."""
        numeric_pipeline = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))],
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ],
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline, categorical_features),
            ],
        )

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            random_state=self.random_state,
        )

        return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    def train(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
        categorical_features: Optional[List[str]] = None,
    ) -> ModelArtifacts:
        """Train a baseline model and return artifacts."""
        categorical_features = categorical_features or []
        numeric_features = [col for col in feature_columns if col not in categorical_features]

        train_df = df[df[target_column].notna()].copy()
        model = self.build_pipeline(numeric_features=numeric_features, categorical_features=categorical_features)
        model.fit(train_df[feature_columns], train_df[target_column])

        return ModelArtifacts(model=model, feature_columns=feature_columns, target_column=target_column)
