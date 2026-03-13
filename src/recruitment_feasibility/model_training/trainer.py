"""Model training utilities for recruitment feasibility prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class InsufficientTrainingDataError(ValueError):
    """Raised when training data does not satisfy minimum quality thresholds."""


@dataclass
class EvaluationMetrics:
    """Validation metrics for a trained model."""

    train_rows: int
    validation_rows: int
    target_min: float
    target_median: float
    target_max: float
    model_mae: float
    model_rmse: float
    model_r2: float
    baseline_mae: float
    baseline_rmse: float
    baseline_r2: float


@dataclass
class ModelArtifacts:
    """Container for trained model objects and metadata."""

    model: Pipeline
    feature_columns: List[str]
    target_column: str
    evaluation: EvaluationMetrics


class RecruitmentModelTrainer:
    """Train interpretable baseline models for recruitment outcome prediction."""

    def __init__(self, random_state: int = 42, min_training_rows: int = 8) -> None:
        self.random_state = random_state
        self.min_training_rows = min_training_rows

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

    def _validate_training_data(self, train_df: pd.DataFrame, target_column: str) -> None:
        if len(train_df) < self.min_training_rows:
            raise InsufficientTrainingDataError(
                f"Need at least {self.min_training_rows} rows with non-null {target_column}; found {len(train_df)}."
            )

        unique_target_values = train_df[target_column].nunique(dropna=True)
        if unique_target_values < 2:
            raise InsufficientTrainingDataError(
                f"Training target {target_column} must include at least 2 unique values; found {unique_target_values}."
            )

    @staticmethod
    def _compute_metrics(actual: pd.Series, predicted: np.ndarray) -> Dict[str, float]:
        return {
            "mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "r2": float(r2_score(actual, predicted)),
        }

    def _evaluate_model(
        self,
        model: Pipeline,
        X_validation: pd.DataFrame,
        y_validation: pd.Series,
        y_train: pd.Series,
    ) -> EvaluationMetrics:
        model_pred = model.predict(X_validation)
        baseline_pred = np.full(shape=len(y_validation), fill_value=float(y_train.mean()))

        model_metrics = self._compute_metrics(y_validation, model_pred)
        baseline_metrics = self._compute_metrics(y_validation, baseline_pred)

        return EvaluationMetrics(
            train_rows=len(y_train),
            validation_rows=len(y_validation),
            target_min=float(y_train.min()),
            target_median=float(y_train.median()),
            target_max=float(y_train.max()),
            model_mae=model_metrics["mae"],
            model_rmse=model_metrics["rmse"],
            model_r2=model_metrics["r2"],
            baseline_mae=baseline_metrics["mae"],
            baseline_rmse=baseline_metrics["rmse"],
            baseline_r2=baseline_metrics["r2"],
        )

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
        self._validate_training_data(train_df, target_column)

        X = train_df[feature_columns]
        y = train_df[target_column]

        X_train, X_validation, y_train, y_validation = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=self.random_state,
        )

        model = self.build_pipeline(numeric_features=numeric_features, categorical_features=categorical_features)
        model.fit(X_train, y_train)
        evaluation = self._evaluate_model(model, X_validation, y_validation, y_train)

        return ModelArtifacts(
            model=model,
            feature_columns=feature_columns,
            target_column=target_column,
            evaluation=evaluation,
        )
