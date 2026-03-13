"""Model training utilities for recruitment feasibility prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
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
    selected_model_name: str
    risk_threshold_low: float
    risk_threshold_high: float


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
        model_name: str,
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

        model_map = {
            "elasticnet": ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=self.random_state, max_iter=5000),
            "gradient_boosting": GradientBoostingRegressor(random_state=self.random_state),
        }

        if model_name not in model_map:
            raise ValueError(f"Unsupported model_name: {model_name}")

        return Pipeline(steps=[("preprocessor", preprocessor), ("model", model_map[model_name])])

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

    def _cross_validate_model(self, model: Pipeline, X: pd.DataFrame, y: pd.Series) -> EvaluationMetrics:
        """Evaluate model via cross-validation and baseline comparison."""
        n_splits = min(5, len(X))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        model_predictions = cross_val_predict(model, X, y, cv=cv)
        baseline_predictions = np.zeros_like(model_predictions, dtype=float)

        for _, validation_idx in cv.split(X):
            train_idx = np.setdiff1d(np.arange(len(y)), validation_idx)
            train_mean = float(y.iloc[train_idx].mean())
            baseline_predictions[validation_idx] = train_mean

        model_metrics = self._compute_metrics(y, model_predictions)
        baseline_metrics = self._compute_metrics(y, baseline_predictions)

        return EvaluationMetrics(
            train_rows=len(y),
            validation_rows=len(y),
            target_min=float(y.min()),
            target_median=float(y.median()),
            target_max=float(y.max()),
            model_mae=model_metrics["mae"],
            model_rmse=model_metrics["rmse"],
            model_r2=model_metrics["r2"],
            baseline_mae=baseline_metrics["mae"],
            baseline_rmse=baseline_metrics["rmse"],
            baseline_r2=baseline_metrics["r2"],
        )

    @staticmethod
    def _risk_thresholds(y: pd.Series) -> tuple[float, float]:
        """Compute lower and upper risk thresholds using target quantiles."""
        return float(y.quantile(0.30)), float(y.quantile(0.70))

    def train(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
        categorical_features: Optional[List[str]] = None,
    ) -> ModelArtifacts:
        """Train baseline candidate models and select the best by MAE."""
        categorical_features = categorical_features or []
        numeric_features = [col for col in feature_columns if col not in categorical_features]

        train_df = df[df[target_column].notna()].copy()
        self._validate_training_data(train_df, target_column)

        X = train_df[feature_columns]
        y = train_df[target_column]

        candidates = {
            "elasticnet": self.build_pipeline(numeric_features, categorical_features, model_name="elasticnet"),
            "gradient_boosting": self.build_pipeline(
                numeric_features,
                categorical_features,
                model_name="gradient_boosting",
            ),
        }

        cv = KFold(n_splits=min(5, len(X)), shuffle=True, random_state=self.random_state)
        mae_scores = {
            model_name: float(-cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error").mean())
            for model_name, model in candidates.items()
        }

        selected_model_name = min(mae_scores, key=mae_scores.get)
        selected_model = candidates[selected_model_name]
        evaluation = self._cross_validate_model(selected_model, X, y)

        selected_model.fit(X, y)
        risk_low, risk_high = self._risk_thresholds(y)

        return ModelArtifacts(
            model=selected_model,
            feature_columns=feature_columns,
            target_column=target_column,
            evaluation=evaluation,
            selected_model_name=selected_model_name,
            risk_threshold_low=risk_low,
            risk_threshold_high=risk_high,
        )
