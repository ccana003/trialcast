"""Simulation engine for translating model predictions into planning metrics."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict

import numpy as np
import pandas as pd

from recruitment_feasibility.common.schemas import SimulationResult
from recruitment_feasibility.model_training.trainer import ModelArtifacts


class RecruitmentSimulator:
    """Run recruitment simulations for a proposed study."""

    def __init__(self, enrollment_model: ModelArtifacts, accrual_model: ModelArtifacts) -> None:
        self.enrollment_model = enrollment_model
        self.accrual_model = accrual_model

    def simulate(self, proposal_features: Dict[str, object], target_enrollment: int) -> SimulationResult:
        """Generate recruitment projections from trained models and proposal inputs."""
        feature_frame = pd.DataFrame([proposal_features])

        enrollment_prob = float(
            self.enrollment_model.model.predict(feature_frame[self.enrollment_model.feature_columns])[0]
        )
        enrollment_prob = float(np.clip(enrollment_prob, 0.001, 0.95))

        accrual_rate = float(self.accrual_model.model.predict(feature_frame[self.accrual_model.feature_columns])[0])
        accrual_rate = max(0.1, accrual_rate)

        contacts_required = target_enrollment / enrollment_prob
        duration_months = target_enrollment / accrual_rate

        risk = self._risk_label(enrollment_prob)

        return SimulationResult(
            recruitment_risk=risk,
            predicted_enrollment_probability=enrollment_prob,
            estimated_contacts_required=contacts_required,
            expected_accrual_rate_per_month=accrual_rate,
            estimated_recruitment_duration_months=duration_months,
        )

    @staticmethod
    def _risk_label(enrollment_probability: float) -> str:
        if enrollment_probability < 0.02:
            return "High"
        if enrollment_probability < 0.05:
            return "Moderate"
        return "Low"

    @staticmethod
    def to_display_dict(result: SimulationResult) -> Dict[str, object]:
        """Convert simulation result dataclass to UI-ready dictionary."""
        return asdict(result)
