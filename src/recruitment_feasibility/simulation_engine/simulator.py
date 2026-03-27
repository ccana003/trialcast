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

    SCENARIO_CAPACITY_FIELDS = [
        "recruitment_staff_count",
        "dedicated_recruiter",
        "site_count",
        "recruitment_methods_count",
    ]

    def __init__(self, enrollment_model: ModelArtifacts, accrual_model: ModelArtifacts) -> None:
        self.enrollment_model = enrollment_model
        self.accrual_model = accrual_model

    def simulate(self, proposal_features: Dict[str, object], target_enrollment: int) -> SimulationResult:
        """Generate recruitment projections from trained models and proposal inputs."""
        feature_frame = pd.DataFrame([proposal_features])

        # --- Predict enrollment probability ---
        enrollment_prob = float(
            self.enrollment_model.model.predict(
                feature_frame[self.enrollment_model.feature_columns]
            )[0]
        )

        # --- Predict accrual rate ---
        accrual_rate = float(
            self.accrual_model.model.predict(
                feature_frame[self.accrual_model.feature_columns]
            )[0]
        )

        # Apply product-level bounds to keep outputs realistic.
        enrollment_prob = float(np.clip(enrollment_prob, 0.05, 0.95))
        accrual_rate = max(1.5, accrual_rate)

        # Domain-informed adjustments:
        # - fewer visits generally improves participation cadence and conversion;
        # - higher eligibility complexity generally slows accrual and conversion.
        visit_count = pd.to_numeric(pd.Series([proposal_features.get("visit_count", 2)]), errors="coerce").iloc[0]
        complexity = pd.to_numeric(pd.Series([proposal_features.get("eligibility_complexity", 1.0)]), errors="coerce").iloc[0]
        visit_count = 2 if pd.isna(visit_count) else float(visit_count)
        complexity = 1.0 if pd.isna(complexity) else float(complexity)

        # Fewer visits -> faster recruitment and better conversion;
        # many visits -> slower recruitment and lower conversion.
        if visit_count <= 2:
            accrual_rate += 0.5
            enrollment_prob += 0.02
        elif visit_count >= 5:
            accrual_rate -= 0.5
            enrollment_prob -= 0.02

        # Higher complexity -> slower recruitment and lower conversion.
        if complexity > 1.5:
            accrual_rate -= 0.5
            enrollment_prob -= 0.02

        # Scale effects:
        # - large national studies compete for participants and may dilute local pace;
        # - small local targets are often easier to complete.
        national_sample = pd.to_numeric(pd.Series([proposal_features.get("national_sample")]), errors="coerce").iloc[0]
        local_sample = pd.to_numeric(pd.Series([proposal_features.get("local_sample")]), errors="coerce").iloc[0]
        if not pd.isna(national_sample) and float(national_sample) >= 500:
            accrual_rate -= 0.3
        if not pd.isna(local_sample) and float(local_sample) <= 40:
            accrual_rate += 0.3

        # Operational effects from recruitment capacity features.
        recruitment_staff_count = int(pd.to_numeric(pd.Series([proposal_features.get("recruitment_staff_count", 1)]), errors="coerce").fillna(1).iloc[0])
        dedicated_recruiter = int(pd.to_numeric(pd.Series([proposal_features.get("dedicated_recruiter", 0)]), errors="coerce").fillna(0).iloc[0])
        site_count = int(pd.to_numeric(pd.Series([proposal_features.get("site_count", 1)]), errors="coerce").fillna(1).iloc[0])
        methods_count = int(pd.to_numeric(pd.Series([proposal_features.get("recruitment_methods_count", 1)]), errors="coerce").fillna(1).iloc[0])

        staff_boost = min(0.06, max(0.0, (recruitment_staff_count - 1) * 0.015))
        dedicated_boost = 0.02 if dedicated_recruiter == 1 else 0.0
        site_boost = min(0.06, max(0.0, (site_count - 1) * 0.012))
        methods_boost = min(0.05, max(0.0, (methods_count - 1) * 0.010))

        enrollment_prob += staff_boost + dedicated_boost + methods_boost
        accrual_rate += (recruitment_staff_count - 1) * 0.45
        accrual_rate += (site_count - 1) * 0.6
        accrual_rate += (methods_count - 1) * 0.35
        if dedicated_recruiter == 1:
            accrual_rate += 0.8

        # Final safety constraints required by product constraints.
        accrual_rate = max(1.5, accrual_rate)
        enrollment_prob = float(np.clip(enrollment_prob, 0.05, 0.95))

        # ============================
        # 📊 Derived metrics
        # ============================
        contacts_required = target_enrollment / enrollment_prob
        duration_months = target_enrollment / accrual_rate

        # Optional cap (prevents absurd outputs)
        duration_months = min(duration_months, 120)

        risk = self._risk_label(enrollment_prob)

        return SimulationResult(
            recruitment_risk=risk,
            predicted_enrollment_probability=enrollment_prob,
            estimated_contacts_required=contacts_required,
            expected_accrual_rate_per_month=accrual_rate,
            estimated_recruitment_duration_months=duration_months,
        )

    def simulate_scenarios(self, base_input: dict, scenarios: list[dict]) -> pd.DataFrame:
        """Simulate multiple what-if scenarios by overriding operational fields."""
        rows = []
        target_enrollment = int(base_input.get("target_enrollment", 100))

        for idx, scenario in enumerate(scenarios):
            scenario_name = str(scenario.get("scenario_name") or f"Scenario {idx + 1}")
            proposal = dict(base_input)
            for field_name in self.SCENARIO_CAPACITY_FIELDS:
                if field_name in scenario:
                    proposal[field_name] = scenario[field_name]
            result = self.simulate(proposal_features=proposal, target_enrollment=target_enrollment)
            rows.append(
                {
                    "scenario_name": scenario_name,
                    "recruitment_staff_count": int(proposal.get("recruitment_staff_count", 1)),
                    "site_count": int(proposal.get("site_count", 1)),
                    "recruitment_methods_count": int(proposal.get("recruitment_methods_count", 1)),
                    "enrollment_probability": result.predicted_enrollment_probability,
                    "predicted_accrual_rate": result.expected_accrual_rate_per_month,
                    "estimated_duration_months": result.estimated_recruitment_duration_months,
                    "contacts_required": result.estimated_contacts_required,
                    "risk_level": result.recruitment_risk,
                }
            )

        return pd.DataFrame(rows)

    def simulate_distribution(
        self,
        predicted_enrollment_probability: float,
        predicted_accrual_rate: float,
        target_enrollment: int,
        n_simulations: int = 1000,
        max_months: int = 60,
        random_state: int = 42,
    ) -> Dict[str, float]:
        """Run Monte Carlo simulations for stochastic month-by-month recruitment."""
        enrollment_prob = float(np.clip(predicted_enrollment_probability, 0.05, 0.95))
        accrual_rate = max(1.5, float(predicted_accrual_rate))

        expected_contacts = max(1.0, accrual_rate / enrollment_prob)
        rng = np.random.default_rng(seed=random_state)
        completion_months = np.zeros(n_simulations, dtype=float)

        for sim_idx in range(n_simulations):
            enrolled_total = 0
            for month in range(1, max_months + 1):
                month_contacts = max(1, int(rng.poisson(lam=expected_contacts)))
                month_enrolled = int(rng.binomial(n=month_contacts, p=enrollment_prob))
                enrolled_total += month_enrolled

                if enrolled_total >= target_enrollment:
                    completion_months[sim_idx] = float(month)
                    break
            else:
                completion_months[sim_idx] = float(max_months)

        return {
            "median_duration_months": float(np.median(completion_months)),
            "p80_duration_months": float(np.quantile(completion_months, 0.80)),
            "p90_duration_months": float(np.quantile(completion_months, 0.90)),
            "probability_within_12_months": float(np.mean(completion_months <= 12.0)),
            "probability_within_24_months": float(np.mean(completion_months <= 24.0)),
        }

    def _risk_label(self, enrollment_probability: float) -> str:
        low = self.enrollment_model.risk_threshold_low
        high = self.enrollment_model.risk_threshold_high

        if enrollment_probability <= low:
            return "High"
        if enrollment_probability <= high:
            return "Moderate"
        return "Low"

    @staticmethod
    def to_display_dict(result: SimulationResult) -> Dict[str, object]:
        """Convert simulation result dataclass to UI-ready dictionary."""
        return asdict(result)
