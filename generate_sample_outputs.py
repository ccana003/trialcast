"""Generate sample CSV outputs for scenario validation and demos."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def build_sample_inputs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "study_id": "TC-001",
                "disease_category": "oncology",
                "eligibility_text": "Adults age 18-75, biopsy-confirmed tumor, 4 visits.",
                "visit_count": 4,
                "recruitment_staff_count": 1,
                "site_count": 1,
                "recruitment_methods_count": 1,
            },
            {
                "study_id": "TC-002",
                "disease_category": "diabetes",
                "eligibility_text": "Adults with A1c >= 7.0, 3 visits.",
                "visit_count": 3,
                "recruitment_staff_count": 2,
                "site_count": 2,
                "recruitment_methods_count": 2,
            },
            {
                "study_id": "TC-003",
                "disease_category": "hypertension",
                "eligibility_text": "Adults age 40-75 with stage 1 hypertension, 2 visits.",
                "visit_count": 2,
                "recruitment_staff_count": 3,
                "site_count": 3,
                "recruitment_methods_count": 3,
            },
        ]
    )


def build_sample_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "study_id": "TC-001",
                "scenario": "baseline",
                "staff": 1,
                "sites": 1,
                "accrual_rate": 4.2,
                "duration_months": 28.6,
                "risk": "High",
            },
            {
                "study_id": "TC-001",
                "scenario": "+1 staff",
                "staff": 2,
                "sites": 1,
                "accrual_rate": 5.0,
                "duration_months": 24.0,
                "risk": "Moderate",
            },
            {
                "study_id": "TC-001",
                "scenario": "add site",
                "staff": 1,
                "sites": 2,
                "accrual_rate": 5.3,
                "duration_months": 22.7,
                "risk": "Moderate",
            },
        ]
    )


def build_sample_scenarios() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario_name": "baseline",
                "delta_staff": 0,
                "delta_sites": 0,
                "delta_duration": 0.0,
                "improvement_flag": 0,
            },
            {
                "scenario_name": "+1 staff",
                "delta_staff": 1,
                "delta_sites": 0,
                "delta_duration": -4.6,
                "improvement_flag": 1,
            },
            {
                "scenario_name": "add site",
                "delta_staff": 0,
                "delta_sites": 1,
                "delta_duration": -5.9,
                "improvement_flag": 1,
            },
        ]
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sample_inputs = build_sample_inputs()
    sample_predictions = build_sample_predictions()
    sample_scenarios = build_sample_scenarios()

    sample_inputs.to_csv(OUTPUT_DIR / "sample_inputs.csv", index=False)
    sample_predictions.to_csv(OUTPUT_DIR / "sample_predictions.csv", index=False)
    sample_scenarios.to_csv(OUTPUT_DIR / "sample_scenarios.csv", index=False)

    print(f"Generated files in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
