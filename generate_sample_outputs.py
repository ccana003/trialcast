from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Make src importable when running from repo root
CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from recruitment_feasibility.data_ingestion.loader import DataIngestionService
from recruitment_feasibility.model_training.trainer import (
    InsufficientTrainingDataError,
    RecruitmentModelTrainer,
)
from recruitment_feasibility.simulation_engine.simulator import RecruitmentSimulator

DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # =========================
    # Load and merge all sources
    # =========================
    ingestion = DataIngestionService(DATA_DIR)
    trainer = RecruitmentModelTrainer()

    sources = ingestion.load_all_sources()
    print("Available sources:", sources.keys())

    merged = ingestion.build_merged_training_frame(sources)
    training_df = merged.copy()
    training_df = training_df.loc[:, ~training_df.columns.duplicated()]

    print("Merged training columns:")
    print(training_df.columns.tolist())

    # =========================
    # Add CTMS-informed targets
    # =========================

    # Ensure numeric
    training_df["ctms_accrued"] = pd.to_numeric(training_df["ctms_accrued"], errors="coerce")
    training_df["local_sample"] = pd.to_numeric(training_df["local_sample"], errors="coerce")

    # Build target safely
    training_df["enrollment_probability"] = (
        training_df["ctms_accrued"] /
        training_df["local_sample"].replace(0, pd.NA)
    )

    # Clean target
    training_df["enrollment_probability"] = training_df["enrollment_probability"].clip(0, 1)

    # Filter bad rows
    training_df = training_df[
        training_df["local_sample"] > 0
    ]
    print("\nEnrollment probability stats:")
    print(training_df["enrollment_probability"].describe())
    print(training_df["enrollment_probability"].value_counts())

    # =========================
    # Add accrual rate (Fix #3)
    # =========================

    # Convert dates
    training_df["recruitment_start_date"] = pd.to_datetime(
        training_df["recruitment_start_date"], errors="coerce"
    )
    training_df["recruitment_end_date"] = pd.to_datetime(
        training_df["recruitment_end_date"], errors="coerce"
    )

    # Calculate duration (months)
    training_df["recruitment_duration_months"] = (
        (training_df["recruitment_end_date"] - training_df["recruitment_start_date"]).dt.days / 30
    )

    # Clean duration (avoid 0 / negative / crazy)
    training_df["recruitment_duration_months"] = training_df["recruitment_duration_months"].clip(lower=0.5)

    # Calculate accrual rate
    training_df["expected_accrual_rate"] = (
        training_df["ctms_accrued"] /
        training_df["recruitment_duration_months"].replace(0, pd.NA)
    )

    # Clean accrual rate (remove extreme values)
    training_df["expected_accrual_rate"] = training_df["expected_accrual_rate"].clip(0, 50)

    print("\nAccrual rate stats:")
    print(training_df["expected_accrual_rate"].describe())

    # =========================
    # Filter rows for enrollment model ONLY
    # =========================
    enrollment_training_df = training_df[
        training_df["enrollment_probability"].notna()
    ]
    # =========================
    # Train models using same setup as Streamlit app
    # =========================
    feature_columns = [
        "study_type",
        "investigator_experience",
        "reviewer_concern_recruitment",
        "visit_count",
        "eligibility_complexity",
        "disease_category",
        "healthy_volunteer_flag",
        "has_feasibility_data",
        "has_recruitment_data",
        "has_protocol_data",
        "local_sample",
        "national_sample",
        "recruitment_staff_count",
        "dedicated_recruiter",
        "site_count",
        "recruitment_methods_count",
        "interest_rate",
    ]

    categorical_features = [
        "study_type",
        "disease_category",
        "reviewer_concern_recruitment",
    ]

    enrollment_model = trainer.train(
        enrollment_training_df,
        feature_columns=feature_columns,
        target_column="enrollment_probability",
        categorical_features=categorical_features,
    )

    accrual_model = trainer.train(
        training_df,
        feature_columns=feature_columns,
        target_column="expected_accrual_rate",
        categorical_features=categorical_features,
    )

    simulator = RecruitmentSimulator(
        enrollment_model=enrollment_model,
        accrual_model=accrual_model,
    )

    # =========================
    # Simulate each historical study
    # =========================
    results: list[dict] = []

    for _, row in training_df.iterrows():
        try:
            local_sample = pd.to_numeric(
                pd.Series([row.get("local_sample")]),
                errors="coerce",
            ).iloc[0]

            participants_enrolled = pd.to_numeric(
                pd.Series([row.get("participants_enrolled")]),
                errors="coerce",
            ).iloc[0]

            target_enrollment = (
                int(local_sample)
                if pd.notna(local_sample) and local_sample > 0
                else int(participants_enrolled)
                if pd.notna(participants_enrolled) and participants_enrolled > 0
                else 100
            )

            sim_result = simulator.simulate(
                proposal_features=row.to_dict(),
                target_enrollment=target_enrollment,
            )

            result_dict = simulator.to_display_dict(sim_result)
            result_dict["study_id"] = row.get("study_id", "unknown")
            results.append(result_dict)

        except Exception as exc:
            study_id = row.get("study_id", "unknown")
            print(f"Skipping {study_id} due to error: {exc}")

    predictions_df = pd.DataFrame(results)

    # =========================
    # Save outputs
    # =========================
    predictions_path = OUTPUT_DIR / "real_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    print(f"\n✅ Real predictions generated in: {predictions_path}")
    print(f"Rows processed: {len(predictions_df)}")

    # Optional: save model diagnostics too
    diagnostics = pd.DataFrame(
        [
            {
                "model_type": "enrollment",
                "selected_model_name": enrollment_model.selected_model_name,
                "target_column": enrollment_model.target_column,
                "train_rows": enrollment_model.evaluation.train_rows,
                "target_min": enrollment_model.evaluation.target_min,
                "target_median": enrollment_model.evaluation.target_median,
                "target_max": enrollment_model.evaluation.target_max,
                "model_mae": enrollment_model.evaluation.model_mae,
                "model_rmse": enrollment_model.evaluation.model_rmse,
                "model_r2": enrollment_model.evaluation.model_r2,
                "baseline_mae": enrollment_model.evaluation.baseline_mae,
                "baseline_rmse": enrollment_model.evaluation.baseline_rmse,
                "baseline_r2": enrollment_model.evaluation.baseline_r2,
                "risk_threshold_low": enrollment_model.risk_threshold_low,
                "risk_threshold_high": enrollment_model.risk_threshold_high,
            },
            {
                "model_type": "accrual",
                "selected_model_name": accrual_model.selected_model_name,
                "target_column": accrual_model.target_column,
                "train_rows": accrual_model.evaluation.train_rows,
                "target_min": accrual_model.evaluation.target_min,
                "target_median": accrual_model.evaluation.target_median,
                "target_max": accrual_model.evaluation.target_max,
                "model_mae": accrual_model.evaluation.model_mae,
                "model_rmse": accrual_model.evaluation.model_rmse,
                "model_r2": accrual_model.evaluation.model_r2,
                "baseline_mae": accrual_model.evaluation.baseline_mae,
                "baseline_rmse": accrual_model.evaluation.baseline_rmse,
                "baseline_r2": accrual_model.evaluation.baseline_r2,
                "risk_threshold_low": accrual_model.risk_threshold_low,
                "risk_threshold_high": accrual_model.risk_threshold_high,
            },
        ]
    )
    diagnostics.to_csv(OUTPUT_DIR / "model_diagnostics.csv", index=False)
    print(f"✅ Diagnostics saved in: {OUTPUT_DIR / 'model_diagnostics.csv'}")


if __name__ == "__main__":
    try:
        main()
    except InsufficientTrainingDataError as exc:
        print(f"Unable to train model with current data: {exc}")