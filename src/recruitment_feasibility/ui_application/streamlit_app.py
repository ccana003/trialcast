"""Streamlit prototype UI for recruitment feasibility simulation."""

from __future__ import annotations

from pathlib import Path
import sys


CURRENT_FILE = Path(__file__).resolve()
SRC_ROOT = CURRENT_FILE.parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd
import streamlit as st

from recruitment_feasibility.data_ingestion.loader import DataIngestionService
from recruitment_feasibility.feature_extraction.eligibility_parser import EligibilityFeatureExtractor
from recruitment_feasibility.model_training.trainer import (
    InsufficientTrainingDataError,
    RecruitmentModelTrainer,
)
from recruitment_feasibility.simulation_engine.simulator import RecruitmentSimulator


REPO_ROOT = CURRENT_FILE.parents[3]
DATA_DIR = REPO_ROOT / "data"


@st.cache_resource(show_spinner=False)
def build_training_assets(data_signature: tuple[float, ...]) -> RecruitmentSimulator:
    """Load data, extract features, train models, and return simulator."""
    _ = data_signature
    ingestion = DataIngestionService(DATA_DIR)
    extractor = EligibilityFeatureExtractor()
    trainer = RecruitmentModelTrainer()

    sources = ingestion.load_all_sources()
    merged = ingestion.build_merged_training_frame(sources)

    feature_df = pd.json_normalize(
        merged["eligibility_criteria_text"].apply(extractor.parse_to_dict),
    )
    training_df = pd.concat([merged, feature_df], axis=1)
    training_df = trainer.add_target_metrics(training_df)

    common_features = [
        "study_type",
        "investigator_experience",
        "reviewer_concern_recruitment",
        "visit_count",
        "eligibility_complexity",
        "disease_category",
        "healthy_volunteer_flag",
    ]
    categorical = ["study_type", "disease_category", "reviewer_concern_recruitment"]

    enrollment_model = trainer.train(
        training_df,
        feature_columns=common_features,
        target_column="enrollment_probability",
        categorical_features=categorical,
    )
    accrual_model = trainer.train(
        training_df,
        feature_columns=common_features,
        target_column="expected_accrual_rate",
        categorical_features=categorical,
    )

    return RecruitmentSimulator(enrollment_model=enrollment_model, accrual_model=accrual_model)


def _data_signature() -> tuple[float, ...]:
    """Return a timestamp signature so cache refreshes when source data changes."""
    return tuple(
        (DATA_DIR / file_name).stat().st_mtime
        for file_name in ["studies.csv", "feasibility_data.csv", "recruitment_data.csv", "protocol_data.csv"]
    )


def _render_model_quality(simulator: RecruitmentSimulator) -> None:
    st.subheader("Model quality")

    enrollment_eval = simulator.enrollment_model.evaluation
    accrual_eval = simulator.accrual_model.evaluation

    st.caption(
        "Models are validated on a hold-out split and compared against a mean-target baseline."
    )

    quality_df = pd.DataFrame(
        [
            {
                "target": "enrollment_probability",
                "train_rows": enrollment_eval.train_rows,
                "validation_rows": enrollment_eval.validation_rows,
                "model_mae": enrollment_eval.model_mae,
                "baseline_mae": enrollment_eval.baseline_mae,
                "model_rmse": enrollment_eval.model_rmse,
                "baseline_rmse": enrollment_eval.baseline_rmse,
                "model_r2": enrollment_eval.model_r2,
                "baseline_r2": enrollment_eval.baseline_r2,
            },
            {
                "target": "expected_accrual_rate",
                "train_rows": accrual_eval.train_rows,
                "validation_rows": accrual_eval.validation_rows,
                "model_mae": accrual_eval.model_mae,
                "baseline_mae": accrual_eval.baseline_mae,
                "model_rmse": accrual_eval.model_rmse,
                "baseline_rmse": accrual_eval.baseline_rmse,
                "model_r2": accrual_eval.model_r2,
                "baseline_r2": accrual_eval.baseline_r2,
            },
        ]
    )
    st.dataframe(quality_df, use_container_width=True)

    st.caption(
        "Data ranges: "
        f"Enrollment probability [{enrollment_eval.target_min:.4f}, {enrollment_eval.target_median:.4f}, {enrollment_eval.target_max:.4f}] | "
        f"Accrual rate [{accrual_eval.target_min:.2f}, {accrual_eval.target_median:.2f}, {accrual_eval.target_max:.2f}]"
    )


def render_app() -> None:
    """Render the Streamlit user interface."""
    st.set_page_config(page_title="Recruitment Feasibility Simulator", layout="wide")
    st.title("Recruitment Feasibility and Simulation Platform")
    st.caption("Prototype MVP using historical institutional recruitment data.")

    try:
        with st.spinner("Loading historical data and training baseline models..."):
            simulator = build_training_assets(_data_signature())
    except InsufficientTrainingDataError as exc:
        st.error(f"Unable to train model with current data: {exc}")
        st.stop()

    _render_model_quality(simulator)

    st.subheader("Proposed Study Inputs")
    col1, col2 = st.columns(2)

    with col1:
        study_type = st.selectbox("Study type", ["interventional", "observational", "registry"])
        target_enrollment = st.number_input("Target enrollment", min_value=1, value=150)
        visit_count = st.number_input("Required visit count", min_value=0, value=2)
        investigator_experience = st.number_input("Investigator experience (years)", min_value=0, value=5)

    with col2:
        reviewer_concern = st.selectbox("Recruitment concern level", ["low", "medium", "high"])
        disease_category = st.selectbox(
            "Primary disease category",
            ["hypertension", "diabetes", "oncology", "asthma", "mental_health", "other"],
        )
        eligibility_text = st.text_area(
            "Paste eligibility criteria",
            value="Adults age 40-75 with hypertension requiring two clinic visits.",
        )

    extractor = EligibilityFeatureExtractor()
    extracted = extractor.parse_to_dict(eligibility_text)

    st.markdown("### Extracted protocol features")
    st.json(extracted)

    if st.button("Run feasibility simulation"):
        proposal = {
            "study_type": study_type,
            "investigator_experience": investigator_experience,
            "reviewer_concern_recruitment": reviewer_concern,
            "visit_count": extracted.get("visit_count") or visit_count,
            "eligibility_complexity": extracted.get("eligibility_complexity") or 1.0,
            "disease_category": disease_category
            if disease_category != "other"
            else (extracted.get("disease_category") or "other"),
            "healthy_volunteer_flag": extracted.get("healthy_volunteer_flag") or 0,
        }

        result = simulator.simulate(proposal, target_enrollment=int(target_enrollment))

        st.markdown("## Simulation Output")
        st.metric("Recruitment Risk", result.recruitment_risk)
        st.metric("Predicted enrollment probability", f"{result.predicted_enrollment_probability:.2%}")
        st.metric("Estimated contacts required", f"{result.estimated_contacts_required:,.0f}")
        st.metric("Expected accrual rate", f"{result.expected_accrual_rate_per_month:.1f} participants/month")
        st.metric(
            "Estimated recruitment duration",
            f"{result.estimated_recruitment_duration_months:.1f} months",
        )


if __name__ == "__main__":
    render_app()
