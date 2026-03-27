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
    trainer = RecruitmentModelTrainer()

    sources = ingestion.load_all_sources()
    merged = ingestion.build_merged_training_frame(sources)

    # Use merged directly (already contains parsed features)
    training_df = merged.copy()

    # Final safety: remove duplicate columns
    training_df = training_df.loc[:, ~training_df.columns.duplicated()]

    # Add target variables
    training_df = trainer.add_target_metrics(training_df)
    print(training_df["enrollment_probability"].describe())
    print(training_df["expected_accrual_rate"].describe())
    print("ENROLLMENT PROBABILITY:")
    print(training_df["enrollment_probability"].describe())

    print("\nACCRUAL RATE:")
    print(training_df["expected_accrual_rate"].describe())

    common_features = [
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

    return RecruitmentSimulator(
        enrollment_model=enrollment_model,
        accrual_model=accrual_model,
    )


def _data_signature() -> tuple[float, ...]:
    """Return a timestamp signature so cache refreshes when source data changes."""
    return tuple(
        (DATA_DIR / file_name).stat().st_mtime
        for file_name in ["studies.csv", "feasibility_data.csv", "recruitment_data.csv", "protocol_data.csv"]
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

    st.subheader("Proposed Study Inputs")
    col1, col2 = st.columns(2)

    with col1:
        study_type = st.selectbox("Study type", ["interventional", "observational", "registry"])
        target_enrollment = st.number_input("Target enrollment", min_value=1, value=150)
        default_visit_count = st.number_input("Fallback visit count", min_value=0, value=2)
        investigator_experience = st.number_input("Investigator experience (years)", min_value=0, value=5)

    with col2:
        reviewer_concern = st.selectbox("Recruitment concern level", ["low", "medium", "high"])
        disease_category = st.selectbox(
            "Primary disease category override",
            ["hypertension", "diabetes", "oncology", "asthma", "mental_health", "other"],
        )

    st.markdown("### Step 1 — Paste eligibility criteria")
    eligibility_text = st.text_area(
        "Eligibility criteria",
        value="Adults age 40-75 with hypertension requiring two clinic visits.",
    )

    extractor = EligibilityFeatureExtractor()

    if "editable_features" not in st.session_state:
        st.session_state.editable_features = extractor.parse_to_dict(eligibility_text)

    if st.button("Step 2 — Run feature extraction"):
        st.session_state.editable_features = extractor.parse_to_dict(eligibility_text)

    st.markdown("### Step 3 — Review and edit extracted features")
    editable = st.session_state.editable_features

    review_col1, review_col2 = st.columns(2)
    with review_col1:
        min_age = st.number_input("Minimum age", min_value=0, value=int(editable.get("min_age") or 0))
        max_age = st.number_input("Maximum age", min_value=0, value=int(editable.get("max_age") or 0))
        visit_count = st.number_input(
            "Visit count",
            min_value=0,
            value=int(editable.get("visit_count") or default_visit_count),
        )

    with review_col2:
        reviewed_disease = st.selectbox(
            "Disease category",
            ["hypertension", "diabetes", "oncology", "asthma", "mental_health", "other"],
            index=0,
        )
        healthy_volunteer_flag = st.selectbox(
            "Healthy volunteer flag",
            options=[0, 1],
            index=1 if int(editable.get("healthy_volunteer_flag") or 0) == 1 else 0,
        )
        eligibility_complexity = st.number_input(
            "Eligibility complexity",
            min_value=0.0,
            max_value=10.0,
            value=float(editable.get("eligibility_complexity") or 1.0),
            step=0.1,
        )

    if st.button("Step 4 — Run feasibility simulation"):
        proposal = {
            "study_type": study_type,
            "investigator_experience": investigator_experience,
            "reviewer_concern_recruitment": reviewer_concern,
            "visit_count": visit_count,
            "eligibility_complexity": eligibility_complexity,
            "disease_category": disease_category if disease_category != "other" else reviewed_disease,
            "healthy_volunteer_flag": int(healthy_volunteer_flag),
            "has_feasibility_data": 1,
            "has_recruitment_data": 0,
            "has_protocol_data": 1,
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