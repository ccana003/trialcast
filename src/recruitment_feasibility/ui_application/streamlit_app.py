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
        "Models are cross-validated and compared against a fold-wise mean-target baseline. "
        f"Selected models: enrollment={simulator.enrollment_model.selected_model_name}, "
        f"accrual={simulator.accrual_model.selected_model_name}."
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


def _render_model_explanation(simulator: RecruitmentSimulator) -> None:
    st.subheader("Model explanation")
    enrollment_artifact = simulator.enrollment_model

    if enrollment_artifact.selected_model_name != "elasticnet":
        st.info("Top driver explanation is available when the selected enrollment model is ElasticNet.")
        return

    model_pipeline = enrollment_artifact.model
    preprocessor = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out().tolist()
    coefficients = pd.Series(model.coef_, index=feature_names)

    top_positive = coefficients.sort_values(ascending=False).head(5)
    top_negative = coefficients.sort_values(ascending=True).head(5)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top positive drivers**")
        st.dataframe(top_positive.rename("coefficient"))

    with col2:
        st.markdown("**Top negative drivers**")
        st.dataframe(top_negative.rename("coefficient"))

    st.caption(
        "Positive coefficients increase predicted enrollment probability, while negative "
        "coefficients decrease it, all else being equal."
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
    _render_model_explanation(simulator)

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
            index=["hypertension", "diabetes", "oncology", "asthma", "mental_health", "other"].index(
                (editable.get("disease_category") or "other")
                if (editable.get("disease_category") or "other") in ["hypertension", "diabetes", "oncology", "asthma", "mental_health", "other"]
                else "other"
            ),
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

    st.caption(f"Reviewed age range: {min_age} to {max_age if max_age > 0 else 'not specified'}")

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
        distribution = simulator.simulate_distribution(
            predicted_enrollment_probability=result.predicted_enrollment_probability,
            predicted_accrual_rate=result.expected_accrual_rate_per_month,
            target_enrollment=int(target_enrollment),
            n_simulations=1000,
        )

        st.markdown("## Simulation Output")
        st.metric("Recruitment Risk", result.recruitment_risk)
        st.metric("Predicted enrollment probability", f"{result.predicted_enrollment_probability:.2%}")
        st.metric("Estimated contacts required", f"{result.estimated_contacts_required:,.0f}")
        st.metric("Expected accrual rate", f"{result.expected_accrual_rate_per_month:.1f} participants/month")
        st.metric(
            "Estimated recruitment duration",
            f"{result.estimated_recruitment_duration_months:.1f} months",
        )

        st.markdown("### Monte Carlo recruitment outlook")
        st.metric("Median duration", f"{distribution['median_duration_months']:.1f} months")
        st.metric("P80 duration", f"{distribution['p80_duration_months']:.1f} months")
        st.metric("P90 duration", f"{distribution['p90_duration_months']:.1f} months")
        st.metric("Finish within 12 months", f"{distribution['probability_within_12_months']:.1%}")
        st.metric("Finish within 24 months", f"{distribution['probability_within_24_months']:.1%}")


if __name__ == "__main__":
    render_app()
