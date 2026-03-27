"""Streamlit prototype UI for recruitment feasibility simulation."""

from __future__ import annotations

from pathlib import Path
import sys

CURRENT_FILE = Path(__file__).resolve()
SRC_ROOT = CURRENT_FILE.parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

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


def _difficulty_label(duration_months: float) -> str:
    """Classify recruitment difficulty from projected duration."""
    if duration_months < 36:
        return "Low"
    if duration_months <= 84:
        return "Moderate"
    return "High"


def _driver_messages(proposal: dict[str, object]) -> list[str]:
    """Summarize key input factors that influence recruitment performance."""
    visit_count = float(proposal.get("visit_count", 2) or 2)
    complexity = float(proposal.get("eligibility_complexity", 1.0) or 1.0)
    disease_category = str(proposal.get("disease_category", "unknown"))
    local_sample = proposal.get("local_sample")
    national_sample = proposal.get("national_sample")

    local_text = "N/A" if local_sample in (None, "") else f"{int(float(local_sample))}"
    national_text = "N/A" if national_sample in (None, "") else f"{int(float(national_sample))}"

    return [
        f"Visit burden: {int(visit_count)} visits",
        f"Eligibility complexity: {complexity:.1f}",
        f"Disease category: {disease_category}",
        f"Sample size context: local={local_text}, national={national_text}",
    ]


def _recommendations(proposal: dict[str, object]) -> list[str]:
    """Generate actionable recommendations from scenario features."""
    recommendations: list[str] = []

    if float(proposal.get("visit_count", 2) or 2) > 2:
        recommendations.append("Reduce visit_count where clinically acceptable to improve accrual pace.")
    if float(proposal.get("eligibility_complexity", 1.0) or 1.0) > 1.5:
        recommendations.append("Lower eligibility complexity to improve enrollment probability.")

    local_sample = proposal.get("local_sample")
    if local_sample not in (None, "") and float(local_sample) >= 120:
        recommendations.append("Consider increasing site count because local_sample target is high for one site.")

    if not recommendations:
        recommendations.append("Current design appears balanced; monitor early screening-to-enrollment conversion.")

    return recommendations


@st.cache_resource(show_spinner=False)
def build_training_assets(data_signature: tuple[float, ...]) -> RecruitmentSimulator:
    """Load data, extract features, train models, and return simulator."""
    _ = data_signature
    ingestion = DataIngestionService(DATA_DIR)
    trainer = RecruitmentModelTrainer()

    sources = ingestion.load_all_sources()
    merged = ingestion.build_merged_training_frame(sources)

    # Use merged directly (already contains parsed features).
    training_df = merged.copy()
    training_df = training_df.loc[:, ~training_df.columns.duplicated()]

    # Add CTMS-informed target variables while preserving all studies.
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
        "local_sample",
        "national_sample",
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
    files = ["studies.csv", "feasibility_data.csv", "recruitment_data.csv", "protocol_data.csv", "ctms_data.csv"]
    signature = []
    for file_name in files:
        file_path = DATA_DIR / file_name
        if file_path.exists():
            signature.append(file_path.stat().st_mtime)
    return tuple(signature)


def _render_result_block(title: str, result) -> None:
    """Render compact metrics for a single scenario."""
    st.markdown(f"### {title}")
    st.metric("Enrollment probability", f"{result.predicted_enrollment_probability:.2%}")
    st.metric("Contacts required", f"{result.estimated_contacts_required:,.0f}")
    st.metric("Accrual rate", f"{result.expected_accrual_rate_per_month:.1f} participants/month")
    st.metric("Duration", f"{result.estimated_recruitment_duration_months:.1f} months")
    st.metric("Recruitment Difficulty Score", _difficulty_label(result.estimated_recruitment_duration_months))


def render_app() -> None:
    """Render the Streamlit decision-support user interface."""
    st.set_page_config(page_title="Recruitment Feasibility Simulator", layout="wide")
    st.title("Recruitment Feasibility Decision-Support Tool")
    st.caption("Uses historical sources plus CTMS (Velos) fields for actionable planning insights.")

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
        local_sample = st.number_input("Local sample target (site)", min_value=1, value=80)

    with col2:
        reviewer_concern = st.selectbox("Recruitment concern level", ["low", "medium", "high"])
        disease_category = st.selectbox(
            "Primary disease category override",
            ["hypertension", "diabetes", "oncology", "asthma", "mental_health", "other"],
        )
        national_sample = st.number_input("National sample target (all sites)", min_value=1, value=450)

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
        st.number_input("Minimum age", min_value=0, value=int(editable.get("min_age") or 0))
        st.number_input("Maximum age", min_value=0, value=int(editable.get("max_age") or 0))
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

    st.markdown("### Step 4 — Scenario comparison controls")
    compare_enabled = st.checkbox("Enable scenario comparison", value=True)
    modified_visit_count = st.number_input("Modified scenario visit_count", min_value=0, value=max(int(visit_count) - 1, 0))
    modified_complexity = st.number_input(
        "Modified scenario eligibility_complexity",
        min_value=0.0,
        max_value=10.0,
        value=max(float(eligibility_complexity) - 0.2, 0.0),
        step=0.1,
    )

    if st.button("Step 5 — Run feasibility simulation"):
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
            "local_sample": local_sample,
            "national_sample": national_sample,
        }

        result_current = simulator.simulate(proposal, target_enrollment=int(target_enrollment))

        st.markdown("## Simulation Output")
        if compare_enabled:
            modified_proposal = proposal.copy()
            modified_proposal["visit_count"] = modified_visit_count
            modified_proposal["eligibility_complexity"] = modified_complexity
            result_modified = simulator.simulate(modified_proposal, target_enrollment=int(target_enrollment))

            left, right = st.columns(2)
            with left:
                _render_result_block("Current Scenario", result_current)
            with right:
                _render_result_block("Modified Scenario", result_modified)
        else:
            _render_result_block("Current Scenario", result_current)

        st.markdown("## Key Drivers")
        for msg in _driver_messages(proposal):
            st.write(f"- {msg}")

        st.markdown("## Recommendations")
        for suggestion in _recommendations(proposal):
            st.write(f"- {suggestion}")


if __name__ == "__main__":
    render_app()
