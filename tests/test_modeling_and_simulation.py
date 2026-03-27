from __future__ import annotations

import pandas as pd

from recruitment_feasibility.data_ingestion.loader import DataIngestionService
from recruitment_feasibility.model_training.trainer import RecruitmentModelTrainer
from recruitment_feasibility.simulation_engine.simulator import RecruitmentSimulator


def _feature_columns() -> list[str]:
    return [
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
        "recruitment_staff_count",
        "dedicated_recruiter",
        "site_count",
        "recruitment_methods_count",
    ]


def _build_training_df() -> pd.DataFrame:
    rows = []
    for i in range(24):
        contacted = 80 + i * 3
        enrolled = 8 + (i % 10)
        rows.append(
            {
                "study_id": f"S{i:03d}",
                "study_type": "interventional" if i % 2 == 0 else "observational",
                "investigator_experience": 2 + (i % 6),
                "reviewer_concern_recruitment": ["low", "medium", "high"][i % 3],
                "visit_count": 1 + (i % 4),
                "eligibility_complexity": 1.0 + i / 10.0,
                "disease_category": ["hypertension", "diabetes", "oncology"][i % 3],
                "healthy_volunteer_flag": i % 2,
                "has_feasibility_data": 1 if i % 5 else 0,
                "has_recruitment_data": 1 if i % 4 else 0,
                "has_protocol_data": 1 if i % 3 else 0,
                "recruitment_staff_count": 1 + (i % 3),
                "dedicated_recruiter": 1 if i % 4 == 0 else 0,
                "site_count": 1 + (i % 4),
                "recruitment_methods_count": 1 + (i % 3),
                "patients_contacted": contacted,
                "participants_enrolled": enrolled,
                "recruitment_start_date": "2023-01-01",
                "recruitment_end_date": "2023-07-01",
            }
        )
    return pd.DataFrame(rows)


def test_loader_adds_missing_data_indicators(tmp_path) -> None:
    service = DataIngestionService(tmp_path)

    sources = {
        "studies": pd.DataFrame({"study_id": ["A", "B", "C"]}),
        "feasibility_data": pd.DataFrame({"study_id": ["A"], "investigator_experience": [5]}),
        "recruitment_data": pd.DataFrame({"study_id": ["B"], "patients_contacted": [100]}),
        "protocol_data": pd.DataFrame(
            {
                "study_id": ["C"],
                "eligibility_criteria_text": ["Adults age 18 to 65 with two visits."],
            }
        ),
    }

    merged = service.build_merged_training_frame(sources)

    by_id = merged.set_index("study_id")
    assert int(by_id.loc["A", "has_feasibility_data"]) == 1
    assert int(by_id.loc["A", "has_recruitment_data"]) == 0
    assert int(by_id.loc["B", "has_recruitment_data"]) == 1
    assert int(by_id.loc["C", "has_protocol_data"]) == 1


def test_loader_applies_recruitment_capacity_defaults(tmp_path) -> None:
    service = DataIngestionService(tmp_path)
    sources = {
        "studies": pd.DataFrame({"study_id": ["A"]}),
        "feasibility_data": pd.DataFrame({"study_id": ["A"]}),
        "recruitment_data": pd.DataFrame({"study_id": ["A"]}),
        "protocol_data": pd.DataFrame({"study_id": ["A"], "eligibility_criteria_text": ["Adults"]}),
    }

    merged = service.build_merged_training_frame(sources)

    assert int(merged.loc[0, "recruitment_staff_count"]) == 1
    assert int(merged.loc[0, "dedicated_recruiter"]) == 0
    assert int(merged.loc[0, "site_count"]) == 1
    assert int(merged.loc[0, "recruitment_methods_count"]) == 1


def test_trainer_selects_model_and_sets_risk_thresholds() -> None:
    trainer = RecruitmentModelTrainer(random_state=1)
    df = trainer.add_target_metrics(_build_training_df())

    features = _feature_columns()
    categorical = ["study_type", "disease_category", "reviewer_concern_recruitment"]

    artifacts = trainer.train(df, feature_columns=features, target_column="enrollment_probability", categorical_features=categorical)

    assert artifacts.selected_model_name in {"elasticnet", "random_forest"}
    assert artifacts.risk_threshold_low <= artifacts.risk_threshold_high


def test_simulate_distribution_returns_summary_metrics() -> None:
    trainer = RecruitmentModelTrainer(random_state=1)
    df = trainer.add_target_metrics(_build_training_df())

    features = _feature_columns()
    categorical = ["study_type", "disease_category", "reviewer_concern_recruitment"]

    enrollment_model = trainer.train(df, feature_columns=features, target_column="enrollment_probability", categorical_features=categorical)
    accrual_model = trainer.train(df, feature_columns=features, target_column="expected_accrual_rate", categorical_features=categorical)

    simulator = RecruitmentSimulator(enrollment_model=enrollment_model, accrual_model=accrual_model)

    distribution = simulator.simulate_distribution(
        predicted_enrollment_probability=0.1,
        predicted_accrual_rate=8.0,
        target_enrollment=120,
        n_simulations=200,
        random_state=7,
    )

    assert set(distribution.keys()) == {
        "median_duration_months",
        "p80_duration_months",
        "p90_duration_months",
        "probability_within_12_months",
        "probability_within_24_months",
    }
    assert 0.0 <= distribution["probability_within_12_months"] <= 1.0
    assert 0.0 <= distribution["probability_within_24_months"] <= 1.0


def test_simulate_changes_contacts_when_visit_burden_changes() -> None:
    trainer = RecruitmentModelTrainer(random_state=1)
    df = trainer.add_target_metrics(_build_training_df())

    features = _feature_columns()
    categorical = ["study_type", "disease_category", "reviewer_concern_recruitment"]

    enrollment_model = trainer.train(df, feature_columns=features, target_column="enrollment_probability", categorical_features=categorical)
    accrual_model = trainer.train(df, feature_columns=features, target_column="expected_accrual_rate", categorical_features=categorical)

    simulator = RecruitmentSimulator(enrollment_model=enrollment_model, accrual_model=accrual_model)

    base_proposal = {
        "study_type": "interventional",
        "investigator_experience": 6,
        "reviewer_concern_recruitment": "medium",
        "visit_count": 2,
        "eligibility_complexity": 1.0,
        "disease_category": "hypertension",
        "healthy_volunteer_flag": 0,
        "has_feasibility_data": 1,
        "has_recruitment_data": 1,
        "has_protocol_data": 1,
        "recruitment_staff_count": 1,
        "dedicated_recruiter": 0,
        "site_count": 1,
        "recruitment_methods_count": 1,
    }

    modified_proposal = dict(base_proposal, visit_count=6, eligibility_complexity=2.0)

    current = simulator.simulate(base_proposal, target_enrollment=120)
    modified = simulator.simulate(modified_proposal, target_enrollment=120)

    assert current.predicted_enrollment_probability > modified.predicted_enrollment_probability
    assert current.estimated_contacts_required < modified.estimated_contacts_required


def test_simulate_scenarios_returns_required_columns() -> None:
    trainer = RecruitmentModelTrainer(random_state=1)
    df = trainer.add_target_metrics(_build_training_df())

    features = _feature_columns()
    categorical = ["study_type", "disease_category", "reviewer_concern_recruitment"]

    enrollment_model = trainer.train(df, feature_columns=features, target_column="enrollment_probability", categorical_features=categorical)
    accrual_model = trainer.train(df, feature_columns=features, target_column="expected_accrual_rate", categorical_features=categorical)

    simulator = RecruitmentSimulator(enrollment_model=enrollment_model, accrual_model=accrual_model)

    base_input = {
        "study_type": "interventional",
        "investigator_experience": 5,
        "reviewer_concern_recruitment": "medium",
        "visit_count": 2,
        "eligibility_complexity": 1.1,
        "disease_category": "hypertension",
        "healthy_volunteer_flag": 0,
        "has_feasibility_data": 1,
        "has_recruitment_data": 1,
        "has_protocol_data": 1,
        "recruitment_staff_count": 1,
        "dedicated_recruiter": 0,
        "site_count": 1,
        "recruitment_methods_count": 1,
        "target_enrollment": 120,
    }

    scenarios = [
        {"scenario_name": "baseline"},
        {"scenario_name": "+1 staff", "recruitment_staff_count": 2},
    ]
    out = simulator.simulate_scenarios(base_input=base_input, scenarios=scenarios)

    assert list(out["scenario_name"]) == ["baseline", "+1 staff"]
    assert set(out.columns) == {
        "scenario_name",
        "recruitment_staff_count",
        "site_count",
        "recruitment_methods_count",
        "enrollment_probability",
        "predicted_accrual_rate",
        "estimated_duration_months",
        "contacts_required",
        "risk_level",
    }
