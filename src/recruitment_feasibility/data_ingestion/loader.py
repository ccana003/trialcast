"""Data ingestion utilities for loading and integrating institutional datasets."""
from __future__ import annotations

from recruitment_feasibility.feature_extraction.eligibility_parser import EligibilityFeatureExtractor

from pathlib import Path
from typing import Dict

import pandas as pd


class DataIngestionService:
    """Load source datasets and create an integrated study-level table.

    The ingestion layer intentionally tolerates partial data coverage where some
    studies may only appear in a subset of source systems.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    def load_csv(self, file_name: str) -> pd.DataFrame:
        """Load a CSV from the configured data directory."""
        file_path = self.data_dir / file_name
        return pd.read_csv(file_path)

    def load_all_sources(self) -> Dict[str, pd.DataFrame]:
        """Load all known source tables for the MVP prototype."""
        return {
            "studies": self.load_csv("studies.csv"),
            "feasibility_data": self.load_csv("feasibility_data.csv"),
            "recruitment_data": self.load_csv("recruitment_data.csv"),
            "protocol_data": self.load_csv("protocol_data.csv"),
        }

    def build_merged_training_frame(self, sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create a denormalized study-level frame using left joins from studies.

        Missing values are preserved by design to support partial records.
        """
        extractor = EligibilityFeatureExtractor()
        merged = sources["studies"].copy()

        merged = merged.merge(sources["feasibility_data"], on="study_id", how="left")
        merged = merged.merge(sources["recruitment_data"], on="study_id", how="left")
        merged = merged.merge(
            sources["protocol_data"][["study_id", "eligibility_criteria_text"]],
            on="study_id",
            how="left",
        )

        merged["has_feasibility_data"] = merged["study_id"].isin(sources["feasibility_data"]["study_id"]).astype(int)
        merged["has_recruitment_data"] = merged["study_id"].isin(sources["recruitment_data"]["study_id"]).astype(int)
        merged["has_protocol_data"] = merged["eligibility_criteria_text"].notna().astype(int)

        # --- Extract eligibility features ---
        feature_df = merged["eligibility_criteria_text"].apply(
            lambda x: extractor.parse_to_dict(x)
        )

        feature_df = pd.DataFrame(list(feature_df))

        # Drop existing columns if they already exist (prevents duplicates)
        merged = merged.drop(columns=[col for col in feature_df.columns if col in merged.columns])

        merged = pd.concat([merged, feature_df], axis=1)

        # --- FINAL SAFETY: ensure no duplicates anywhere ---
        merged = merged.loc[:, ~merged.columns.duplicated()]

        # Ensure expected model features always exist and provide stable defaults
        # for partially-populated studies. This keeps all rows in the dataset.
        if "visit_count" not in merged.columns:
            merged["visit_count"] = 2
        else:
            merged["visit_count"] = pd.to_numeric(merged["visit_count"], errors="coerce").fillna(2)

        if "eligibility_complexity" not in merged.columns:
            merged["eligibility_complexity"] = 1.0
        else:
            merged["eligibility_complexity"] = pd.to_numeric(
                merged["eligibility_complexity"],
                errors="coerce",
            ).fillna(1.0)

        return merged
