"""Data ingestion utilities for loading and integrating institutional datasets."""

from __future__ import annotations

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
        merged = sources["studies"].copy()

        merged = merged.merge(sources["feasibility_data"], on="study_id", how="left")
        merged = merged.merge(sources["recruitment_data"], on="study_id", how="left")
        merged = merged.merge(
            sources["protocol_data"][["study_id", "eligibility_criteria_text"]],
            on="study_id",
            how="left",
        )

        return merged
