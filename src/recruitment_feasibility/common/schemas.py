"""Data schemas and shared type definitions for the recruitment feasibility prototype."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class StudyRecord:
    """Core one-row-per-study metadata."""

    study_id: str
    study_title: str
    study_type: str
    principal_investigator: Optional[str] = None
    study_start_year: Optional[int] = None


@dataclass
class SimulationResult:
    """Output payload returned by the simulation engine."""

    recruitment_risk: str
    predicted_enrollment_probability: float
    estimated_contacts_required: float
    expected_accrual_rate_per_month: float
    estimated_recruitment_duration_months: float
