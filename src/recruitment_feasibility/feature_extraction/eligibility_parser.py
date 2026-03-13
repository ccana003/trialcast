"""Rule-based eligibility criteria parser for extracting structured protocol features."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional


AGE_RANGE_PATTERN = re.compile(r"(?:age|aged?)\s*(\d{1,3})\s*[-–to]{1,3}\s*(\d{1,3})", re.IGNORECASE)
SINGLE_AGE_PATTERN = re.compile(r"(?:age|aged?)\s*(?:>=|>|at least)?\s*(\d{1,3})", re.IGNORECASE)
VISIT_COUNT_PATTERN = re.compile(r"(\d+)\s+(?:clinic\s+)?visits?", re.IGNORECASE)

KNOWN_DISEASE_KEYWORDS = {
    "hypertension": "hypertension",
    "diabetes": "diabetes",
    "cancer": "oncology",
    "asthma": "asthma",
    "depression": "mental_health",
}


@dataclass
class EligibilityFeatures:
    """Structured features extracted from free-text eligibility criteria."""

    min_age: Optional[int]
    max_age: Optional[int]
    disease_category: Optional[str]
    visit_count: Optional[int]
    healthy_volunteer_flag: int
    eligibility_complexity: float


class EligibilityFeatureExtractor:
    """Extract protocol features from unstructured eligibility text.

    This MVP uses deterministic regex and keyword heuristics for transparency.
    """

    def parse(self, text: Optional[str]) -> EligibilityFeatures:
        """Extract structured features from eligibility text."""
        if not text or not text.strip():
            return EligibilityFeatures(None, None, None, None, 0, 0.0)

        lowered = text.lower()
        min_age, max_age = self._extract_age_range(text)
        visit_count = self._extract_visit_count(text)
        disease_category = self._extract_disease_category(lowered)
        healthy_volunteer_flag = int("healthy volunteer" in lowered)
        complexity = self._estimate_complexity(text)

        return EligibilityFeatures(
            min_age=min_age,
            max_age=max_age,
            disease_category=disease_category,
            visit_count=visit_count,
            healthy_volunteer_flag=healthy_volunteer_flag,
            eligibility_complexity=complexity,
        )

    def parse_to_dict(self, text: Optional[str]) -> Dict[str, Optional[float]]:
        """Convenience helper returning a dictionary for DataFrame integration."""
        features = self.parse(text)
        return {
            "min_age": features.min_age,
            "max_age": features.max_age,
            "disease_category": features.disease_category,
            "visit_count": features.visit_count,
            "healthy_volunteer_flag": features.healthy_volunteer_flag,
            "eligibility_complexity": features.eligibility_complexity,
        }

    def _extract_age_range(self, text: str) -> tuple[Optional[int], Optional[int]]:
        range_match = AGE_RANGE_PATTERN.search(text)
        if range_match:
            return int(range_match.group(1)), int(range_match.group(2))

        single_match = SINGLE_AGE_PATTERN.search(text)
        if single_match:
            return int(single_match.group(1)), None

        return None, None

    def _extract_visit_count(self, text: str) -> Optional[int]:
        visit_match = VISIT_COUNT_PATTERN.search(text)
        if visit_match:
            return int(visit_match.group(1))
        return None

    def _extract_disease_category(self, lowered_text: str) -> Optional[str]:
        for keyword, category in KNOWN_DISEASE_KEYWORDS.items():
            if keyword in lowered_text:
                return category
        return None

    def _estimate_complexity(self, text: str) -> float:
        """Heuristic complexity score based on length and conjunctions."""
        token_count = len(text.split())
        conjunction_count = len(re.findall(r"\b(and|or|with|without)\b", text, re.IGNORECASE))
        return round(min(10.0, token_count / 20.0 + conjunction_count * 0.5), 2)
