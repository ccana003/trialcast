"""Rule-based eligibility criteria parser for extracting structured protocol features."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional


AGE_RANGE_PATTERNS = [
    re.compile(r"(?:age|aged?)\s*(\d{1,3})\s*(?:-|–|to)\s*(\d{1,3})", re.IGNORECASE),
    re.compile(r"between\s*(\d{1,3})\s*and\s*(\d{1,3})\s*(?:years?|yo)?", re.IGNORECASE),
]
MIN_AGE_PATTERNS = [
    re.compile(r"(?:age|aged?)\s*(?:>=|=>|at\s+least|minimum\s+of)?\s*(\d{1,3})", re.IGNORECASE),
    re.compile(r"older\s+than\s*(\d{1,3})", re.IGNORECASE),
]
VISIT_COUNT_PATTERN = re.compile(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b\s+(?:on-site\s+|in-person\s+|clinic\s+)?visits?\b", re.IGNORECASE)

KNOWN_DISEASE_KEYWORDS = {
    "hypertension": "hypertension",
    "diabetes": "diabetes",
    "cancer": "oncology",
    "asthma": "asthma",
    "depression": "mental_health",
}

WORD_TO_NUMBER = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
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
        normalized_text = self._normalize_text(text)

        min_age, max_age = self._extract_age_range(normalized_text)
        visit_count = self._extract_visit_count(normalized_text)
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

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.split())

    def _extract_age_range(self, text: str) -> tuple[Optional[int], Optional[int]]:
        for pattern in AGE_RANGE_PATTERNS:
            match = pattern.search(text)
            if match:
                return int(match.group(1)), int(match.group(2))

        for pattern in MIN_AGE_PATTERNS:
            match = pattern.search(text)
            if match:
                min_age = int(match.group(1))
                if "older than" in match.group(0).lower():
                    min_age += 1
                return min_age, None

        return None, None

    def _extract_visit_count(self, text: str) -> Optional[int]:
        visit_match = VISIT_COUNT_PATTERN.search(text)
        if not visit_match:
            return None

        token = visit_match.group(1).lower()
        if token.isdigit():
            return int(token)
        return WORD_TO_NUMBER.get(token)

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
