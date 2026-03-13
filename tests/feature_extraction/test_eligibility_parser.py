from recruitment_feasibility.feature_extraction.eligibility_parser import EligibilityFeatureExtractor


def test_extracts_standard_age_range_and_visit_words() -> None:
    extractor = EligibilityFeatureExtractor()

    result = extractor.parse("Adults age 18 to 65 with diabetes requiring two clinic visits.")

    assert result.min_age == 18
    assert result.max_age == 65
    assert result.visit_count == 2
    assert result.disease_category == "diabetes"


def test_extracts_between_age_phrase() -> None:
    extractor = EligibilityFeatureExtractor()

    result = extractor.parse("Participants between 21 and 70 years old with asthma.")

    assert result.min_age == 21
    assert result.max_age == 70


def test_extracts_min_age_from_inequality() -> None:
    extractor = EligibilityFeatureExtractor()

    result = extractor.parse("Inclusion criteria: age >= 18 and healthy volunteer.")

    assert result.min_age == 18
    assert result.max_age is None
    assert result.healthy_volunteer_flag == 1


def test_extracts_older_than_age_phrase() -> None:
    extractor = EligibilityFeatureExtractor()

    result = extractor.parse("Adults older than 50 with hypertension and one visit.")

    assert result.min_age == 51
    assert result.max_age is None
    assert result.visit_count == 1


def test_returns_none_when_no_age_or_visits() -> None:
    extractor = EligibilityFeatureExtractor()

    result = extractor.parse("Subjects with depression are eligible.")

    assert result.min_age is None
    assert result.max_age is None
    assert result.visit_count is None
    assert result.disease_category == "mental_health"
