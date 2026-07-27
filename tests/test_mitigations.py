"""
tests/test_mitigations.py
=========================
Unit tests for bias detection and mitigation logic.

Tests:
  - calculate_flip_rate()           — position bias flip rate calculation
  - analyze_score_distribution()    — score clustering detection
  - check_self_enhancement_risk()   — model family comparison
  - WINNER_SWAP_MAP completeness    — mapping correctness
"""

import pytest

from src.mitigations import (
    WINNER_SWAP_MAP,
    calculate_flip_rate,
    check_self_enhancement_risk,
    analyze_score_distribution,
)


# ---------------------------------------------------------------------------
# Flip Rate tests
# ---------------------------------------------------------------------------

class TestCalculateFlipRate:
    def test_no_inconsistencies(self):
        results = [
            {"position_consistent": True},
            {"position_consistent": True},
            {"position_consistent": True},
        ]
        assert calculate_flip_rate(results) == 0.0

    def test_all_inconsistent(self):
        results = [
            {"position_consistent": False},
            {"position_consistent": False},
        ]
        assert calculate_flip_rate(results) == 1.0

    def test_half_inconsistent(self):
        results = [
            {"position_consistent": True},
            {"position_consistent": False},
        ]
        rate = calculate_flip_rate(results)
        assert rate == pytest.approx(0.5)

    def test_empty_results(self):
        assert calculate_flip_rate([]) == 0.0

    def test_missing_key_defaults_consistent(self):
        """Results missing 'position_consistent' key should default to True (not a flip)."""
        results = [{"final_winner": "Model_A"}]
        assert calculate_flip_rate(results) == 0.0


# ---------------------------------------------------------------------------
# Winner swap map tests
# ---------------------------------------------------------------------------

class TestWinnerSwapMap:
    def test_all_labels_covered(self):
        """All expected winner labels must be in the swap map."""
        expected_labels = {"Model_A", "Model_B", "Tie"}
        assert set(WINNER_SWAP_MAP.keys()) >= expected_labels

    def test_model_a_maps_to_model_b(self):
        assert WINNER_SWAP_MAP["Model_A"] == "Model_B"

    def test_model_b_maps_to_model_a(self):
        assert WINNER_SWAP_MAP["Model_B"] == "Model_A"

    def test_tie_maps_to_tie(self):
        assert WINNER_SWAP_MAP["Tie"] == "Tie"

    def test_swap_is_symmetric(self):
        """Applying swap twice should return original."""
        for original, swapped in WINNER_SWAP_MAP.items():
            if swapped in WINNER_SWAP_MAP:
                assert WINNER_SWAP_MAP[swapped] == original or original == "Tie"


# ---------------------------------------------------------------------------
# Self-enhancement risk tests
# ---------------------------------------------------------------------------

class TestSelfEnhancementRisk:
    def test_same_family_openai(self):
        result = check_self_enhancement_risk(
            judge_model="gpt-4o",
            candidate_model_a="gpt-3.5-turbo",
            candidate_model_b="claude-3-haiku",
        )
        assert result["risk_detected"] is True
        assert result["judge_family"] == "openai"

    def test_no_risk_different_families(self):
        result = check_self_enhancement_risk(
            judge_model="claude-3-5-sonnet",
            candidate_model_a="gpt-3.5-turbo",
            candidate_model_b="gpt-4o-mini",
        )
        assert result["risk_detected"] is False

    def test_both_candidates_same_family_as_judge(self):
        result = check_self_enhancement_risk(
            judge_model="gpt-4o",
            candidate_model_a="gpt-3.5-turbo",
            candidate_model_b="gpt-4o-mini",
        )
        assert result["risk_detected"] is True

    def test_unknown_model_no_risk(self):
        """Unknown model families should not trigger risk."""
        result = check_self_enhancement_risk(
            judge_model="some-unknown-model",
            candidate_model_a="another-unknown",
            candidate_model_b="yet-another",
        )
        assert result["risk_detected"] is False

    def test_recommendation_included(self):
        result = check_self_enhancement_risk(
            judge_model="gpt-4o",
            candidate_model_a="gpt-3.5-turbo",
            candidate_model_b="claude-3-haiku",
        )
        assert "recommendation" in result
        assert len(result["recommendation"]) > 10


# ---------------------------------------------------------------------------
# Score distribution / clustering tests
# ---------------------------------------------------------------------------

class TestAnalyzeScoreDistribution:
    def test_clustered_scores(self):
        """All scores in band 3-4 → clustering detected."""
        scores = [3.2, 3.4, 3.6, 3.8, 3.1, 3.5, 3.7, 3.3, 3.9, 3.0]
        result = analyze_score_distribution(scores, clustering_threshold=0.7)
        assert result["is_clustered"] is True

    def test_healthy_distribution(self):
        """Spread scores → no clustering."""
        scores = [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.0]
        result = analyze_score_distribution(scores, clustering_threshold=0.7)
        assert result["is_clustered"] is False

    def test_empty_scores(self):
        result = analyze_score_distribution([], clustering_threshold=0.7)
        assert result["score_count"] == 0
        assert result["is_clustered"] is False

    def test_single_score(self):
        result = analyze_score_distribution([4.0])
        assert result["score_count"] == 1
        # std is 0 for a single value
        assert result["std_score"] == 0.0

    def test_mean_calculation(self):
        scores = [2.0, 4.0]
        result = analyze_score_distribution(scores)
        assert result["mean_score"] == pytest.approx(3.0)

    def test_std_calculation(self):
        scores = [1.0, 5.0]
        result = analyze_score_distribution(scores)
        assert result["std_score"] > 0

    def test_dominant_band_identified(self):
        scores = [3.1, 3.2, 3.3, 3.4, 3.5, 1.0, 5.0]
        result = analyze_score_distribution(scores)
        assert "dominant_band" in result
        assert result["dominant_band"] is not None

    def test_recommendation_included(self):
        scores = [3.0, 3.1, 3.2, 3.3, 3.4]
        result = analyze_score_distribution(scores)
        assert "recommendation" in result
        assert len(result["recommendation"]) > 10


# ---------------------------------------------------------------------------
# Integration smoke test (no API calls)
# ---------------------------------------------------------------------------

class TestMitigationsIntegration:
    def test_flip_rate_and_distribution_pipeline(self):
        """Simulate a mini pipeline: pairwise results → flip rate → clustering."""
        pairwise_results = [
            {"position_consistent": True,  "final_winner": "Model_A"},
            {"position_consistent": False, "final_winner": "Tie (Position Inconsistency)"},
            {"position_consistent": True,  "final_winner": "Model_B"},
            {"position_consistent": True,  "final_winner": "Model_A"},
        ]
        flip_rate = calculate_flip_rate(pairwise_results)
        assert flip_rate == pytest.approx(0.25)

        # Simulate scores from the same suite
        scores = [4.5, 3.0, 2.5, 4.0]
        dist = analyze_score_distribution(scores)
        assert dist["score_count"] == 4
        assert isinstance(dist["is_clustered"], bool)
