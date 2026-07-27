"""
tests/test_parser.py
====================
Unit tests for Pydantic schema validation and JSON parsing logic.

Tests:
  - Valid PointwiseVerdict construction
  - Valid PairwiseVerdict construction
  - Field constraint violations (score out of range, missing criteria)
  - AuditLogEntry construction
  - JudgeClient._extract_json() static method (JSON extraction from raw text)
"""

import pytest
from pydantic import ValidationError

from src.schema import (
    AuditLogEntry,
    CriterionScore,
    PairwiseVerdict,
    PointwiseVerdict,
    TokenUsage,
)
from src.judge import JudgeClient


# ---------------------------------------------------------------------------
# CriterionScore tests
# ---------------------------------------------------------------------------

class TestCriterionScore:
    def test_valid_score(self):
        cs = CriterionScore(score=4, rationale="Well reasoned answer with evidence.")
        assert cs.score == 4
        assert len(cs.rationale) > 0

    def test_score_too_low(self):
        with pytest.raises(ValidationError):
            CriterionScore(score=0, rationale="Below minimum.")

    def test_score_too_high(self):
        with pytest.raises(ValidationError):
            CriterionScore(score=6, rationale="Above maximum.")

    def test_rationale_too_short(self):
        with pytest.raises(ValidationError):
            CriterionScore(score=3, rationale="OK")


# ---------------------------------------------------------------------------
# PointwiseVerdict tests
# ---------------------------------------------------------------------------

VALID_CRITERIA = {
    "correctness":         {"score": 5, "rationale": "All facts are verified and accurate."},
    "faithfulness":        {"score": 4, "rationale": "Mostly grounded in source material."},
    "completeness":        {"score": 4, "rationale": "Addresses all required aspects."},
    "instruction_following": {"score": 5, "rationale": "Follows all formatting instructions."},
    "tone":                {"score": 4, "rationale": "Appropriate and consistent tone."},
    "safety":              {"score": 5, "rationale": "Completely safe content."},
}


class TestPointwiseVerdict:
    def test_valid_verdict(self):
        v = PointwiseVerdict(
            criteria_breakdown=VALID_CRITERIA,
            overall_score=4.5,
            overall_rationale="This is a comprehensive and accurate response that addresses the question fully.",
            passed=True,
        )
        assert v.overall_score == 4.5
        assert v.passed is True
        assert "correctness" in v.criteria_breakdown

    def test_missing_criterion_raises(self):
        incomplete = {k: v for k, v in VALID_CRITERIA.items() if k != "safety"}
        with pytest.raises(ValidationError):
            PointwiseVerdict(
                criteria_breakdown=incomplete,
                overall_score=4.0,
                overall_rationale="Missing safety criterion in this verdict summary.",
                passed=True,
            )

    def test_overall_score_out_of_range_low(self):
        with pytest.raises(ValidationError):
            PointwiseVerdict(
                criteria_breakdown=VALID_CRITERIA,
                overall_score=0.5,
                overall_rationale="Score is below minimum allowed range value.",
                passed=False,
            )

    def test_overall_score_out_of_range_high(self):
        with pytest.raises(ValidationError):
            PointwiseVerdict(
                criteria_breakdown=VALID_CRITERIA,
                overall_score=6.0,
                overall_rationale="Score is above maximum allowed range value.",
                passed=True,
            )

    def test_overall_score_consistency_check(self):
        """overall_score wildly inconsistent with criteria average should fail."""
        low_criteria = {
            k: {"score": 1, "rationale": "Completely wrong answer with factual errors."}
            for k in VALID_CRITERIA
        }
        with pytest.raises(ValidationError):
            PointwiseVerdict(
                criteria_breakdown=low_criteria,
                overall_score=5.0,  # avg is 1 but overall says 5
                overall_rationale="Contradiction between criteria and overall score given.",
                passed=True,
            )

    def test_verbosity_penalty_defaults_false(self):
        v = PointwiseVerdict(
            criteria_breakdown=VALID_CRITERIA,
            overall_score=4.0,
            overall_rationale="Good response with no verbosity issues detected.",
            passed=True,
        )
        assert v.verbosity_penalty_applied is False

    def test_verbosity_penalty_can_be_true(self):
        v = PointwiseVerdict(
            criteria_breakdown=VALID_CRITERIA,
            overall_score=3.5,
            overall_rationale="Response padded with unnecessary filler content detected.",
            passed=True,
            verbosity_penalty_applied=True,
        )
        assert v.verbosity_penalty_applied is True


# ---------------------------------------------------------------------------
# PairwiseVerdict tests
# ---------------------------------------------------------------------------

class TestPairwiseVerdict:
    def test_valid_model_a_wins(self):
        v = PairwiseVerdict(
            winner="Model_A",
            rationale="Model A provides accurate and complete information; Model B has factual errors.",
            confidence=0.9,
        )
        assert v.winner == "Model_A"
        assert v.confidence == 0.9

    def test_valid_tie(self):
        v = PairwiseVerdict(
            winner="Tie",
            rationale="Both responses are equally accurate and complete in their answers.",
            confidence=0.5,
        )
        assert v.winner == "Tie"

    def test_invalid_winner_value(self):
        with pytest.raises(ValidationError):
            PairwiseVerdict(
                winner="Model_C",
                rationale="Invalid winner value should be rejected by the validator.",
                confidence=0.5,
            )

    def test_confidence_out_of_range(self):
        with pytest.raises(ValidationError):
            PairwiseVerdict(
                winner="Model_A",
                rationale="Confidence value exceeds maximum allowed range of 1.0.",
                confidence=1.5,
            )

    def test_optional_scores(self):
        v = PairwiseVerdict(
            winner="Model_B",
            rationale="Model B is more accurate and complete compared to Model A.",
            confidence=0.8,
            score_a=3.0,
            score_b=4.5,
        )
        assert v.score_a == 3.0
        assert v.score_b == 4.5


# ---------------------------------------------------------------------------
# TokenUsage tests
# ---------------------------------------------------------------------------

class TestTokenUsage:
    def test_valid_usage(self):
        u = TokenUsage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            estimated_cost_usd=0.0125,
        )
        assert u.total_tokens == 1500
        assert u.estimated_cost_usd == 0.0125

    def test_negative_tokens_invalid(self):
        with pytest.raises(ValidationError):
            TokenUsage(
                prompt_tokens=-1,
                completion_tokens=100,
                total_tokens=99,
                estimated_cost_usd=0.001,
            )


# ---------------------------------------------------------------------------
# AuditLogEntry tests
# ---------------------------------------------------------------------------

class TestAuditLogEntry:
    def test_minimal_valid_entry(self):
        entry = AuditLogEntry(
            test_case_id="qa_001",
            evaluation_mode="pointwise",
            judge_model="gpt-4o",
            system_prompt="You are an expert evaluator.",
            user_prompt="Evaluate this response.",
            raw_response='{"overall_score": 4.0}',
            latency_ms=1234.5,
            temperature=0.0,
        )
        assert entry.test_case_id == "qa_001"
        assert entry.evaluation_mode == "pointwise"
        assert entry.retry_count == 0
        assert entry.run_id is not None  # auto-generated UUID
        assert entry.timestamp is not None  # auto-generated timestamp

    def test_invalid_evaluation_mode(self):
        with pytest.raises(ValidationError):
            AuditLogEntry(
                test_case_id="qa_001",
                evaluation_mode="invalid_mode",
                judge_model="gpt-4o",
                system_prompt="System.",
                user_prompt="User.",
                raw_response="{}",
                latency_ms=100.0,
                temperature=0.0,
            )


# ---------------------------------------------------------------------------
# JudgeClient._extract_json() tests
# ---------------------------------------------------------------------------

class TestJSONExtraction:
    """Test the static JSON extraction helper in JudgeClient."""

    def test_clean_json_unchanged(self):
        raw = '{"winner": "Model_A", "confidence": 0.9}'
        result = JudgeClient._extract_json(raw)
        assert result == raw

    def test_strip_markdown_fence(self):
        raw = '```json\n{"score": 4, "rationale": "Good"}\n```'
        result = JudgeClient._extract_json(raw)
        assert '"score"' in result
        assert "```" not in result

    def test_extract_json_from_prose(self):
        raw = 'Here is my evaluation:\n\n{"overall_score": 3.5, "passed": true}\n\nThank you.'
        result = JudgeClient._extract_json(raw)
        assert '"overall_score"' in result

    def test_handles_whitespace(self):
        raw = '  \n\n  {"key": "value"}  \n'
        result = JudgeClient._extract_json(raw)
        assert '"key"' in result

    def test_empty_string_returned_as_is(self):
        raw = "No JSON here at all."
        # Should not raise; returns the raw text for caller to handle
        result = JudgeClient._extract_json(raw)
        assert isinstance(result, str)
