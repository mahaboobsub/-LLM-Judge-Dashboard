"""
src/schema.py
=============
Pydantic v2 data models for all structured outputs produced by the
LLM-as-Judge evaluation pipeline.

Models:
  - CriterionScore       : Per-criterion score + step-by-step rationale
  - PointwiseVerdict     : Full rubric breakdown for a single response
  - PairwiseVerdict      : A-vs-B comparison verdict
  - AuditLogEntry        : Complete audit record of one judge call
  - TestCaseResult       : Wraps a verdict with its source test-case metadata
  - SuiteReport          : Aggregated statistics for a full test suite run
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Core scoring primitives
# ---------------------------------------------------------------------------


class CriterionScore(BaseModel):
    """Score and step-by-step rationale for one evaluation criterion."""

    score: int = Field(
        ...,
        ge=1,
        le=5,
        description="Score from 1 (poor) to 5 (excellent) on this criterion.",
    )
    rationale: str = Field(
        ...,
        min_length=10,
        description=(
            "Step-by-step evidence and reasoning that supports the assigned score. "
            "Must be written BEFORE the score is assigned (chain-of-thought)."
        ),
    )


# ---------------------------------------------------------------------------
# Pointwise verdict
# ---------------------------------------------------------------------------


class PointwiseVerdict(BaseModel):
    """
    Structured verdict for a single model output evaluated against a rubric.
    Used in pointwise and reference-based evaluation modes.
    """

    criteria_breakdown: Dict[str, CriterionScore] = Field(
        ...,
        description=(
            "Per-criterion scores. Keys must include at least: "
            "correctness, faithfulness, completeness, "
            "instruction_following, tone, safety."
        ),
    )
    overall_score: float = Field(
        ...,
        ge=1.0,
        le=5.0,
        description="Weighted aggregate score across all criteria (1.0–5.0).",
    )
    overall_rationale: str = Field(
        ...,
        min_length=20,
        description="Summary judgment of the output quality, referencing criterion scores.",
    )
    passed: bool = Field(
        ...,
        description="True if overall_score meets the minimum release threshold.",
    )
    verbosity_penalty_applied: bool = Field(
        default=False,
        description="True if a verbosity penalty was deducted from the score.",
    )

    @field_validator("criteria_breakdown")
    @classmethod
    def validate_required_criteria(
        cls, v: Dict[str, CriterionScore]
    ) -> Dict[str, CriterionScore]:
        required = {
            "correctness",
            "faithfulness",
            "completeness",
            "instruction_following",
            "tone",
            "safety",
        }
        missing = required - set(v.keys())
        if missing:
            raise ValueError(f"Missing required criteria in breakdown: {missing}")
        return v

    @model_validator(mode="after")
    def overall_score_must_be_consistent(self) -> "PointwiseVerdict":
        """Warn if overall_score is wildly inconsistent with per-criterion average."""
        if self.criteria_breakdown:
            avg = sum(c.score for c in self.criteria_breakdown.values()) / len(
                self.criteria_breakdown
            )
            # Allow up to 1.5 deviation (weights may differ from simple average)
            if abs(self.overall_score - avg) > 1.5:
                raise ValueError(
                    f"overall_score ({self.overall_score:.2f}) deviates too far from "
                    f"per-criterion average ({avg:.2f}). Check weighting logic."
                )
        return self


# ---------------------------------------------------------------------------
# Pairwise verdict
# ---------------------------------------------------------------------------


class PairwiseVerdict(BaseModel):
    """
    Structured verdict for A-vs-B pairwise comparison.
    Used in pairwise evaluation mode.
    """

    winner: Literal["Model_A", "Model_B", "Tie"] = Field(
        ...,
        description="Declared winner of the comparison, or 'Tie' if equally matched.",
    )
    rationale: str = Field(
        ...,
        min_length=20,
        description=(
            "Comparative justification focusing on key differences between responses. "
            "Chain-of-thought reasoning must precede the verdict."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Judge certainty score (0.0 = completely uncertain, 1.0 = fully certain).",
    )
    score_a: Optional[float] = Field(
        default=None,
        ge=1.0,
        le=5.0,
        description="Optional absolute quality score for Model A (1–5).",
    )
    score_b: Optional[float] = Field(
        default=None,
        ge=1.0,
        le=5.0,
        description="Optional absolute quality score for Model B (1–5).",
    )


# ---------------------------------------------------------------------------
# Audit log entry
# ---------------------------------------------------------------------------


class TokenUsage(BaseModel):
    """Token consumption tracking for a single API call."""

    prompt_tokens: int = Field(..., ge=0)
    completion_tokens: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)
    estimated_cost_usd: float = Field(
        ..., ge=0.0, description="Estimated USD cost based on model pricing."
    )


class AuditLogEntry(BaseModel):
    """
    Immutable record of a single judge API call.
    Captures everything needed for full reproducibility.
    """

    run_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this evaluation run.",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
        description="UTC timestamp of when the judge call was made.",
    )
    test_case_id: str = Field(..., description="ID of the test case being evaluated.")
    evaluation_mode: Literal["pointwise", "pairwise", "reference_based"] = Field(
        ..., description="Which evaluation mode was used."
    )
    judge_model: str = Field(..., description="Model name/identifier used as judge.")
    system_prompt: str = Field(..., description="Full system prompt sent to the judge.")
    user_prompt: str = Field(..., description="Full user prompt sent to the judge.")
    raw_response: str = Field(..., description="Unmodified raw text response from the judge API.")
    parsed_verdict: Optional[Dict[str, Any]] = Field(
        default=None, description="Successfully parsed and validated verdict as dict."
    )
    parse_error: Optional[str] = Field(
        default=None, description="Error message if JSON parsing failed (after all retries)."
    )
    retry_count: int = Field(
        default=0, ge=0, description="Number of retry attempts before successful parse."
    )
    token_usage: Optional[TokenUsage] = Field(
        default=None, description="Token usage and cost tracking."
    )
    latency_ms: float = Field(
        ..., ge=0.0, description="Total wall-clock latency in milliseconds."
    )
    temperature: float = Field(
        ..., ge=0.0, le=2.0, description="Temperature setting used for this call."
    )


# ---------------------------------------------------------------------------
# Test case result (wrapper)
# ---------------------------------------------------------------------------


class TestCaseResult(BaseModel):
    """
    Full result for one test case, combining verdict with source metadata.
    """

    test_case_id: str
    evaluation_mode: Literal["pointwise", "pairwise", "reference_based"]
    verdict: Optional[Dict[str, Any]] = None  # serialized PointwiseVerdict or PairwiseVerdict
    audit_log: AuditLogEntry
    position_consistent: Optional[bool] = Field(
        default=None,
        description="For pairwise mode: whether forward and reversed verdicts agree.",
    )
    forward_winner: Optional[str] = None
    reverse_winner: Optional[str] = None
    final_winner: Optional[str] = Field(
        default=None,
        description="Final winner after position bias reconciliation.",
    )


# ---------------------------------------------------------------------------
# Suite report
# ---------------------------------------------------------------------------


class CriterionStats(BaseModel):
    """Aggregated statistics for a single criterion across the suite."""

    criterion: str
    mean_score: float
    min_score: float
    max_score: float
    std_score: float


class SuiteReport(BaseModel):
    """
    Macro-level aggregated report for a complete test suite evaluation run.
    """

    suite_name: str
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )
    evaluation_mode: Literal["pointwise", "pairwise", "reference_based"]
    total_cases: int
    passed_cases: int
    failed_cases: int
    pass_rate: float = Field(..., ge=0.0, le=1.0)
    mean_overall_score: float
    std_overall_score: float
    criterion_stats: List[CriterionStats] = Field(default_factory=list)

    # Pairwise-specific metrics
    win_count_a: Optional[int] = None
    win_count_b: Optional[int] = None
    tie_count: Optional[int] = None
    flip_rate: Optional[float] = Field(
        default=None,
        description="Proportion of cases where swapping order changed the winner (position bias metric).",
    )

    # Cost tracking
    total_tokens_used: Optional[int] = None
    total_cost_usd: Optional[float] = None
    total_latency_ms: Optional[float] = None
