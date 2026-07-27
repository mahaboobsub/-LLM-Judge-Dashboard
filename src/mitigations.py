"""
src/mitigations.py
==================
Bias detection and code-level mitigation strategies.

Implements mitigations for the five core LLM judge biases:
  1. Position Bias     — double-pass order swapping + flip rate tracking
  2. Verbosity Bias    — padded-answer probe + rubric penalty enforcement
  3. Self-Enhancement  — model family cross-check (advisory)
  4. Sycophancy/Style  — chain-of-thought enforcement + confidently-wrong probes
  5. Score Clustering  — few-shot anchors (in prompts.py) + distribution analysis

Functions:
  evaluate_pairwise_unbiased()   — double-pass pairwise with bias check
  calculate_flip_rate()          — aggregate flip rate across a suite
  run_verbosity_probe()          — padded vs. terse probe test
  run_sycophancy_probe()         — confidently-wrong probe test
  check_self_enhancement_risk()  — advisory check for model family match
  analyze_score_distribution()   — detect score clustering in a results list
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.schema import AuditLogEntry, PairwiseVerdict, PointwiseVerdict


# ---------------------------------------------------------------------------
# 1. Position Bias Mitigation — Double-Pass Order Swapping
# ---------------------------------------------------------------------------

WINNER_SWAP_MAP = {
    "Model_A": "Model_B",
    "Model_B": "Model_A",
    "Tie": "Tie",
}


def evaluate_pairwise_unbiased(
    judge_client: Any,
    test_case: Dict[str, Any],
    response_a: str,
    response_b: str,
) -> Dict[str, Any]:
    """
    Run pairwise evaluation twice with swapped order to detect position bias.

    Pass 1: Judge sees (Model_A=response_a, Model_B=response_b)
    Pass 2: Judge sees (Model_A=response_b, Model_B=response_a)

    If the winner changes after accounting for the swap, the result is
    marked as positionally inconsistent and resolved to "Tie".

    Parameters
    ----------
    judge_client : JudgeClient
        Instantiated judge client.
    test_case : dict
        The test case dict (must contain 'id', 'input').
    response_a, response_b : str
        The two candidate responses.

    Returns
    -------
    dict with keys:
        final_winner          : str   — "Model_A", "Model_B", or "Tie (...)"
        position_consistent   : bool  — True if both passes agree
        forward_winner        : str   — Winner from pass 1 (original order)
        reverse_winner_mapped : str   — Winner from pass 2 (mapped back to original labels)
        confidence_avg        : float — Average confidence across both passes
        forward_audit         : AuditLogEntry
        reverse_audit         : AuditLogEntry
    """
    # --- Pass 1: Original order (A, B) ---
    verdict_forward, audit_forward = judge_client.judge_pairwise(
        test_case=test_case,
        response_a=response_a,
        response_b=response_b,
    )

    # --- Pass 2: Swapped order (B presented as A, A presented as B) ---
    verdict_reverse, audit_reverse = judge_client.judge_pairwise(
        test_case=test_case,
        response_a=response_b,   # B now in position A
        response_b=response_a,   # A now in position B
    )

    # --- Map swapped result back to original labels ---
    forward_winner = verdict_forward.winner if verdict_forward else "PARSE_ERROR"
    raw_reverse_winner = verdict_reverse.winner if verdict_reverse else "PARSE_ERROR"
    reverse_winner_mapped = WINNER_SWAP_MAP.get(raw_reverse_winner, "Tie")

    # --- Position consistency check ---
    is_consistent = forward_winner == reverse_winner_mapped

    if is_consistent:
        final_winner = forward_winner
    else:
        final_winner = "Tie (Position Inconsistency)"

    # --- Average confidence (if both verdicts available) ---
    confidences = []
    if verdict_forward:
        confidences.append(verdict_forward.confidence)
    if verdict_reverse:
        confidences.append(verdict_reverse.confidence)
    confidence_avg = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "final_winner": final_winner,
        "position_consistent": is_consistent,
        "forward_winner": forward_winner,
        "reverse_winner_mapped": reverse_winner_mapped,
        "confidence_avg": confidence_avg,
        "forward_audit": audit_forward,
        "reverse_audit": audit_reverse,
        "forward_verdict": verdict_forward.model_dump() if verdict_forward else None,
        "reverse_verdict": verdict_reverse.model_dump() if verdict_reverse else None,
    }


def calculate_flip_rate(results: List[Dict[str, Any]]) -> float:
    """
    Compute the position bias flip rate across a list of pairwise results.

    Flip Rate = (# cases where position_consistent is False) / (total cases)

    Parameters
    ----------
    results : list of dicts returned by evaluate_pairwise_unbiased()

    Returns
    -------
    float : flip rate (0.0 = no position bias, 1.0 = always inconsistent)
    """
    if not results:
        return 0.0

    inconsistent_count = sum(
        1 for r in results if not r.get("position_consistent", True)
    )
    return inconsistent_count / len(results)


# ---------------------------------------------------------------------------
# 2. Verbosity Bias Probe
# ---------------------------------------------------------------------------

def run_verbosity_probe(
    judge_client: Any,
    probe_case: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Test whether the judge penalizes verbose but correct answers appropriately.

    A well-calibrated judge should NOT reward a padded response over a
    terse response when both contain identical correct information.

    Parameters
    ----------
    judge_client : JudgeClient
    probe_case : dict from adversarial_probes.json with probe_type='verbosity'
        Expected keys: id, input, system_prompt, response_terse, response_padded,
                       expected_result (e.g., "terse_wins_or_tie")

    Returns
    -------
    dict with:
        probe_id         : str
        probe_passed     : bool   — True if judge did NOT reward verbosity
        terse_score      : float
        padded_score     : float
        result           : str    — "terse_wins", "padded_wins", "tie"
        expected_result  : str
        terse_audit      : AuditLogEntry
        padded_audit     : AuditLogEntry
    """
    # Build minimal test case wrapper
    tc_terse = {
        "id": f"{probe_case['id']}_terse",
        "input": probe_case["input"],
        "system_prompt": probe_case.get("system_prompt", ""),
    }
    tc_padded = {
        "id": f"{probe_case['id']}_padded",
        "input": probe_case["input"],
        "system_prompt": probe_case.get("system_prompt", ""),
    }

    verdict_terse, audit_terse = judge_client.judge_pointwise(
        test_case=tc_terse,
        response=probe_case["response_terse"],
    )
    verdict_padded, audit_padded = judge_client.judge_pointwise(
        test_case=tc_padded,
        response=probe_case["response_padded"],
    )

    terse_score = verdict_terse.overall_score if verdict_terse else 0.0
    padded_score = verdict_padded.overall_score if verdict_padded else 0.0

    if terse_score > padded_score:
        result = "terse_wins"
    elif padded_score > terse_score:
        result = "padded_wins"
    else:
        result = "tie"

    expected = probe_case.get("expected_result", "terse_wins_or_tie")
    if expected == "terse_wins_or_tie":
        probe_passed = result in ("terse_wins", "tie")
    else:
        probe_passed = result == expected

    return {
        "probe_id": probe_case["id"],
        "probe_type": "verbosity",
        "probe_passed": probe_passed,
        "terse_score": terse_score,
        "padded_score": padded_score,
        "result": result,
        "expected_result": expected,
        "terse_audit": audit_terse,
        "padded_audit": audit_padded,
    }


# ---------------------------------------------------------------------------
# 3. Sycophancy / Style Bias Probe
# ---------------------------------------------------------------------------

def run_sycophancy_probe(
    judge_client: Any,
    probe_case: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Test whether the judge is fooled by confident, polished but wrong responses.

    Parameters
    ----------
    judge_client : JudgeClient
    probe_case : dict from adversarial_probes.json with probe_type='sycophancy'
        Expected keys: id, input, system_prompt, response_confident_wrong,
                       response_hesitant_correct (or response_plain_correct),
                       expected_winner

    Returns
    -------
    dict with:
        probe_id          : str
        probe_passed      : bool  — True if judge correctly identified the correct answer
        declared_winner   : str
        expected_winner   : str
        pairwise_result   : dict
    """
    # Use whichever "correct" response key is present
    correct_key = "response_hesitant_correct" if "response_hesitant_correct" in probe_case \
        else "response_plain_correct" if "response_plain_correct" in probe_case \
        else "response_correct"

    tc = {
        "id": probe_case["id"],
        "input": probe_case["input"],
        "system_prompt": probe_case.get("system_prompt", ""),
    }

    # Run unbiased pairwise: confident_wrong vs. correct
    result = evaluate_pairwise_unbiased(
        judge_client=judge_client,
        test_case=tc,
        response_a=probe_case["response_confident_wrong"],
        response_b=probe_case[correct_key],
    )

    # "Model_B" = correct response; "Model_A" = confident wrong response
    final_winner = result["final_winner"]
    expected = probe_case.get("expected_winner", "hesitant_correct")

    # Map expected_winner labels to Model_A/Model_B
    # response_a = confident_wrong = Model_A
    # response_b = correct        = Model_B
    expected_mapped = "Model_B"  # correct response is always placed in B
    probe_passed = final_winner in ("Model_B", "Tie (Position Inconsistency)") \
        if "wrong" in probe_case.get("response_confident_wrong", "").lower() \
        else final_winner == expected_mapped

    # Simpler: correct response is Model_B; judge should pick Model_B
    probe_passed = final_winner == "Model_B"

    return {
        "probe_id": probe_case["id"],
        "probe_type": "sycophancy",
        "probe_passed": probe_passed,
        "declared_winner": final_winner,
        "expected_winner": "Model_B (correct response)",
        "pairwise_result": {
            k: v for k, v in result.items()
            if k not in ("forward_audit", "reverse_audit")
        },
    }


# ---------------------------------------------------------------------------
# 4. Self-Enhancement Bias Check (Advisory)
# ---------------------------------------------------------------------------

# Known model family groups
MODEL_FAMILIES: Dict[str, str] = {
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "gpt-3.5-turbo": "openai",
    "gpt-4": "openai",
    "claude-3-5-sonnet": "anthropic",
    "claude-3-haiku": "anthropic",
    "claude-3-opus": "anthropic",
    "gemini-pro": "google",
    "gemini-1.5-pro": "google",
    "llama-3": "meta",
    "llama-2": "meta",
    "mistral": "mistral",
}


def check_self_enhancement_risk(
    judge_model: str,
    candidate_model_a: str,
    candidate_model_b: str,
) -> Dict[str, Any]:
    """
    Advisory check: warn if judge and candidate models share the same family.

    Best practice: use a judge from a DIFFERENT model family than the candidates.

    Returns
    -------
    dict with:
        risk_detected  : bool
        judge_family   : str
        candidate_families : dict
        recommendation : str
    """
    def get_family(model_name: str) -> str:
        for key, family in MODEL_FAMILIES.items():
            if key.lower() in model_name.lower():
                return family
        return "unknown"

    judge_family = get_family(judge_model)
    family_a = get_family(candidate_model_a)
    family_b = get_family(candidate_model_b)

    risk_a = judge_family == family_a and judge_family != "unknown"
    risk_b = judge_family == family_b and judge_family != "unknown"
    risk_detected = risk_a or risk_b

    recommendation = (
        "⚠️  RISK: Judge and at least one candidate share the same model family. "
        "Consider using a judge from a different provider to avoid self-enhancement bias."
        if risk_detected
        else "✅  No self-enhancement risk detected. Judge and candidates are from different families."
    )

    return {
        "risk_detected": risk_detected,
        "judge_family": judge_family,
        "candidate_families": {"model_a": family_a, "model_b": family_b},
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# 5. Score Clustering Analysis
# ---------------------------------------------------------------------------

def analyze_score_distribution(
    scores: List[float],
    scale_min: float = 1.0,
    scale_max: float = 5.0,
    clustering_threshold: float = 0.7,
) -> Dict[str, Any]:
    """
    Detect score clustering: if >threshold proportion of scores fall in a
    single 1-point band, the judge may be under-utilizing the scale.

    Parameters
    ----------
    scores : list of float — overall_score values from a suite run
    scale_min, scale_max : float — expected scale bounds
    clustering_threshold : float — proportion that triggers a clustering warning

    Returns
    -------
    dict with:
        score_count       : int
        mean_score        : float
        std_score         : float
        score_range_used  : tuple[float, float]
        is_clustered      : bool
        dominant_band     : str | None  — e.g., "3.0–4.0"
        dominant_band_pct : float
        recommendation    : str
    """
    import statistics

    if not scores:
        return {"score_count": 0, "is_clustered": False, "recommendation": "No scores to analyze."}

    mean_s = statistics.mean(scores)
    std_s = statistics.stdev(scores) if len(scores) > 1 else 0.0
    min_s = min(scores)
    max_s = max(scores)

    # Count scores in each 1-point band
    bands = {}
    for low in range(int(scale_min), int(scale_max)):
        high = low + 1
        key = f"{low}.0–{high}.0"
        count = sum(1 for s in scores if low <= s < high)
        bands[key] = count
    # Include the max value in the last band
    last_key = f"{int(scale_max) - 1}.0–{int(scale_max)}.0"
    bands[last_key] += sum(1 for s in scores if s == scale_max)

    dominant_band = max(bands, key=bands.__getitem__)
    dominant_count = bands[dominant_band]
    dominant_pct = dominant_count / len(scores)

    is_clustered = dominant_pct >= clustering_threshold

    recommendation = (
        f"⚠️  Score clustering detected! {dominant_pct:.1%} of scores fall in band {dominant_band}. "
        "Consider adding more diverse few-shot anchor examples or adjusting the rubric."
        if is_clustered
        else f"✅  Score distribution appears healthy (std={std_s:.2f}, range={min_s:.1f}–{max_s:.1f})."
    )

    return {
        "score_count": len(scores),
        "mean_score": round(mean_s, 3),
        "std_score": round(std_s, 3),
        "score_range_used": (min_s, max_s),
        "band_distribution": bands,
        "is_clustered": is_clustered,
        "dominant_band": dominant_band,
        "dominant_band_pct": round(dominant_pct, 4),
        "recommendation": recommendation,
    }
