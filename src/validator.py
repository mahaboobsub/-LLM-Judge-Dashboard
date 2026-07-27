"""
src/validator.py
================
Judge validation module.

Implements the three validation checks from Step 6:
  1. Human/Gold Agreement  — Cohen's Kappa + exact match accuracy
  2. Test-Retest Consistency — same suite run twice, measure stability
  3. Adversarial Probe Suite — probe pass/fail rates for the judge itself

Functions:
    calculate_judge_accuracy_and_kappa()  — Kappa and exact agreement
    run_consistency_test()                — test-retest at T > 0
    run_adversarial_suite()               — probe suite pass rates
    generate_validation_report()          — consolidate all three into a dict
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import cohen_kappa_score


# ---------------------------------------------------------------------------
# 1. Human / Gold Agreement
# ---------------------------------------------------------------------------

def calculate_judge_accuracy_and_kappa(
    human_scores: List[int],
    judge_scores: List[int],
    weights: str = "quadratic",
) -> Dict[str, Any]:
    """
    Compute statistical agreement between human labels and LLM judge ratings.

    Parameters
    ----------
    human_scores : list of int  — Human annotator integer scores (e.g., 1–5)
    judge_scores : list of int  — LLM judge integer scores (e.g., 1–5)
    weights      : str          — 'quadratic' for ordinal scale alignment
                                  (quadratic penalizes larger disagreements more)

    Returns
    -------
    dict with:
        exact_agreement_rate : float — Proportion of exact matches
        cohens_kappa         : float — Quadratic weighted Cohen's Kappa
        near_agreement_rate  : float — Within-1-point agreement rate
        sample_size          : int
        interpretation       : str   — Human-readable kappa interpretation
    """
    if len(human_scores) != len(judge_scores):
        raise ValueError(
            f"Length mismatch: human_scores has {len(human_scores)} items "
            f"but judge_scores has {len(judge_scores)} items."
        )
    if len(human_scores) < 2:
        raise ValueError("Need at least 2 samples to compute agreement metrics.")

    y_human = np.array(human_scores)
    y_judge = np.array(judge_scores)

    # Exact agreement
    exact_match = float(np.mean(y_human == y_judge))

    # Within-1-point agreement
    near_match = float(np.mean(np.abs(y_human - y_judge) <= 1))

    # Cohen's Kappa (quadratic weighted)
    kappa = float(cohen_kappa_score(y_human, y_judge, weights=weights))

    # Interpretation
    interpretation = _interpret_kappa(kappa)

    return {
        "exact_agreement_rate": round(exact_match, 4),
        "near_agreement_rate": round(near_match, 4),
        "cohens_kappa": round(kappa, 4),
        "kappa_weights": weights,
        "sample_size": len(human_scores),
        "interpretation": interpretation,
    }


def _interpret_kappa(kappa: float) -> str:
    """Return a human-readable interpretation of Cohen's Kappa."""
    if kappa < 0:
        return "Poor (less than chance agreement)"
    elif kappa < 0.20:
        return "Slight agreement"
    elif kappa < 0.40:
        return "Fair agreement"
    elif kappa < 0.60:
        return "Moderate agreement"
    elif kappa < 0.80:
        return "Substantial agreement"
    else:
        return "Almost perfect agreement"


# ---------------------------------------------------------------------------
# 2. Test-Retest Consistency
# ---------------------------------------------------------------------------

def run_consistency_test(
    judge_client: Any,
    test_cases: List[Dict[str, Any]],
    responses: List[str],
    n_runs: int = 2,
    temperature: float = 0.3,
) -> Dict[str, Any]:
    """
    Evaluate the same test suite n_runs times under temperature > 0
    and measure how often scores remain identical across runs.

    Parameters
    ----------
    judge_client : JudgeClient (with temperature set to the desired value)
    test_cases   : list of test case dicts
    responses    : list of model responses (same length as test_cases)
    n_runs       : int — number of repeat evaluations
    temperature  : float — temperature setting used (informational only)

    Returns
    -------
    dict with:
        n_runs            : int
        temperature       : float
        total_cases       : int
        consistency_rate  : float — proportion of cases with identical score across all runs
        mean_score_std    : float — average std of per-case scores across runs
        per_case_results  : list of per-case consistency data
        recommendation    : str
    """
    if len(test_cases) != len(responses):
        raise ValueError("test_cases and responses must have the same length.")

    # Collect scores for each case across all runs
    per_case_scores: Dict[str, List[float]] = {tc["id"]: [] for tc in test_cases}

    for run_idx in range(n_runs):
        for tc, resp in zip(test_cases, responses):
            verdict, _ = judge_client.judge_pointwise(test_case=tc, response=resp)
            score = verdict.overall_score if verdict else None
            if score is not None:
                per_case_scores[tc["id"]].append(score)

    # Compute consistency metrics
    per_case_results = []
    all_stds = []
    consistent_count = 0

    for case_id, scores in per_case_scores.items():
        if len(scores) < 2:
            std = 0.0
            is_consistent = True
        else:
            std = float(np.std(scores))
            is_consistent = std == 0.0  # All runs produced identical score

        all_stds.append(std)
        if is_consistent:
            consistent_count += 1

        per_case_results.append({
            "case_id": case_id,
            "scores_across_runs": scores,
            "mean_score": round(float(np.mean(scores)), 3) if scores else None,
            "std_score": round(std, 4),
            "is_consistent": is_consistent,
        })

    total = len(test_cases)
    consistency_rate = consistent_count / total if total > 0 else 0.0
    mean_std = float(np.mean(all_stds)) if all_stds else 0.0

    recommendation = (
        f"✅ High consistency: {consistency_rate:.1%} of cases produced identical scores "
        f"(mean std={mean_std:.3f}) at T={temperature}."
        if consistency_rate >= 0.8
        else f"⚠️  Low consistency: only {consistency_rate:.1%} of cases were stable "
        f"(mean std={mean_std:.3f}) at T={temperature}. Consider lowering temperature "
        "or using a more deterministic judge model."
    )

    return {
        "n_runs": n_runs,
        "temperature": temperature,
        "total_cases": total,
        "consistent_cases": consistent_count,
        "consistency_rate": round(consistency_rate, 4),
        "mean_score_std": round(mean_std, 4),
        "per_case_results": per_case_results,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# 3. Adversarial Probe Suite
# ---------------------------------------------------------------------------

def run_adversarial_suite(
    judge_client: Any,
    probe_cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Run the full adversarial probe suite and record pass/fail rates.

    Dispatches each probe to the appropriate bias-specific runner:
      - verbosity  → mitigations.run_verbosity_probe()
      - sycophancy → mitigations.run_sycophancy_probe()
      - position_bias → mitigations.evaluate_pairwise_unbiased()
      - score_clustering → pointwise evaluation + range check

    Parameters
    ----------
    judge_client : JudgeClient
    probe_cases  : list of probe case dicts from adversarial_probes.json

    Returns
    -------
    dict with:
        total_probes      : int
        passed_probes     : int
        failed_probes     : int
        pass_rate         : float
        by_type           : dict mapping probe_type → {total, passed, failed, pass_rate}
        per_probe_results : list of per-probe dicts
        recommendation    : str
    """
    from src.mitigations import run_verbosity_probe, run_sycophancy_probe

    per_probe_results = []
    by_type: Dict[str, Dict[str, Any]] = {}

    for probe in probe_cases:
        probe_type = probe.get("probe_type", "unknown")
        result: Dict[str, Any] = {"probe_id": probe["id"], "probe_type": probe_type}

        try:
            if probe_type == "verbosity":
                r = run_verbosity_probe(judge_client, probe)
                result["probe_passed"] = r["probe_passed"]
                result["details"] = {k: v for k, v in r.items() if "audit" not in k}

            elif probe_type == "sycophancy":
                r = run_sycophancy_probe(judge_client, probe)
                result["probe_passed"] = r["probe_passed"]
                result["details"] = r

            elif probe_type == "position_bias":
                from src.mitigations import evaluate_pairwise_unbiased
                tc = {
                    "id": probe["id"],
                    "input": probe["input"],
                    "system_prompt": probe.get("system_prompt", ""),
                }
                r = evaluate_pairwise_unbiased(
                    judge_client=judge_client,
                    test_case=tc,
                    response_a=probe["response_a"],
                    response_b=probe["response_b"],
                )
                expected_winner = probe.get("expected_winner", "Model_B")
                result["probe_passed"] = r["final_winner"] == expected_winner
                result["details"] = {
                    k: v for k, v in r.items() if "audit" not in k
                }

            elif probe_type == "score_clustering":
                tc = {
                    "id": probe["id"],
                    "input": probe["input"],
                    "system_prompt": probe.get("system_prompt", ""),
                }
                verdict, _ = judge_client.judge_pointwise(
                    test_case=tc,
                    response=probe["response"],
                )
                score = verdict.overall_score if verdict else None
                expected_range = probe.get("expected_score_range", [1, 5])
                result["probe_passed"] = (
                    score is not None
                    and expected_range[0] <= score <= expected_range[1]
                )
                result["details"] = {
                    "judged_score": score,
                    "expected_range": expected_range,
                }
            else:
                result["probe_passed"] = None
                result["details"] = {"error": f"Unknown probe type: {probe_type}"}

        except Exception as e:
            result["probe_passed"] = False
            result["details"] = {"error": str(e)}

        # Aggregate by type
        if probe_type not in by_type:
            by_type[probe_type] = {"total": 0, "passed": 0, "failed": 0}
        by_type[probe_type]["total"] += 1
        if result.get("probe_passed"):
            by_type[probe_type]["passed"] += 1
        else:
            by_type[probe_type]["failed"] += 1

        per_probe_results.append(result)

    # Finalize by_type rates
    for ptype, counts in by_type.items():
        t = counts["total"]
        counts["pass_rate"] = round(counts["passed"] / t, 4) if t > 0 else 0.0

    total = len(probe_cases)
    passed = sum(1 for r in per_probe_results if r.get("probe_passed"))
    failed = total - passed
    overall_pass_rate = passed / total if total > 0 else 0.0

    recommendation = (
        f"✅ Judge passed {passed}/{total} adversarial probes ({overall_pass_rate:.1%}). "
        "The judge demonstrates good bias resistance."
        if overall_pass_rate >= 0.75
        else f"⚠️  Judge passed only {passed}/{total} probes ({overall_pass_rate:.1%}). "
        "Significant bias detected. Review rubric, prompts, and consider a different judge model."
    )

    return {
        "total_probes": total,
        "passed_probes": passed,
        "failed_probes": failed,
        "pass_rate": round(overall_pass_rate, 4),
        "by_type": by_type,
        "per_probe_results": per_probe_results,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# Consolidated Validation Report
# ---------------------------------------------------------------------------

def generate_validation_report(
    kappa_result: Optional[Dict[str, Any]] = None,
    consistency_result: Optional[Dict[str, Any]] = None,
    adversarial_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Consolidate all three validation checks into a single summary report.

    Parameters
    ----------
    kappa_result         : output of calculate_judge_accuracy_and_kappa()
    consistency_result   : output of run_consistency_test()
    adversarial_result   : output of run_adversarial_suite()

    Returns
    -------
    dict with overall_verdict and component results.
    """
    flags = []

    if kappa_result:
        kappa = kappa_result.get("cohens_kappa", 0)
        if kappa < 0.4:
            flags.append(f"Low kappa ({kappa:.3f}) — poor human agreement")

    if consistency_result:
        cr = consistency_result.get("consistency_rate", 0)
        if cr < 0.7:
            flags.append(f"Low consistency ({cr:.1%}) — judge is unstable")

    if adversarial_result:
        apr = adversarial_result.get("pass_rate", 0)
        if apr < 0.7:
            flags.append(f"Low adversarial pass rate ({apr:.1%}) — judge is biased")

    if not flags:
        overall_verdict = "✅ PASSED — Judge meets all quality thresholds."
    else:
        overall_verdict = "❌ FAILED — Issues detected:\n" + "\n".join(f"  • {f}" for f in flags)

    return {
        "overall_verdict": overall_verdict,
        "flags": flags,
        "kappa_validation": kappa_result,
        "consistency_validation": consistency_result,
        "adversarial_validation": adversarial_result,
    }
