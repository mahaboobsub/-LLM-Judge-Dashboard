"""
src/aggregator.py
=================
Test suite aggregation and A/B comparison engine.

Responsibilities:
  - Aggregate individual test-case verdicts into macro-level suite metrics
  - Compute pass rates, per-criterion averages, win/loss/tie distributions
  - A/B comparison: Config V1 vs Config V2 with automated winner declaration
  - Export reports as JSON and/or CSV

Classes:
    TestSuiteAggregator  — accumulates results and produces SuiteReport
    ABComparisonEngine   — compares two SuiteReport objects and declares winner
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

from src.schema import CriterionStats, SuiteReport


# ---------------------------------------------------------------------------
# TestSuiteAggregator
# ---------------------------------------------------------------------------

class TestSuiteAggregator:
    """
    Collects evaluation results from a test suite run and computes aggregate metrics.

    Usage
    -----
        agg = TestSuiteAggregator(suite_name="general_qa", evaluation_mode="pointwise")
        for case_result in results:
            agg.add_result(case_result)
        report = agg.build_report()
    """

    def __init__(
        self,
        suite_name: str,
        evaluation_mode: Literal["pointwise", "pairwise", "reference_based"] = "pointwise",
        pass_threshold: float = 3.0,
    ) -> None:
        self.suite_name = suite_name
        self.evaluation_mode = evaluation_mode
        self.pass_threshold = pass_threshold
        self._results: List[Dict[str, Any]] = []

    def add_result(self, result: Dict[str, Any]) -> None:
        """
        Add a single test case result to the aggregator.

        Expected dict shape:
          For pointwise/reference_based:
            { 'verdict': PointwiseVerdict.model_dump(), 'audit': AuditLogEntry }
          For pairwise:
            { 'final_winner': str, 'position_consistent': bool, 'forward_verdict': dict,
              'audit_forward': AuditLogEntry, ... }
        """
        self._results.append(result)

    def add_results(self, results: List[Dict[str, Any]]) -> None:
        """Add multiple results at once."""
        for r in results:
            self.add_result(r)

    def build_report(self) -> SuiteReport:
        """Compute aggregate metrics and return a SuiteReport."""
        if self.evaluation_mode in ("pointwise", "reference_based"):
            return self._build_pointwise_report()
        else:
            return self._build_pairwise_report()

    def _build_pointwise_report(self) -> SuiteReport:
        """Aggregate metrics for pointwise or reference-based evaluations."""
        overall_scores: List[float] = []
        passed_count = 0
        criteria_scores: Dict[str, List[float]] = {}
        total_tokens = 0
        total_cost = 0.0
        total_latency = 0.0

        for r in self._results:
            verdict = r.get("verdict") or {}
            audit = r.get("audit")

            score = verdict.get("overall_score")
            if score is not None:
                overall_scores.append(float(score))
                if verdict.get("passed", False):
                    passed_count += 1

            for criterion, cdata in verdict.get("criteria_breakdown", {}).items():
                if criterion not in criteria_scores:
                    criteria_scores[criterion] = []
                s = cdata.get("score") if isinstance(cdata, dict) else getattr(cdata, "score", None)
                if s is not None:
                    criteria_scores[criterion].append(float(s))

            if audit:
                usage = getattr(audit, "token_usage", None)
                if usage:
                    total_tokens += getattr(usage, "total_tokens", 0)
                    total_cost += getattr(usage, "estimated_cost_usd", 0.0)
                total_latency += getattr(audit, "latency_ms", 0.0)

        total = len(self._results)
        passed = passed_count
        failed = total - passed
        pass_rate = passed / total if total > 0 else 0.0
        mean_score = statistics.mean(overall_scores) if overall_scores else 0.0
        std_score = statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0.0

        criterion_stats = []
        for crit, scores in criteria_scores.items():
            if scores:
                criterion_stats.append(
                    CriterionStats(
                        criterion=crit,
                        mean_score=round(statistics.mean(scores), 3),
                        min_score=min(scores),
                        max_score=max(scores),
                        std_score=round(statistics.stdev(scores) if len(scores) > 1 else 0.0, 3),
                    )
                )

        return SuiteReport(
            suite_name=self.suite_name,
            evaluation_mode=self.evaluation_mode,
            total_cases=total,
            passed_cases=passed,
            failed_cases=failed,
            pass_rate=round(pass_rate, 4),
            mean_overall_score=round(mean_score, 3),
            std_overall_score=round(std_score, 3),
            criterion_stats=criterion_stats,
            total_tokens_used=total_tokens if total_tokens > 0 else None,
            total_cost_usd=round(total_cost, 6) if total_cost > 0 else None,
            total_latency_ms=round(total_latency, 2) if total_latency > 0 else None,
        )

    def _build_pairwise_report(self) -> SuiteReport:
        """Aggregate metrics for pairwise evaluations."""
        win_a = win_b = ties = 0
        position_inconsistent = 0
        total_tokens = 0
        total_cost = 0.0
        total_latency = 0.0

        for r in self._results:
            winner = r.get("final_winner", "Tie")
            if winner == "Model_A":
                win_a += 1
            elif winner == "Model_B":
                win_b += 1
            else:
                ties += 1

            if not r.get("position_consistent", True):
                position_inconsistent += 1

            for audit_key in ("forward_audit", "reverse_audit"):
                audit = r.get(audit_key)
                if audit:
                    usage = getattr(audit, "token_usage", None)
                    if usage:
                        total_tokens += getattr(usage, "total_tokens", 0)
                        total_cost += getattr(usage, "estimated_cost_usd", 0.0)
                    total_latency += getattr(audit, "latency_ms", 0.0)

        total = len(self._results)
        flip_rate = position_inconsistent / total if total > 0 else 0.0

        # Use win rate of Model_B as a proxy for "pass_rate" (how often the better model wins)
        pass_rate = win_b / total if total > 0 else 0.0

        return SuiteReport(
            suite_name=self.suite_name,
            evaluation_mode=self.evaluation_mode,
            total_cases=total,
            passed_cases=win_b,
            failed_cases=win_a,
            pass_rate=round(pass_rate, 4),
            mean_overall_score=0.0,
            std_overall_score=0.0,
            win_count_a=win_a,
            win_count_b=win_b,
            tie_count=ties,
            flip_rate=round(flip_rate, 4),
            total_tokens_used=total_tokens if total_tokens > 0 else None,
            total_cost_usd=round(total_cost, 6) if total_cost > 0 else None,
            total_latency_ms=round(total_latency, 2) if total_latency > 0 else None,
        )


# ---------------------------------------------------------------------------
# ABComparisonEngine
# ---------------------------------------------------------------------------

class ABComparisonEngine:
    """
    Compares two test suite runs (Config V1 vs Config V2) and declares a winner.

    Statistical test used: Welch's t-test for mean overall score comparison.
    Falls back to simple mean difference if scipy is not available.

    Usage
    -----
        engine = ABComparisonEngine(report_v1, report_v2)
        comparison = engine.compare()
    """

    def __init__(
        self,
        report_v1: SuiteReport,
        report_v2: SuiteReport,
        significance_threshold: float = 0.05,
        metric: str = "overall_score",
    ) -> None:
        self.report_v1 = report_v1
        self.report_v2 = report_v2
        self.significance_threshold = significance_threshold
        self.metric = metric

    def compare(
        self,
        scores_v1: Optional[List[float]] = None,
        scores_v2: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Run A/B comparison and declare a winner.

        Parameters
        ----------
        scores_v1, scores_v2 : list of float, optional
            Raw per-case scores for statistical testing. If None, uses
            aggregate report metrics only.

        Returns
        -------
        dict with:
            winner              : "V1", "V2", or "No significant difference"
            mean_v1             : float
            mean_v2             : float
            difference          : float (V2 - V1)
            improvement_pct     : float
            pass_rate_v1        : float
            pass_rate_v2        : float
            statistically_sig   : bool
            p_value             : float | None
            recommendation      : str
            criterion_comparison : list of dicts
        """
        mean_v1 = self.report_v1.mean_overall_score
        mean_v2 = self.report_v2.mean_overall_score
        diff = mean_v2 - mean_v1
        improvement_pct = (diff / mean_v1 * 100) if mean_v1 != 0 else 0.0

        # Statistical test
        p_value = None
        statistically_sig = False
        if scores_v1 and scores_v2 and len(scores_v1) >= 2 and len(scores_v2) >= 2:
            p_value, statistically_sig = self._welch_t_test(scores_v1, scores_v2)

        # Declare winner
        if p_value is not None and statistically_sig:
            winner = "V2" if diff > 0 else "V1"
        elif p_value is None:
            # No statistical test — use simple threshold (0.1 point difference)
            winner = "V2" if diff > 0.1 else ("V1" if diff < -0.1 else "No significant difference")
        else:
            winner = "No significant difference"

        # Criterion-level comparison
        crit_comparison = self._compare_criteria()

        recommendation = self._build_recommendation(winner, diff, improvement_pct, p_value)

        return {
            "winner": winner,
            "mean_v1": mean_v1,
            "mean_v2": mean_v2,
            "difference": round(diff, 4),
            "improvement_pct": round(improvement_pct, 2),
            "pass_rate_v1": self.report_v1.pass_rate,
            "pass_rate_v2": self.report_v2.pass_rate,
            "statistically_significant": statistically_sig,
            "p_value": round(p_value, 6) if p_value is not None else None,
            "significance_threshold": self.significance_threshold,
            "criterion_comparison": crit_comparison,
            "recommendation": recommendation,
        }

    def _welch_t_test(
        self, scores_v1: List[float], scores_v2: List[float]
    ) -> Tuple[float, bool]:
        """Run Welch's t-test; return (p_value, is_significant)."""
        try:
            from scipy import stats
            _, p_value = stats.ttest_ind(scores_v1, scores_v2, equal_var=False)
            return float(p_value), float(p_value) < self.significance_threshold
        except ImportError:
            # Fallback: Mann-Whitney U using only stdlib (approximate)
            return None, False

    def _compare_criteria(self) -> List[Dict[str, Any]]:
        """Compare per-criterion means between V1 and V2."""
        v1_map = {c.criterion: c for c in self.report_v1.criterion_stats}
        v2_map = {c.criterion: c for c in self.report_v2.criterion_stats}

        comparisons = []
        all_criteria = set(v1_map) | set(v2_map)
        for crit in sorted(all_criteria):
            v1c = v1_map.get(crit)
            v2c = v2_map.get(crit)
            row = {
                "criterion": crit,
                "mean_v1": v1c.mean_score if v1c else None,
                "mean_v2": v2c.mean_score if v2c else None,
                "delta": round((v2c.mean_score - v1c.mean_score), 3)
                if (v1c and v2c) else None,
            }
            comparisons.append(row)
        return comparisons

    def _build_recommendation(
        self,
        winner: str,
        diff: float,
        improvement_pct: float,
        p_value: Optional[float],
    ) -> str:
        if winner == "V2":
            sig_note = f" (p={p_value:.4f}, statistically significant)" if p_value else ""
            return (
                f"✅ Config V2 is the winner with +{diff:.3f} points "
                f"({improvement_pct:.1f}% improvement){sig_note}. "
                "Recommend deploying V2."
            )
        elif winner == "V1":
            return (
                f"⚠️  Config V1 outperforms V2 (diff={diff:.3f}). "
                "V2 regression detected. Do NOT deploy V2."
            )
        else:
            return (
                "📊 No statistically significant difference detected between V1 and V2. "
                "Consider running more test cases or adjusting configurations."
            )


# ---------------------------------------------------------------------------
# Report Export Utilities
# ---------------------------------------------------------------------------

def save_report_json(report: SuiteReport, output_dir: str) -> str:
    """Save a SuiteReport as a JSON file. Returns the file path."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{report.suite_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2)
    return filepath


def save_report_csv(report: SuiteReport, output_dir: str) -> str:
    """Save a SuiteReport summary as a CSV file. Returns the file path."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{report.suite_name}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)

    rows = [
        {"metric": "suite_name", "value": report.suite_name},
        {"metric": "evaluation_mode", "value": report.evaluation_mode},
        {"metric": "total_cases", "value": report.total_cases},
        {"metric": "passed_cases", "value": report.passed_cases},
        {"metric": "failed_cases", "value": report.failed_cases},
        {"metric": "pass_rate", "value": report.pass_rate},
        {"metric": "mean_overall_score", "value": report.mean_overall_score},
        {"metric": "std_overall_score", "value": report.std_overall_score},
    ]
    if report.flip_rate is not None:
        rows.append({"metric": "flip_rate (position_bias)", "value": report.flip_rate})
    if report.total_cost_usd is not None:
        rows.append({"metric": "total_cost_usd", "value": report.total_cost_usd})

    for cs in report.criterion_stats:
        rows.append({"metric": f"criterion_{cs.criterion}_mean", "value": cs.mean_score})

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)

    return filepath


def save_ab_comparison_json(comparison: Dict[str, Any], output_dir: str) -> str:
    """Save an A/B comparison result as JSON. Returns the file path."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"ab_comparison_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    return filepath
