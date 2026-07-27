"""
main.py
=======
CLI entrypoint for the LLM-as-Judge Evaluation Pipeline.

Subcommands:
    pointwise   — Evaluate Model A responses on the test suite (pointwise)
    pairwise    — Compare Model A vs Model B responses (pairwise + position bias)
    reference   — Reference-based evaluation against ground truth answers
    validate    — Run full judge validation (Kappa + consistency + adversarial probes)
    ab-report   — Run both V1 and V2 evaluations and generate A/B comparison report

Usage:
    python main.py pointwise --suite data/test_suites/general_qa.json
    python main.py pairwise  --suite data/test_suites/general_qa.json
    python main.py reference --suite data/test_suites/general_qa.json
    python main.py validate  --suite data/test_suites/general_qa.json \\
                             --probes data/test_suites/adversarial_probes.json \\
                             --gold  data/gold_labels/human_annotated.json
    python main.py ab-report --suite data/test_suites/general_qa.json

Environment variables (set in .env or shell):
    OPENAI_API_KEY     — OpenAI API key
    ANTHROPIC_API_KEY  — Anthropic API key
    JUDGE_MODEL        — Override default judge model (default: gpt-4o)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from src.judge import JudgeClient
from src.logger import JSONLinesLogger, log_result
from src.aggregator import (
    TestSuiteAggregator,
    ABComparisonEngine,
    save_report_json,
    save_report_csv,
    save_ab_comparison_json,
)
from src.mitigations import (
    evaluate_pairwise_unbiased,
    calculate_flip_rate,
    check_self_enhancement_risk,
    analyze_score_distribution,
)
from src.validator import (
    calculate_judge_accuracy_and_kappa,
    run_adversarial_suite,
    generate_validation_report,
)

console = Console()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config(config_path: str = "config/suite_config.yaml") -> Dict[str, Any]:
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def make_judge(config: Dict[str, Any], temperature_override: Optional[float] = None) -> JudgeClient:
    """Instantiate JudgeClient from config."""
    judge_cfg = config.get("judge", {})
    model = os.environ.get("JUDGE_MODEL", judge_cfg.get("model", "gpt-4o"))
    temperature = temperature_override if temperature_override is not None \
        else judge_cfg.get("temperature", 0.0)
    return JudgeClient(
        model=model,
        temperature=temperature,
        max_tokens=judge_cfg.get("max_tokens", 2048),
        max_retries=judge_cfg.get("max_retries", 3),
        retry_wait_seconds=judge_cfg.get("retry_wait_seconds", 2),
        pass_threshold=config.get("evaluation", {}).get("pass_threshold", 3.0),
    )


def print_banner(title: str) -> None:
    console.print(Panel(f"[bold cyan]{title}[/bold cyan]", border_style="cyan"))


# ---------------------------------------------------------------------------
# Subcommand: pointwise
# ---------------------------------------------------------------------------

def cmd_pointwise(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    print_banner("🔍  Pointwise Evaluation")
    suite = load_json(args.suite)
    judge = make_judge(config)
    logger = JSONLinesLogger(
        log_dir=config.get("logging", {}).get("directory", "logs"),
        run_name="pointwise",
    )
    agg = TestSuiteAggregator(
        suite_name=os.path.basename(args.suite),
        evaluation_mode="pointwise",
        pass_threshold=config.get("evaluation", {}).get("pass_threshold", 3.0),
    )

    table = Table(box=box.ROUNDED, title="Pointwise Results")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Score", justify="right")
    table.add_column("Passed", justify="center")
    table.add_column("Verbosity Pen.")
    table.add_column("Parse Error")

    all_scores = []

    for case in suite:
        response = case.get("response_a", case.get("response", ""))
        verdict, audit = judge.judge_pointwise(test_case=case, response=response)
        log_result(logger, audit)

        row = {
            "verdict": verdict.model_dump() if verdict else None,
            "audit": audit,
        }
        agg.add_result(row)

        score = verdict.overall_score if verdict else None
        passed = "✅" if (verdict and verdict.passed) else "❌"
        vp = "Yes" if (verdict and verdict.verbosity_penalty_applied) else "No"
        err = audit.parse_error or ""
        if score:
            all_scores.append(score)

        table.add_row(
            case["id"],
            f"{score:.2f}" if score else "N/A",
            passed,
            vp,
            err[:40] if err else "—",
        )

    console.print(table)

    # Score distribution
    if all_scores:
        dist = analyze_score_distribution(all_scores)
        console.print(f"\n[bold]Score Distribution:[/bold] {dist['recommendation']}")

    # Build and save report
    report = agg.build_report()
    out_dir = config.get("reports", {}).get("output_directory", "reports")
    path = save_report_json(report, out_dir)
    save_report_csv(report, out_dir)

    console.print(f"\n[green]✓ Pass Rate:[/green] {report.pass_rate:.1%}")
    console.print(f"[green]✓ Mean Score:[/green] {report.mean_overall_score:.3f}")
    console.print(f"[green]✓ Report saved:[/green] {path}")
    console.print(f"[dim]Audit log:[/dim] {logger.log_path} ({logger.entry_count} entries)")


# ---------------------------------------------------------------------------
# Subcommand: pairwise
# ---------------------------------------------------------------------------

def cmd_pairwise(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    print_banner("⚖️   Pairwise Evaluation (with Position Bias Mitigation)")
    suite = load_json(args.suite)
    judge = make_judge(config)
    logger = JSONLinesLogger(
        log_dir=config.get("logging", {}).get("directory", "logs"),
        run_name="pairwise",
    )
    agg = TestSuiteAggregator(
        suite_name=os.path.basename(args.suite),
        evaluation_mode="pairwise",
    )

    # Self-enhancement check
    cand_cfg = config.get("candidates", {})
    risk = check_self_enhancement_risk(
        judge_model=judge.model,
        candidate_model_a=cand_cfg.get("model_a", "gpt-3.5-turbo"),
        candidate_model_b=cand_cfg.get("model_b", "gpt-4o-mini"),
    )
    if risk["risk_detected"]:
        console.print(f"[yellow]{risk['recommendation']}[/yellow]")
    else:
        console.print(f"[green]{risk['recommendation']}[/green]")

    table = Table(box=box.ROUNDED, title="Pairwise Results")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Forward", justify="center")
    table.add_column("Reverse", justify="center")
    table.add_column("Final Winner", justify="center")
    table.add_column("Consistent", justify="center")
    table.add_column("Conf")

    all_results = []

    for case in suite:
        resp_a = case.get("response_a", "")
        resp_b = case.get("response_b", "")

        result = evaluate_pairwise_unbiased(
            judge_client=judge,
            test_case=case,
            response_a=resp_a,
            response_b=resp_b,
        )

        # Log both audit entries
        if result["forward_audit"]:
            log_result(logger, result["forward_audit"])
        if result["reverse_audit"]:
            log_result(logger, result["reverse_audit"])

        agg.add_result(result)
        all_results.append(result)

        fw = result["forward_winner"]
        rv = result["reverse_winner_mapped"]
        final = result["final_winner"]
        consistent = "✅" if result["position_consistent"] else "⚠️"
        conf = f"{result['confidence_avg']:.2f}"

        table.add_row(case["id"], fw, rv, final, consistent, conf)

    console.print(table)

    flip_rate = calculate_flip_rate(all_results)
    console.print(f"\n[bold]Position Bias Flip Rate:[/bold] {flip_rate:.2%}")
    if flip_rate > 0.3:
        console.print("[red]⚠️  High flip rate! Significant position bias detected.[/red]")
    else:
        console.print("[green]✅ Flip rate is within acceptable range.[/green]")

    report = agg.build_report()
    out_dir = config.get("reports", {}).get("output_directory", "reports")
    path = save_report_json(report, out_dir)

    console.print(f"\n[green]✓ Flip Rate:[/green] {flip_rate:.2%}")
    console.print(f"[green]✓ Model A Wins:[/green] {report.win_count_a}")
    console.print(f"[green]✓ Model B Wins:[/green] {report.win_count_b}")
    console.print(f"[green]✓ Ties:[/green] {report.tie_count}")
    console.print(f"[green]✓ Report saved:[/green] {path}")


# ---------------------------------------------------------------------------
# Subcommand: reference
# ---------------------------------------------------------------------------

def cmd_reference(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    print_banner("📖  Reference-Based Evaluation")
    suite = load_json(args.suite)
    judge = make_judge(config)
    logger = JSONLinesLogger(
        log_dir=config.get("logging", {}).get("directory", "logs"),
        run_name="reference_based",
    )
    agg = TestSuiteAggregator(
        suite_name=os.path.basename(args.suite),
        evaluation_mode="reference_based",
        pass_threshold=config.get("evaluation", {}).get("pass_threshold", 3.0),
    )

    table = Table(box=box.ROUNDED, title="Reference-Based Results")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Score", justify="right")
    table.add_column("Passed", justify="center")

    for case in suite:
        if "reference_answer" not in case:
            console.print(f"[yellow]Skipping {case['id']}: no reference_answer[/yellow]")
            continue
        response = case.get("response_a", case.get("response", ""))
        verdict, audit = judge.judge_reference_based(test_case=case, response=response)
        log_result(logger, audit)
        agg.add_result({"verdict": verdict.model_dump() if verdict else None, "audit": audit})

        score = verdict.overall_score if verdict else None
        passed = "✅" if (verdict and verdict.passed) else "❌"
        table.add_row(case["id"], f"{score:.2f}" if score else "N/A", passed)

    console.print(table)

    report = agg.build_report()
    out_dir = config.get("reports", {}).get("output_directory", "reports")
    path = save_report_json(report, out_dir)
    console.print(f"\n[green]✓ Pass Rate:[/green] {report.pass_rate:.1%}")
    console.print(f"[green]✓ Report saved:[/green] {path}")


# ---------------------------------------------------------------------------
# Subcommand: validate
# ---------------------------------------------------------------------------

def cmd_validate(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    print_banner("🧪  Judge Validation Suite")
    suite = load_json(args.suite)
    judge = make_judge(config)

    results: Dict[str, Any] = {}

    # --- 1. Kappa validation ---
    if args.gold:
        console.print("\n[bold]1. Human/Gold Agreement (Cohen's Kappa)[/bold]")
        gold_data = load_json(args.gold)
        human_scores = [int(g["human_overall_score"]) for g in gold_data]

        judge_scores = []
        for g in gold_data:
            tc = {"id": g["id"], "input": g["input"], "system_prompt": ""}
            verdict, _ = judge.judge_pointwise(test_case=tc, response=g["response"])
            judge_scores.append(int(round(verdict.overall_score)) if verdict else 3)

        kappa_result = calculate_judge_accuracy_and_kappa(human_scores, judge_scores)
        results["kappa_validation"] = kappa_result
        console.print(f"  Exact Agreement: {kappa_result['exact_agreement_rate']:.2%}")
        console.print(f"  Near Agreement (±1): {kappa_result['near_agreement_rate']:.2%}")
        console.print(f"  Cohen's Kappa (quadratic): {kappa_result['cohens_kappa']:.4f}")
        console.print(f"  Interpretation: [italic]{kappa_result['interpretation']}[/italic]")
    else:
        kappa_result = None
        console.print("[yellow]Skipping Kappa validation (no --gold file provided)[/yellow]")

    # --- 2. Adversarial probes ---
    adversarial_result = None
    if args.probes:
        console.print("\n[bold]2. Adversarial Probe Suite[/bold]")
        probes = load_json(args.probes)
        adversarial_result = run_adversarial_suite(judge_client=judge, probe_cases=probes)
        results["adversarial_validation"] = adversarial_result

        adv_table = Table(box=box.SIMPLE, title="Probe Results by Type")
        adv_table.add_column("Probe Type")
        adv_table.add_column("Total", justify="right")
        adv_table.add_column("Passed", justify="right")
        adv_table.add_column("Pass Rate", justify="right")

        for ptype, counts in adversarial_result["by_type"].items():
            adv_table.add_row(
                ptype,
                str(counts["total"]),
                str(counts["passed"]),
                f"{counts['pass_rate']:.1%}",
            )
        console.print(adv_table)
        console.print(adversarial_result["recommendation"])

    # --- 3. Consolidated report ---
    console.print("\n[bold]Validation Summary[/bold]")
    final_report = generate_validation_report(
        kappa_result=kappa_result,
        adversarial_result=adversarial_result,
    )
    console.print(final_report["overall_verdict"])

    out_dir = config.get("reports", {}).get("output_directory", "reports")
    os.makedirs(out_dir, exist_ok=True)
    import time
    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(out_dir, f"validation_report_{ts}.json")
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2, default=str)
    console.print(f"\n[green]✓ Validation report saved:[/green] {report_path}")


# ---------------------------------------------------------------------------
# Subcommand: ab-report
# ---------------------------------------------------------------------------

def cmd_ab_report(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    print_banner("📊  A/B Comparison Report (V1 vs V2)")
    suite = load_json(args.suite)
    out_dir = config.get("reports", {}).get("output_directory", "reports")

    def run_suite(response_key: str, run_label: str) -> tuple:
        j = make_judge(config)
        agg = TestSuiteAggregator(
            suite_name=f"{os.path.basename(args.suite)}_{run_label}",
            evaluation_mode="pointwise",
        )
        scores = []
        for case in suite:
            response = case.get(response_key, "")
            if not response:
                continue
            verdict, audit = j.judge_pointwise(test_case=case, response=response)
            row = {"verdict": verdict.model_dump() if verdict else None, "audit": audit}
            agg.add_result(row)
            if verdict:
                scores.append(verdict.overall_score)
        return agg.build_report(), scores

    console.print("[dim]Running V1 (Model A responses)...[/dim]")
    report_v1, scores_v1 = run_suite("response_a", "V1")

    console.print("[dim]Running V2 (Model B responses)...[/dim]")
    report_v2, scores_v2 = run_suite("response_b", "V2")

    engine = ABComparisonEngine(
        report_v1=report_v1,
        report_v2=report_v2,
        significance_threshold=config.get("ab_comparison", {}).get("significance_threshold", 0.05),
    )
    comparison = engine.compare(scores_v1=scores_v1, scores_v2=scores_v2)

    # Display
    ab_table = Table(box=box.ROUNDED, title="A/B Comparison Summary")
    ab_table.add_column("Metric")
    ab_table.add_column("V1 (Model A)", justify="right")
    ab_table.add_column("V2 (Model B)", justify="right")
    ab_table.add_column("Delta", justify="right")

    ab_table.add_row(
        "Mean Overall Score",
        f"{comparison['mean_v1']:.3f}",
        f"{comparison['mean_v2']:.3f}",
        f"{comparison['difference']:+.3f}",
    )
    ab_table.add_row(
        "Pass Rate",
        f"{comparison['pass_rate_v1']:.1%}",
        f"{comparison['pass_rate_v2']:.1%}",
        f"{(comparison['pass_rate_v2'] - comparison['pass_rate_v1']):+.1%}",
    )
    for row in comparison.get("criterion_comparison", []):
        ab_table.add_row(
            f"  {row['criterion']}",
            f"{row['mean_v1']:.2f}" if row["mean_v1"] else "—",
            f"{row['mean_v2']:.2f}" if row["mean_v2"] else "—",
            f"{row['delta']:+.2f}" if row["delta"] is not None else "—",
        )

    console.print(ab_table)
    console.print(f"\n[bold]Winner:[/bold] [cyan]{comparison['winner']}[/cyan]")
    console.print(comparison["recommendation"])

    # Save reports
    save_report_json(report_v1, out_dir)
    save_report_json(report_v2, out_dir)
    path = save_ab_comparison_json(comparison, out_dir)
    console.print(f"\n[green]✓ A/B report saved:[/green] {path}")


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge Evaluation Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Shared arguments
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--suite",
        default="data/test_suites/general_qa.json",
        help="Path to test suite JSON file",
    )
    common.add_argument(
        "--config",
        default="config/suite_config.yaml",
        help="Path to suite config YAML",
    )

    # pointwise
    sp = subparsers.add_parser("pointwise", parents=[common], help="Pointwise evaluation")

    # pairwise
    subparsers.add_parser("pairwise", parents=[common], help="Pairwise A/B evaluation")

    # reference
    subparsers.add_parser("reference", parents=[common], help="Reference-based evaluation")

    # validate
    val_p = subparsers.add_parser("validate", parents=[common], help="Run judge validation")
    val_p.add_argument(
        "--probes",
        default="data/test_suites/adversarial_probes.json",
        help="Path to adversarial probes JSON",
    )
    val_p.add_argument(
        "--gold",
        default="data/gold_labels/human_annotated.json",
        help="Path to human-annotated gold labels JSON",
    )

    # ab-report
    subparsers.add_parser("ab-report", parents=[common], help="A/B comparison report")

    args = parser.parse_args()
    config = load_config(args.config)

    dispatch = {
        "pointwise": cmd_pointwise,
        "pairwise":  cmd_pairwise,
        "reference": cmd_reference,
        "validate":  cmd_validate,
        "ab-report": cmd_ab_report,
    }

    dispatch[args.command](args, config)


if __name__ == "__main__":
    main()
