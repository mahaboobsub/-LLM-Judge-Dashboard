# LLM-as-Judge Evaluation Pipeline 
URL - https://judgedashboardofllm.streamlit.app/
<img width="1909" height="795" alt="image" src="https://github.com/user-attachments/assets/86d09c65-4798-4fa5-b599-b7b71657517b" />


A production-grade, automated pipeline for scoring and comparing LLM outputs using another LLM as a judge — with structured rubrics, bias mitigations, full audit logging, and statistical validation.

---

## 📁 Project Structure

```
llm-as-judge-pipeline/
├── config/
│   ├── rubric.yaml               # Criteria definitions, scale anchors, weights
│   └── suite_config.yaml         # Execution parameters, model selections
├── data/
│   ├── test_suites/
│   │   ├── general_qa.json       # 10 standard evaluation test cases
│   │   └── adversarial_probes.json  # Bias probe test cases
│   └── gold_labels/
│       └── human_annotated.json  # Human-graded labels for Kappa validation
├── logs/                         # Auto-created: .jsonl audit trail files
├── reports/                      # Auto-created: JSON/CSV report outputs
├── src/
│   ├── schema.py                 # Pydantic v2 models for all structured outputs
│   ├── prompts.py                # System prompts, few-shot anchors, user prompt builders
│   ├── judge.py                  # JudgeClient: API wrappers + JSON repair + audit logging
│   ├── mitigations.py            # Bias detection and code-level mitigations
│   ├── aggregator.py             # Metrics aggregation + A/B comparison engine
│   ├── validator.py              # Cohen's Kappa, consistency, adversarial probe suite
│   └── logger.py                 # JSON Lines append-only logger
├── tests/
│   ├── test_parser.py            # Schema validation + JSON extraction unit tests
│   └── test_mitigations.py       # Bias mitigation logic unit tests
├── main.py                       # CLI entrypoint
├── requirements.txt
└── .env.example
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY (or ANTHROPIC_API_KEY)
```

### 3. Run Evaluations

```bash
# Pointwise evaluation of Model A responses
python main.py pointwise --suite data/test_suites/general_qa.json

# Pairwise A-vs-B with position bias mitigation
python main.py pairwise --suite data/test_suites/general_qa.json

# Reference-based evaluation against ground truth
python main.py reference --suite data/test_suites/general_qa.json

# Full judge validation (Kappa + adversarial probes)
python main.py validate \
    --suite  data/test_suites/general_qa.json \
    --probes data/test_suites/adversarial_probes.json \
    --gold   data/gold_labels/human_annotated.json

# A/B comparison report (V1 vs V2)
python main.py ab-report --suite data/test_suites/general_qa.json
```

### 4. Run Unit Tests

```bash
python -m pytest tests/ -v
```

---

## 🧠 Architecture Overview

### Evaluation Modes

| Mode | Description | Best For |
|---|---|---|
| **Pointwise** | Score one response against a rubric | Absolute quality gating |
| **Pairwise** | Compare A vs B side-by-side | Regression testing, model comparison |
| **Reference-Based** | Score against a ground-truth answer | Factual Q&A, translation |

### Bias Mitigations

| Bias | Mitigation Strategy |
|---|---|
| **Position Bias** | Double-pass order swapping; inconsistent verdicts → "Tie" |
| **Verbosity Bias** | Explicit rubric penalty + padded-answer probe test |
| **Self-Enhancement** | Model family cross-check advisory |
| **Sycophancy/Style** | Chain-of-thought required before scoring + confidently-wrong probe |
| **Score Clustering** | Few-shot anchor examples (scores 1, 3, 5) injected into every prompt |

### Key Metrics

- **Flip Rate** — Proportion of pairwise cases where order swap changed winner (position bias measure)
- **Pass Rate** — Proportion of responses meeting the minimum score threshold
- **Cohen's Kappa (κ)** — Quadratic weighted agreement between judge and human annotators
- **Consistency Rate** — Proportion of cases with identical scores across repeated runs

---

## 📊 Output Files

### Audit Logs (`logs/`)

Every judge call is logged as a JSON Lines record containing:
- Full system and user prompts sent to the judge
- Raw API response text
- Parsed and validated verdict
- Token usage (prompt tokens, completion tokens, cost in USD)
- Wall-clock latency in milliseconds
- Retry count and any parse errors

### Reports (`reports/`)

- `report_*.json` — Full SuiteReport with per-criterion stats
- `report_*.csv` — Summary metrics in spreadsheet format
- `ab_comparison_*.json` — A/B comparison with winner declaration
- `validation_report_*.json` — Full judge validation results

---

## ⚙️ Configuration

### `config/rubric.yaml`

Defines scoring criteria, weights, and 1/3/5 anchor examples:

```yaml
criteria:
  correctness:
    weight: 0.30
    anchors:
      1: "Contains clear factual errors."
      3: "Mostly correct with minor inaccuracies."
      5: "Completely accurate."
```

### `config/suite_config.yaml`

Controls judge model, evaluation mode, logging, and A/B parameters:

```yaml
judge:
  model: "gpt-4o"
  temperature: 0.0
  max_retries: 3

evaluation:
  pass_threshold: 3.0
  run_bias_checks: true
```

---

## 📐 Pydantic Schemas

### PointwiseVerdict

```python
class PointwiseVerdict(BaseModel):
    criteria_breakdown: Dict[str, CriterionScore]
    overall_score: float          # 1.0–5.0
    overall_rationale: str
    passed: bool
    verbosity_penalty_applied: bool
```

### PairwiseVerdict

```python
class PairwiseVerdict(BaseModel):
    winner: Literal["Model_A", "Model_B", "Tie"]
    rationale: str
    confidence: float             # 0.0–1.0
    score_a: Optional[float]
    score_b: Optional[float]
```

---

## 🧪 Judge Validation

### Cohen's Kappa (κ) Interpretation

| κ value | Interpretation |
|---|---|
| < 0.20 | Slight agreement |
| 0.20–0.40 | Fair agreement |
| 0.40–0.60 | Moderate agreement |
| 0.60–0.80 | Substantial agreement |
| ≥ 0.80 | Almost perfect agreement |

### Adversarial Probe Types

| Probe Type | What It Tests |
|---|---|
| `verbosity` | Does judge penalize padded but correct answers? |
| `sycophancy` | Does judge detect confident but factually wrong answers? |
| `position_bias` | Does judge prefer the correct answer regardless of position? |
| `score_clustering` | Does judge use the full 1–5 scale (not just 3–4)? |

---

## 💰 Cost Tracking

The pipeline tracks token usage and estimates USD cost per run using the pricing table in `src/judge.py`. Costs are logged in each audit entry and totalled in suite reports.

Example (gpt-4o pricing):
- Prompt: $0.005 / 1K tokens
- Completion: $0.015 / 1K tokens

---

## 🔧 Extending the Pipeline

### Add a New Criterion

1. Edit `config/rubric.yaml` to add the new criterion definition.
2. Update the `validate_required_criteria` validator in `src/schema.py`.
3. Update the rubric description in `src/prompts.py`.

### Add a New Judge Model

1. Add pricing to `MODEL_PRICING` in `src/judge.py`.
2. Set `JUDGE_MODEL=your-model` in `.env` or update `config/suite_config.yaml`.

### Add New Test Cases

Add entries to `data/test_suites/general_qa.json` following the existing schema:
```json
{
  "id": "qa_011",
  "input": "Your question here",
  "system_prompt": "Optional system instructions",
  "reference_answer": "Ground truth answer",
  "response_a": "Model A response",
  "response_b": "Model B response"
}
```

---

## ✅ Assignment Checklist

- [x] **Structured Parsing**: Pydantic v2 models with strict validation + JSON repair/retry logic
- [x] **Multi-Criteria Rubric**: 6 criteria (Correctness, Faithfulness, Completeness, Instruction-Following, Tone, Safety) with explicit 1/3/5 anchors
- [x] **Position Bias Calculation**: Double-pass pairwise + Flip Rate metric reported
- [x] **Verbosity Control**: Rubric penalty + padded-answer probe tests
- [x] **Sycophancy Mitigation**: Chain-of-thought required + confidently-wrong probes
- [x] **Audit Trail & Logging**: Every prompt/response/token/cost/latency logged to `.jsonl`
- [x] **A/B Suite Report**: Pass rates, criterion averages, automated winner declaration
- [x] **Cohen's Kappa**: Against human-annotated gold labels
- [x] **CLI Entrypoint**: `main.py` with 5 subcommands
- [x] **Unit Tests**: `tests/test_parser.py` + `tests/test_mitigations.py`
