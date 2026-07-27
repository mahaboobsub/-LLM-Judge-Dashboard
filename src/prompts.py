"""
src/prompts.py
==============
Prompt templates for the LLM-as-Judge pipeline.

Contains:
  - Few-shot calibration anchor examples (scores 1, 3, 5)
  - Pointwise evaluation system prompt
  - Pairwise evaluation system prompt
  - Reference-based evaluation system prompt
  - Utility functions to build populated prompt strings
"""

from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Few-shot anchor examples (Score Clustering Mitigation)
# ---------------------------------------------------------------------------

FEW_SHOT_ANCHORS = """
=== CALIBRATION ANCHOR EXAMPLES ===

These examples show what each score level looks like. Use them to calibrate
your scoring and avoid clustering all responses around the middle (3).

--- SCORE 1 (POOR) ---
Input: "What is the capital of France?"
Response: "I think it might be Lyon or maybe Marseille? France has many great cities."
Correctness: 1 — Factually wrong; Paris is not mentioned.
Faithfulness: 1 — No basis for the claim.
Overall: 1 — Completely fails to answer correctly.

--- SCORE 3 (ACCEPTABLE) ---
Input: "What is the capital of France?"
Response: "France is a beautiful country in Western Europe with a rich culture, history,
and cuisine. It is known for landmarks like the Eiffel Tower, the Louvre, and the
Palace of Versailles. Its capital city is Paris, which sits on the Seine river and
has been the center of French civilization for centuries."
Correctness: 5 — Paris is correct.
Completeness: 5 — Answers the question.
Verbosity: penalized — 80% of the response is unnecessary filler.
Overall: 3 — Contains the answer but is excessively verbose for a simple factual query.

--- SCORE 5 (EXCELLENT) ---
Input: "What is the capital of France?"
Response: "Paris is the capital of France."
Correctness: 5 — Perfectly accurate.
Completeness: 5 — Directly and fully answers the question.
Conciseness: 5 — No unnecessary content.
Overall: 5 — Exemplary response.

=== END OF CALIBRATION EXAMPLES ===
"""

# ---------------------------------------------------------------------------
# Rubric criteria description (injected into all prompts)
# ---------------------------------------------------------------------------

RUBRIC_DESCRIPTION = """
=== EVALUATION RUBRIC ===

Score each criterion from 1 to 5. You MUST write step-by-step reasoning
for each criterion BEFORE assigning its score.

CRITERIA:
1. Correctness (weight: 30%)
   Is every factual claim accurate and verifiable? Check numbers, names,
   dates, formulas, and logic chains. Penalize any factual error heavily.

2. Faithfulness (weight: 20%)
   Does the response strictly use information from the provided context
   or well-established knowledge? Penalize hallucinated details.

3. Completeness (weight: 20%)
   Does the response address ALL parts of the question or task?
   Missing a required sub-task should reduce this score proportionally.

4. Instruction Following (weight: 15%)
   Does the response comply with ALL explicit formatting, length,
   style, or structural instructions in the prompt?

5. Tone (weight: 10%)
   Is the language register, style, and voice appropriate for the
   stated context and audience?

6. Safety (weight: 5%)
   Is the content free of harmful, offensive, misleading, or
   policy-violating material?

VERBOSITY PENALTY:
   If the response pads a correct, simple answer with more than ~30%
   unrelated or unnecessary content, deduct up to 0.5 from overall_score.
   Mark verbosity_penalty_applied as true.

OUTPUT FORMAT:
   You MUST return ONLY valid JSON matching the requested schema.
   Do NOT include any text outside the JSON block.
=== END RUBRIC ===
"""

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

POINTWISE_SYSTEM_PROMPT = f"""You are an expert, impartial AI quality evaluator. Your task is to assess
the quality of a single AI model response against a structured rubric.

GOLDEN RULES:
- You are COMPLETELY IMPARTIAL. Do not favor any model, style, or length.
- A polite, confident, or well-formatted response is NOT inherently better.
  Judge ONLY on correctness, accuracy, and instruction adherence.
- A concise correct answer is ALWAYS better than a verbose correct answer
  (given the same accuracy and completeness).
- Write your reasoning for each criterion BEFORE assigning the score.
- Do NOT assign all scores as 3. Use the FULL 1–5 scale.

{FEW_SHOT_ANCHORS}

{RUBRIC_DESCRIPTION}

Return ONLY valid JSON. No markdown, no code fences, no extra text."""

PAIRWISE_SYSTEM_PROMPT = f"""You are an expert, impartial AI quality evaluator. Your task is to compare
two AI model responses (Model A and Model B) to the same input and declare a winner.

GOLDEN RULES:
- You are COMPLETELY IMPARTIAL. Do not favor Model A or Model B based on position.
- A polite, confident, or well-formatted response is NOT inherently better.
  Judge ONLY on factual correctness, completeness, and instruction adherence.
- If one response is factually wrong and the other is correct, the correct one wins—
  regardless of writing style, confidence, or length.
- Write comparative step-by-step reasoning for EACH criterion BEFORE declaring winner.
- If both responses are equally good, declare "Tie".

{FEW_SHOT_ANCHORS}

{RUBRIC_DESCRIPTION}

Return ONLY valid JSON matching the PairwiseVerdict schema. No markdown, no extra text."""

REFERENCE_BASED_SYSTEM_PROMPT = f"""You are an expert, impartial AI quality evaluator. Your task is to assess
an AI model response by comparing it to a provided reference (ground-truth) answer.

GOLDEN RULES:
- The reference answer is your ground truth. Evaluate how closely the response
  aligns with it in terms of correctness, completeness, and faithfulness.
- Do NOT penalize the model for using different wording if the meaning is correct.
- Do penalize any factual deviation from the reference, even if politely phrased.
- Write step-by-step reasoning for each criterion BEFORE assigning scores.

{FEW_SHOT_ANCHORS}

{RUBRIC_DESCRIPTION}

Return ONLY valid JSON. No markdown, no code fences, no extra text."""

# ---------------------------------------------------------------------------
# User prompt builders
# ---------------------------------------------------------------------------


def build_pointwise_user_prompt(
    input_text: str,
    response: str,
    system_prompt: str = "",
    pass_threshold: float = 3.0,
) -> str:
    """Build the user-turn prompt for pointwise evaluation."""
    ctx = f"\n[SYSTEM CONTEXT PROVIDED TO MODEL]\n{system_prompt}\n" if system_prompt else ""
    return f"""Evaluate the following AI model response.

[INPUT QUESTION / TASK]
{input_text}
{ctx}
[MODEL RESPONSE TO EVALUATE]
{response}

[INSTRUCTIONS]
1. For each criterion, write your step-by-step reasoning first.
2. Then assign a score 1–5.
3. Compute overall_score as a weighted average (correctness×0.30, faithfulness×0.20,
   completeness×0.20, instruction_following×0.15, tone×0.10, safety×0.05).
4. Set passed=true if overall_score >= {pass_threshold}.
5. Apply verbosity_penalty_applied=true if excessive padding was detected.

Return ONLY this JSON structure:
{{
  "criteria_breakdown": {{
    "correctness": {{"score": <int 1-5>, "rationale": "<step-by-step reasoning>"}},
    "faithfulness": {{"score": <int 1-5>, "rationale": "<reasoning>"}},
    "completeness": {{"score": <int 1-5>, "rationale": "<reasoning>"}},
    "instruction_following": {{"score": <int 1-5>, "rationale": "<reasoning>"}},
    "tone": {{"score": <int 1-5>, "rationale": "<reasoning>"}},
    "safety": {{"score": <int 1-5>, "rationale": "<reasoning>"}}
  }},
  "overall_score": <float 1.0-5.0>,
  "overall_rationale": "<summary of judgment>",
  "passed": <true|false>,
  "verbosity_penalty_applied": <true|false>
}}"""


def build_pairwise_user_prompt(
    input_text: str,
    response_a: str,
    response_b: str,
    system_prompt: str = "",
) -> str:
    """Build the user-turn prompt for pairwise A-vs-B evaluation."""
    ctx = f"\n[SYSTEM CONTEXT PROVIDED TO BOTH MODELS]\n{system_prompt}\n" if system_prompt else ""
    return f"""Compare these two AI model responses to the same input.

[INPUT QUESTION / TASK]
{input_text}
{ctx}
[MODEL A RESPONSE]
{response_a}

[MODEL B RESPONSE]
{response_b}

[INSTRUCTIONS]
1. Analyze each criterion for both Model A and Model B.
2. Write step-by-step comparative reasoning BEFORE declaring a winner.
3. Declare winner: "Model_A", "Model_B", or "Tie".
4. Assign confidence between 0.0 (uncertain) and 1.0 (certain).
5. Optionally assign absolute scores (score_a, score_b) on 1–5 scale.

Return ONLY this JSON structure:
{{
  "winner": "<Model_A|Model_B|Tie>",
  "rationale": "<comparative step-by-step reasoning>",
  "confidence": <float 0.0-1.0>,
  "score_a": <float 1.0-5.0 or null>,
  "score_b": <float 1.0-5.0 or null>
}}"""


def build_reference_based_user_prompt(
    input_text: str,
    response: str,
    reference_answer: str,
    system_prompt: str = "",
    pass_threshold: float = 3.0,
) -> str:
    """Build the user-turn prompt for reference-based evaluation."""
    ctx = f"\n[SYSTEM CONTEXT]\n{system_prompt}\n" if system_prompt else ""
    return f"""Evaluate the following AI model response against the provided reference answer.

[INPUT QUESTION / TASK]
{input_text}
{ctx}
[REFERENCE ANSWER (Ground Truth)]
{reference_answer}

[MODEL RESPONSE TO EVALUATE]
{response}

[INSTRUCTIONS]
Compare the model response to the reference answer.
1. For each criterion, write step-by-step reasoning BEFORE assigning a score.
2. Set passed=true if overall_score >= {pass_threshold}.

Return ONLY the JSON structure with criteria_breakdown, overall_score,
overall_rationale, passed, and verbosity_penalty_applied fields."""
