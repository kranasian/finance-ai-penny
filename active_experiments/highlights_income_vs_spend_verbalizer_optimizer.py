"""
Optimizer runner for ``P:HighlightsIncomeVsSpendVerbalizer``.

Run from ``finance-ai-penny`` repo root:

  python3 active_experiments/highlights_income_vs_spend_verbalizer_optimizer.py --test 0
  python3 active_experiments/highlights_income_vs_spend_verbalizer_optimizer.py --test all
  python3 active_experiments/highlights_income_vs_spend_verbalizer_optimizer.py --list
"""

from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from typing import Any

try:
  from dotenv import load_dotenv
except Exception:  # pragma: no cover
  load_dotenv = None

try:
  from google import genai
  from google.genai import types
except Exception:  # pragma: no cover
  genai = None  # type: ignore[assignment]
  types = None  # type: ignore[assignment]

if load_dotenv is not None:
  load_dotenv()

GEMINI_3_1_FLASH_LITE_MODEL = "gemini-3.1-flash-lite"
TEMPLATE_NAME = "P:HighlightsIncomeVsSpendVerbalizer"
_TEST_SEPARATOR = "=" * 72
_SECTION_RULE = "-" * 72

SYSTEM_PROMPT = """**OBJECTIVE:** You are Penny, a positive but pragmatic financial advisor telling a close friend about their finances. Your task is to summarize their financial status highlights.

**INPUT:**
A JSON object with:
* `"current_date"`: (string) The date of the update.
* `"percent_of_month_passed"`: (number) The elapsed percentage of the month.
* `"insights"`: A list of objects. Each object represents a financial metric:
    * `"metric"`: (string) The metric name (e.g., "Income", "Spent", "Bills", "Food", "Shopping", "Others").
    * `"actual"`: (number) The actual dollar amount spent or earned.
    * `"status"`: (string) The status of the metric relative to the expected-to-date target. Can be `"significantly_above"`, `"notably_above"`, `"on_track"`, `"notably_below"`, or `"significantly_below"`.
---

### **OUTPUT GENERATION RULES:**

* **`title` Field:**
    * Must be a maximum of **6 words**.
    * Incorporate relevant emojis.
    * Do NOT use the word "budget" (or "Budget") under any circumstances. Use terms like "Targets", "Expectations", "Plan", or "Overview".
    * Vary the titles to fit the month's specific story (e.g., focus on savings, plan adjustments, or income milestones).

* **`summary` Field:**
    * Must be a **single line**.
    * Must be a maximum of **50 words**.
    * Do not include any greeting.
    * Speak directly to the user in the second-person perspective (`you`, `your`, `yours`).
    * Do NOT use first-person plural pronouns like `we`, `our`, `us`, `we've`.
    * Highlight **at most 4 metrics total**: Income + up to 3 non-Income insights with the highest deviation:
            * Prioritize `"significantly_above"` / `"significantly_below"`.
            * Next, `"notably_above"` / `"notably_below"`.
            * Use `"on_track"` for non-Income metrics only if you still need more points.
    * Do NOT show or output any expected numbers, targets, or monthly expected totals. Only output the actual numbers.
    * Describe the status of each selected metric using adjectives that match the `"status"` field exactly:
        * `"significantly_above"`: use "significantly/well/far above" or similar.
        * `"significantly_below"`: use "significantly/well/far below" or similar.
        * `"notably_above"`: use "notably/moderately above" or similar.
        * `"notably_below"`: use "notably/moderately below" or similar.
        * `"on_track"`: use "on track", "almost matched", "right around expectations", or similar.
    * Format all actual monetary amounts with **NO DECIMALS** but **WITH COMMAS** (e.g., $1,234).
    * Incorporate relevant emojis to make the message relatable.

* **SYNTACTIC & LEXICAL DIVERSITY:**
    * Avoid using a single rigid sentence template (like 'Your X was Y, while Z was W').
    * Vary your sentence structures across runs.
    * Use a rich, varied vocabulary of positive financial adjectives. The message should feel naturally conversational.

---

### **CRITICAL NEGATIVE CONSTRAINTS:**

* **PROHIBITED WORD:** Do NOT use the word "budget", "Budget", "budgeted", or "budgets" in any part of the output. Use alternative words like "target", "expectation", "plan", or "planned".
* **STRICT MAX 4 METRICS LIMIT:** Never mention more than 4 metrics. Always include Income + at most 3 others.
* **NO FORCED CONTRAST:** If all highlighted points have the same status direction (e.g., all are below expectations), do NOT force a contrast or invent an opposite direction. Describe them all accurately.
* Do not compare spending and income.
* Do not perform any calculations or math in the output text.
* Do not output expected amounts or targets (only actual figures).
"""


def _build_output_schema() -> "types.Schema":
  if types is None:  # pragma: no cover
    raise RuntimeError("Install `google-genai` for this optimizer.")
  return types.Schema(
    type=types.Type.OBJECT,
    required=["title", "summary"],
    properties={
      "title": types.Schema(
        type=types.Type.STRING,
        description="A concise, 6-word maximum title with relevant emojis.",
      ),
      "summary": types.Schema(
        type=types.Type.STRING,
        description="A single-line summary, maximum 50 words, with relevant emojis.",
      ),
    },
  )


def _print_section_banner(title: str) -> None:
  print(f"\n{_SECTION_RULE}\n{title}\n{_SECTION_RULE}\n")


def _extract_stream_chunk_text(chunk: Any) -> str:
  pieces: list[str] = []
  for cand in getattr(chunk, "candidates", None) or []:
    content = getattr(cand, "content", None)
    if not content:
      continue
    for part in getattr(content, "parts", None) or []:
      if getattr(part, "thought", False):
        continue
      t = getattr(part, "text", None)
      if isinstance(t, str) and t:
        pieces.append(t)
  if pieces:
    return "".join(pieces)
  with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=r".*non-text parts in the response.*")
    agg = getattr(chunk, "text", None)
    return agg if isinstance(agg, str) else ""


def _parse_model_json_object(raw: str) -> dict[str, Any]:
  text = raw.strip()
  if text.startswith("```json"):
    text = text[7:]
  elif text.startswith("```"):
    text = text[3:]
  if text.endswith("```"):
    text = text[:-3]
  text = text.strip()
  parsed = json.loads(text)
  if not isinstance(parsed, dict):
    raise ValueError("Response must be a JSON object")
  return parsed


TEST_CASES: list[dict[str, Any]] = [
  {
    "name": "mid_month_income_down_bills_others_up",
    "batch": 0,
    "input": """
{
  "current_date": "June 24, 2025",
  "percent_of_month_passed": 80,
  "insights": [
    {
      "metric": "Income",
      "actual": 6076,
      "status": "significantly_below"
    },
    {
      "metric": "Spent",
      "actual": 447,
      "status": "significantly_below"
    },
    {
      "metric": "Bills",
      "actual": 3384,
      "status": "significantly_above"
    },
    {
      "metric": "Others",
      "actual": 2937,
      "status": "significantly_above"
    }
  ]
}
""",
    "output": {
      "title": "Mid-Year Financial Plan Review 📈",
      "summary": (
        "Your income is significantly below expectations at $6,076, while your bills are "
        "significantly above at $3,384 and your other spending is also significantly above at $2,937. 💸"
      ),
    },
  },
  {
    "name": "early_month_spending_well_below",
    "batch": 1,
    "input": """
{
  "current_date": "July 11, 2026",
  "percent_of_month_passed": 35,
  "insights": [
    {
      "metric": "Spent",
      "actual": 406,
      "status": "significantly_below"
    },
    {
      "metric": "Food",
      "actual": 223,
      "status": "significantly_below"
    },
    {
      "metric": "Shopping",
      "actual": 182,
      "status": "significantly_below"
    }
  ]
}
""",
    "output": {
      "title": "Great Progress With Your Plan 🌟",
      "summary": (
        "You are doing fantastic, as your total spending of $406 is significantly below expectations, "
        "while your food costs of $223 and shopping expenses of $182 are also kept significantly below your plan. 🥂"
      ),
    },
  },
  {
    "name": "income_on_track_must_still_be_mentioned",
    "batch": 2,
    "input": """
{
  "current_date": "July 14, 2026",
  "percent_of_month_passed": 45,
  "insights": [
    {
      "metric": "Income",
      "actual": 2555,
      "status": "on_track"
    },
    {
      "metric": "Bills",
      "actual": 3255,
      "status": "significantly_above"
    },
    {
      "metric": "Spent",
      "actual": 3192,
      "status": "significantly_above"
    },
    {
      "metric": "Others",
      "actual": 683,
      "status": "significantly_above"
    }
  ]
}
""",
    "output": {
      "title": "Income Steady, Bills Climbing ⚠️",
      "summary": (
        "Your income is on track at $2,555, but bills are significantly above expectations at $3,255 "
        "and total spending is also significantly above at $3,192. 💸"
      ),
    },
  },
]


def _normalize_insight_input(insight_input: str | dict[str, Any]) -> tuple[str, dict[str, Any]]:
  """Return (multi-line JSON text for the model, parsed dict for sandbox checks)."""
  if isinstance(insight_input, str):
    text = insight_input.strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
      raise ValueError("insight_input JSON must be an object")
    return text, parsed
  if not isinstance(insight_input, dict) or not insight_input:
    raise ValueError("insight_input must be a non-empty dict or JSON string")
  return json.dumps(insight_input, indent=2), insight_input


class HighlightsIncomeVsSpendVerbalizerOptimizer:
  """Gemini runner for ``P:HighlightsIncomeVsSpendVerbalizer`` prompt tuning."""

  def __init__(
    self,
    model_name: str = GEMINI_3_1_FLASH_LITE_MODEL,
    *,
    thinking_budget: int = 0,
  ):
    if genai is None or types is None:  # pragma: no cover
      raise RuntimeError(
        "Gemini client dependencies not available. Install `google-genai` (and optionally `python-dotenv`)."
      )
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set. Set it in .env or environment.")
    self.client = genai.Client(api_key=api_key)
    self.model_name = model_name
    self.thinking_budget = thinking_budget
    self.temperature = 0.5
    self.top_p = 0.95
    self.top_k = 40
    self.max_output_tokens = 256
    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    ]
    self.system_prompt = SYSTEM_PROMPT
    self.output_schema = _build_output_schema()

  def generate_response(self, insight_input: str | dict[str, Any]) -> dict[str, Any]:
    request_text_str, _ = _normalize_insight_input(insight_input)
    request_text = types.Part.from_text(text=request_text_str)
    contents = [types.Content(role="user", parts=[request_text])]
    cfg = types.GenerateContentConfig(
      temperature=self.temperature,
      top_p=self.top_p,
      top_k=self.top_k,
      max_output_tokens=self.max_output_tokens,
      safety_settings=self.safety_settings,
      system_instruction=[types.Part.from_text(text=self.system_prompt)],
      thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
      response_schema=self.output_schema,
      response_mime_type="application/json",
    )

    output_text = ""
    for chunk in self.client.models.generate_content_stream(
      model=self.model_name,
      contents=contents,
      config=cfg,
    ):
      output_text += _extract_stream_chunk_text(chunk)

    if not output_text.strip():
      raise ValueError("Empty response from model.")
    return _parse_model_json_object(output_text)


def mechanical_sandbox_check(insight_input: dict[str, Any], verbalizer_output: dict[str, Any]) -> dict[str, Any]:
  """Deterministic checks for title/summary constraints."""
  issues: list[str] = []
  title = verbalizer_output.get("title")
  summary = verbalizer_output.get("summary")

  if not isinstance(title, str) or not title.strip():
    issues.append("title must be a non-empty string")
  else:
    # Count words ignoring emoji / punctuation tokens that are symbols only
    words = [w for w in re.findall(r"[A-Za-z0-9']+", title) if w]
    if len(words) > 6:
      issues.append(f"title has {len(words)} words (max 6)")
    if re.search(r"\bbudgets?\b|\bbudgeted\b", title, flags=re.IGNORECASE):
      issues.append("title contains prohibited word 'budget'")

  if not isinstance(summary, str) or not summary.strip():
    issues.append("summary must be a non-empty string")
  else:
    if "\n" in summary.strip():
      issues.append("summary must be a single line")
    words = summary.strip().split()
    if len(words) > 50:
      issues.append(f"summary has {len(words)} words (max 50)")
    if re.search(r"\bbudgets?\b|\bbudgeted\b", summary, flags=re.IGNORECASE):
      issues.append("summary contains prohibited word 'budget'")
    if re.search(r"\b(we|our|us|we've|we're|we'll)\b", summary, flags=re.IGNORECASE):
      issues.append("summary uses prohibited first-person plural pronoun")

    # Count mentioned metrics from input that appear in summary (case-insensitive)
    metrics = [
      str(item.get("metric", "")).strip()
      for item in insight_input.get("insights", [])
      if isinstance(item, dict) and item.get("metric")
    ]
    mentioned = []
    summary_l = summary.lower()
    for metric in metrics:
      token = metric.lower()
      if token == "spent":
        if "spend" in summary_l or "spent" in summary_l:
          mentioned.append(metric)
      elif token in summary_l:
        mentioned.append(metric)
    if len(mentioned) > 4:
      issues.append(f"summary mentions more than 4 metrics: {mentioned}")
    if any(m.lower() == "income" for m in metrics) and not any(m.lower() == "income" for m in mentioned):
      issues.append("Income is in insights but missing from summary")

  good = len(issues) == 0
  return {
    "good_copy": good,
    "info_correct": good,
    "eval_text": "\n".join(f"- {x}" for x in issues) if issues else "",
  }


def test_optimizer(
  batch_index: int | None = None,
  *,
  run_sandbox: bool = True,
  check: bool = False,
) -> list[str]:
  optimizer = HighlightsIncomeVsSpendVerbalizerOptimizer()
  failures: list[str] = []

  if batch_index is not None:
    cases = [tc for tc in TEST_CASES if int(tc.get("batch", -1)) == int(batch_index)]
    if not cases:
      raise SystemExit(f"No tests found for batch={batch_index}")
  else:
    cases = TEST_CASES

  for i, tc in enumerate(cases):
    if i:
      print(f"\n{_TEST_SEPARATOR}\n")
    batch = tc.get("batch")
    batch_s = str(batch) if isinstance(batch, int) else "—"
    print(f"# Test: {tc['name']}  (batch {batch_s})  [{TEMPLATE_NAME}]\n")
    insight_input = tc["input"]
    insight_text, insight_obj = _normalize_insight_input(insight_input)
    _print_section_banner("Input")
    print(insight_text)
    try:
      result = optimizer.generate_response(insight_input)
      _print_section_banner("Output")
      print(json.dumps(result, indent=2))
      if tc.get("output") is not None:
        _print_section_banner("Example Output (reference)")
        print(json.dumps(tc["output"], indent=2))
      if run_sandbox:
        sand = mechanical_sandbox_check(insight_obj, result)
        _print_section_banner("Sandbox (mechanical)")
        print(json.dumps(sand, indent=2))
        if check and not sand["good_copy"]:
          failures.append(f"{tc.get('name')}: sandbox failed\n{sand['eval_text']}")
    except Exception as e:
      print(f"Error: {e}")
      if check:
        failures.append(f"{tc.get('name')}: {e}")

  if check:
    print(f"\n{_TEST_SEPARATOR}\n")
    print(f"# Total tests: {len(cases)}\n")
    if failures:
      print("# CHECK FAILURES\n")
      for line in failures:
        print(line)
      raise SystemExit(1)
    if cases:
      print("# CHECK: all sandbox checks passed.\n")

  return failures


def main() -> None:
  parser = argparse.ArgumentParser(description=f"Optimizer for {TEMPLATE_NAME}")
  parser.add_argument(
    "--test",
    nargs="?",
    default=None,
    help="Batch index (0-based) or 'all'. Omit with --list to list cases.",
  )
  parser.add_argument("--no-sandbox", action="store_true", help="Skip mechanical sandbox checks.")
  parser.add_argument(
    "--check",
    action="store_true",
    help="Require mechanical sandbox checks to pass; exit non-zero on failure.",
  )
  parser.add_argument("--list", action="store_true", help="List available test cases and exit.")
  args = parser.parse_args()

  if args.list:
    print("Available test cases:")
    for tc in TEST_CASES:
      batch = tc.get("batch")
      batch_s = str(batch) if isinstance(batch, int) else "—"
      print(f"  batch {batch_s}: {tc.get('name')}")
    return

  batch_index = None
  if args.test is not None and str(args.test).lower() != "all":
    batch_index = int(args.test)

  test_optimizer(
    batch_index,
    run_sandbox=not args.no_sandbox,
    check=args.check,
  )


if __name__ == "__main__":
  main()
