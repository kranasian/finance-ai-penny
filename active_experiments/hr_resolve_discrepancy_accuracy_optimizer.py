"""
Resolve-discrepancy rubric optimizer — **accuracy** only.

Grades only the **accuracy** axis for ``Hr:ResolveDiscrepancyAccuracy`` checker templates.

Run from ``finance-ai-penny`` repo root:

  python3 active_experiments/hr_resolve_discrepancy_accuracy_optimizer.py --test all
  python3 active_experiments/hr_resolve_discrepancy_accuracy_optimizer.py --test all --check
  python3 active_experiments/hr_resolve_discrepancy_accuracy_optimizer.py --batch 1 --check
  python3 active_experiments/hr_resolve_discrepancy_accuracy_optimizer.py --model gemini-flash-lite-latest
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

try:
  from dotenv import load_dotenv
except Exception:  # pragma: no cover
  load_dotenv = None

try:
  from google import genai  # type: ignore[import-not-found]
  from google.genai import types  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
  genai = None
  types = None

if load_dotenv is not None:
  load_dotenv()

_TEST_SEPARATOR = "=" * 72
_SECTION_RULE = "-" * 72


SYSTEM_PROMPT = """Grade **accuracy** only for resolve-discrepancy outcomes. Return JSON `{score, notes}` (integer 1–5; one sentence `notes`).

You are a **strict rubric grader** for the checker bundle below (XML-style wrappers).

**Bundle:** `<RESOLVE_CONTEXT>` … `<RESOLVE_OUTCOME>` … `<RESOLVE_TOOL_CALLS>`.

- **`<RESOLVE_CONTEXT>`** — Prior rationalize, `# Insight Metadata`, `# Discrepancy`.
- **`<RESOLVE_OUTCOME>`** — `# Is Discrepancy`, `# Reason`, `# Resolution`.
- **`<RESOLVE_TOOL_CALLS>`** — This run's tool trace.

Grade **only** what is visible. Do not invent missing data.

**Out of scope — never lower accuracy solely for these (completeness handles them):**
- Skipped investigation tools before verdict.
- Yes without resolution tool calls, No with resolution calls, or phantom `## Resolution` blocks without matching tool calls.
- Missing headings or `No action.` formatting when verdict is No.

**Accuracy axis:**
1. **Verdict:** **Yes** only for **factual** contradiction (direction up vs down, Explain **$** vs tools/context, wrong category slug, timing misread). **No** when the dispute is **magnitude-only** (slightly vs significantly) while totals/direction agree. **No** when $0 in a partial window is **expected payroll timing**, not a decline.
2. **Grounding:** `# Reason` must not invent merchants, amounts, or dates absent from `<RESOLVE_CONTEXT>` or investigation tool results in `<RESOLVE_TOOL_CALLS>`.
3. **Resolution fit (when Yes and resolution tools appear):**
   - **`hide`** — wrong/misleading framing, false direction, or materially incorrect insight.
   - **`update_score`** — narrative mostly acceptable; only urgency misfit.
   - **`update_forecast`** — forecast stale/wrong; use canonical slug (`shelter_home`, `shelter_utilities`, `income_salary`, …) and `month_date` `YYYY-MM-01`.
   Prefer **`hide`** when the insight is materially misleading despite tolerable dollar amounts.

**Category slugs (when `update_forecast` or text cites a category):** must be exact tokens such as `shelter_home`, `shelter_utilities`, `income_salary`, `meals_groceries` — not Title Case labels ("Home", "Utilities").

**Calibration:** **5** = no meaningful accuracy gap. **4** = one minor gap. **3** = clear fixable issue (e.g. `update_score` instead of `hide` for false direction). **2** = several problems. **1** = verdict largely wrong (Yes on magnitude-only, No when tools/context contradict the insight, invented facts).

Return **only** the JSON object (`score`, `notes`).
"""


def _build_output_schema(_types: Any) -> Any:
  return _types.Schema(
    type=_types.Type.OBJECT,
    required=["score", "notes"],
    properties={
      "score": _types.Schema(type=_types.Type.INTEGER, description="Integer 1–5."),
      "notes": _types.Schema(type=_types.Type.STRING, description="One short sentence."),
    },
  )


TEST_CASES: list[dict[str, Any]] = [
  {
    "name": "accurate_yes_hide_flat_home_direction",
    "batch": 1,
    "input": """<RESOLVE_CONTEXT>

# Rationalize What

Explain: Home is slightly up the first 24 days of this month at $2850 (vs April). Shelter is thus slightly up the first 24 days of this month to $3211. (partial month May 1-31, 2026)

Category Taxonomy: parent Shelter (shelter), with leaf categories Home (shelter_home), Utilities (shelter_utilities), Upkeep (shelter_upkeep)

# Rationalize Response

## Figures

*   **Home (shelter_home)**
    *   May 2026: from May 1-24, $2850
    *   Apr 2026: from Apr 1-24, $2850 · entire month $2850
    *   Mar 2026: from Mar 1-24, $2850 · entire month $2850
*   **Shelter (shelter)**
    *   May 2026: from May 1-24, $3211
    *   Apr 2026: from Apr 1-24, $3184 · entire month $3184

## Drivers

*   **Home** is slightly up in May 2026 ($2850) compared to April ($2850).

# Insight Metadata

- Urgency Score: 2.4
- Home Forecast for first 24 days of this month: $2850

# Discrepancy

Home is slightly up the first 24 days of this month at $2850 (vs April). Transaction check confirms $2850 paid to Property Group LLC on 2026-05-03 and 2026-04-03; Home is identical in April and May ($2850). The insight claim of an increase is factually incorrect.

</RESOLVE_CONTEXT>

<RESOLVE_OUTCOME>

# Is Discrepancy
Yes

# Reason
Home is flat at $2850 in both partial periods; the slightly up direction is unsupported.

# Resolution

## Resolution 1
- Tool: hide
- Args: {}
- Status: success
- Message: Insight hidden.

</RESOLVE_OUTCOME>

<RESOLVE_TOOL_CALLS>

# Round 1

## Invoked tools

1. **`lookup_user_aggregate_spending`**

```json
{
  "category": "shelter_home",
  "periods": 3
}
```

# Round 2

## Invoked tools

1. **`hide`**

```json
{}
```

</RESOLVE_TOOL_CALLS>
""",
    "output": '{"score": 5, "notes": "Yes and hide match a factual direction contradiction with grounded $2850 evidence."}',
  },
  {
    "name": "accurate_yes_hide_utilities_direction",
    "batch": 1,
    "input": """<RESOLVE_CONTEXT>

# Rationalize What

Explain: Utilities is slightly down the first 21 days of this month at $343. Shelter is thus slightly up the first 21 days of this month to $3193. (partial month May 1-31, 2026)

Category Taxonomy: parent Shelter (shelter), with leaf categories Home (shelter_home), Utilities (shelter_utilities), Upkeep (shelter_upkeep)

# Rationalize Response

## Figures

*   **Utilities (shelter_utilities)**
    *   May 2026: from May 1-21, $343
    *   Apr 2026: from Apr 1-21, $334 · entire month $334
    *   Mar 2026: from Mar 1-21, $379 · entire month $379
*   **Shelter (shelter)**
    *   May 2026: from May 1-21, $3193
    *   Apr 2026: from Apr 1-21, $3184 · entire month $3184

## Drivers

*   Utilities is slightly down in May vs April.
*   Shelter is slightly up driven by utilities.

# Insight Metadata

- Urgency Score: 2.0
- Utilities Forecast for first 21 days of this month: $340

# Discrepancy

Utilities is slightly down the first 21 days of this month at $343. May utilities are $343.05 vs April $334.11 — utilities are up month-over-month, not down. The utilities direction in the insight contradicts tool-backed totals.

</RESOLVE_CONTEXT>

<RESOLVE_OUTCOME>

# Is Discrepancy
Yes

# Reason
Utilities are up $343.05 vs $334.11; the insight wrongly says down.

# Resolution

## Resolution 1
- Tool: hide
- Args: {}
- Status: success
- Message: Insight hidden.

</RESOLVE_OUTCOME>

<RESOLVE_TOOL_CALLS>

# Round 1

## Invoked tools

1. **`lookup_user_aggregate_spending`**

```json
{
  "category": "shelter_utilities",
  "periods": 3
}
```

# Round 2

## Invoked tools

1. **`hide`**

```json
{}
```

</RESOLVE_TOOL_CALLS>
""",
    "output": '{"score": 5, "notes": "Correct Yes for direction mismatch; hide is appropriate for a misleading insight."}',
  },
  {
    "name": "accurate_yes_hide_salary_timing_not_decline",
    "batch": 1,
    "input": """<RESOLVE_CONTEXT>

# Rationalize What

Explain: Salary is significantly down the first 2 days of this week at $0. (partial week May 24-30, 2026)

Category Taxonomy: parent Income (income), with leaf categories Salary (income_salary)

# Rationalize Response

## Figures

*   **Week of May 24–30, 2026 (Salary):** from May 24–26, $0
*   **Week of May 17–23, 2026 (Salary):** from May 17–19, $0 · entire week $0
*   **Week of May 10–16, 2026 (Salary):** from May 10–12, $0 · entire week $3,184.62

## Drivers

Salary is significantly down the first 2 days of this week at $0 because payroll pays on the 1st and 15th.

# Insight Metadata

- Urgency Score: 3.1

# Discrepancy

Salary is significantly down the first 2 days of this week at $0. Income SQL confirms bi-monthly deposits on the 1st and 15th — $0 through day 2 is expected timing, not a drop in salary.

</RESOLVE_CONTEXT>

<RESOLVE_OUTCOME>

# Is Discrepancy
Yes

# Reason
$0 through day 2 matches bi-monthly timing; significantly down is misleading, not a factual decline.

# Resolution

## Resolution 1
- Tool: hide
- Args: {}
- Status: success
- Message: Insight hidden.

</RESOLVE_OUTCOME>

<RESOLVE_TOOL_CALLS>

# Round 1

## Invoked tools

1. **`retrieve_user_income_transactions_by_sql`**

```json
{
  "category": "income_salary",
  "limit": 25
}
```

# Round 2

## Invoked tools

1. **`hide`**

```json
{}
```

</RESOLVE_TOOL_CALLS>
""",
    "output": '{"score": 5, "notes": "Yes is warranted for misleading down framing on expected pre-payday $0; hide fits."}',
  },
  {
    "name": "accurate_no_magnitude_only_dispute",
    "batch": 2,
    "input": """<RESOLVE_CONTEXT>

# Rationalize What

Explain: Home is slightly up the first 24 days of this month at $2850 (vs April). (partial month May 1-31, 2026)

# Rationalize Response

## Figures

*   **Home (shelter_home)**
    *   May 2026: from May 1-24, $2850
    *   Apr 2026: from Apr 1-24, $2850 · entire month $2850

## Drivers

*   **Home** is slightly up in May 2026 ($2850) compared to April ($2850).

# Insight Metadata

- Urgency Score: 2.4

# Discrepancy

Rationalize said slightly up instead of significantly up; May and April partial Home totals are both $2850 with no direction change.

</RESOLVE_CONTEXT>

<RESOLVE_OUTCOME>

# Is Discrepancy
No

# Reason
Only intensity wording differs; $2850 is flat in both partial months with no up/down contradiction.

# Resolution

No action.

</RESOLVE_OUTCOME>

<RESOLVE_TOOL_CALLS>

# Round 1

## Invoked tools

1. **`lookup_user_aggregate_spending`**

```json
{
  "category": "shelter_home",
  "periods": 3
}
```

</RESOLVE_TOOL_CALLS>
""",
    "output": '{"score": 5, "notes": "Correct No — magnitude wording alone is not a factual discrepancy."}',
  },
  {
    "name": "inaccurate_yes_magnitude_only",
    "batch": 2,
    "input": """<RESOLVE_CONTEXT>

# Rationalize What

Explain: Home is slightly up the first 24 days of this month at $2850 (vs April). (partial month May 1-31, 2026)

# Rationalize Response

## Figures

*   **Home (shelter_home)**
    *   May 2026: from May 1-24, $2850
    *   Apr 2026: from Apr 1-24, $2850 · entire month $2850

## Drivers

*   **Home** is slightly up in May 2026 ($2850) compared to April ($2850).

# Insight Metadata

- Urgency Score: 2.4

# Discrepancy

Insight used slightly instead of significantly while Home stayed $2850 in both partial months.

</RESOLVE_CONTEXT>

<RESOLVE_OUTCOME>

# Is Discrepancy
Yes

# Reason
Wording intensity does not match the user's preferred adverb.

# Resolution

## Resolution 1
- Tool: hide
- Args: {}
- Status: success
- Message: Insight hidden.

</RESOLVE_OUTCOME>

<RESOLVE_TOOL_CALLS>

# Round 1

## Invoked tools

1. **`hide`**

```json
{}
```

</RESOLVE_TOOL_CALLS>
""",
    "output": '{"score": 1, "notes": "Yes for magnitude-only dispute is wrong when totals and direction are flat."}',
  },
  {
    "name": "inaccurate_no_when_direction_contradicted",
    "batch": 2,
    "input": """<RESOLVE_CONTEXT>

# Rationalize What

Explain: Utilities is slightly down the first 21 days of this month at $343. (partial month May 1-31, 2026)

# Rationalize Response

## Figures

*   **Utilities (shelter_utilities)**
    *   May 2026: from May 1-21, $343
    *   Apr 2026: from Apr 1-21, $334 · entire month $334

## Drivers

*   Utilities is slightly down in May vs April.

# Insight Metadata

- Urgency Score: 2.0

# Discrepancy

Utilities is slightly down at $343. May utilities are $343.05 vs April $334.11 — utilities are up month-over-month, not down.

</RESOLVE_CONTEXT>

<RESOLVE_OUTCOME>

# Is Discrepancy
No

# Reason
The insight still communicates useful shelter context even if utilities direction is debatable.

# Resolution

No action.

</RESOLVE_OUTCOME>

<RESOLVE_TOOL_CALLS>

# Round 1

## Invoked tools

1. **`lookup_user_aggregate_spending`**

```json
{
  "category": "shelter_utilities",
  "periods": 3
}
```

</RESOLVE_TOOL_CALLS>
""",
    "output": '{"score": 1, "notes": "No is wrong when discrepancy and aggregates show utilities up, not down."}',
  },
  {
    "name": "inaccurate_update_score_when_insight_should_hide",
    "batch": 2,
    "input": """<RESOLVE_CONTEXT>

# Rationalize What

Explain: Home is slightly up the first 24 days of this month at $2850 (vs April). (partial month May 1-31, 2026)

# Rationalize Response

## Figures

*   **Home (shelter_home)**
    *   May 2026: from May 1-24, $2850
    *   Apr 2026: from Apr 1-24, $2850 · entire month $2850
    *   Mar 2026: from Mar 1-24, $2850 · entire month $2850

## Drivers

*   **Home** is slightly up in May 2026 ($2850) compared to April ($2850).

# Insight Metadata

- Urgency Score: 2.4

# Discrepancy

Home is slightly up at $2850 (vs April). Home is identical in April and May ($2850). The insight claim of an increase is factually incorrect.

</RESOLVE_CONTEXT>

<RESOLVE_OUTCOME>

# Is Discrepancy
Yes

# Reason
Home is flat; direction claim is false but figures are otherwise fine.

# Resolution

## Resolution 1
- Tool: update_score
- Args: {"combined_urgency_score": 1.0}
- Status: success
- Message: Score updated.

</RESOLVE_OUTCOME>

<RESOLVE_TOOL_CALLS>

# Round 1

## Invoked tools

1. **`lookup_user_aggregate_spending`**

```json
{
  "category": "shelter_home",
  "periods": 3
}
```

# Round 2

## Invoked tools

1. **`update_score`**

```json
{
  "combined_urgency_score": 1.0
}
```

</RESOLVE_TOOL_CALLS>
""",
    "output": '{"score": 3, "notes": "Yes is right but update_score understates a materially misleading wrong-direction insight that should be hidden."}',
  },
]


class ResolveDiscrepancyAccuracyCheckerOptimizer:
  def __init__(
    self,
    model_name: str = "gemini-flash-lite-latest",
    *,
    max_output_tokens: int = 256,
    thinking_budget: int = 0,
  ):
    if genai is None or types is None:
      raise ImportError(
        "Missing dependency for Gemini client. Install `google-genai` (and ensure your venv is active) "
        "to run this script."
      )
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set.")
    self.client = genai.Client(api_key=api_key)
    self._types = types
    self.model_name = model_name
    self.temperature = 0.0
    self.top_p = 0.95
    self.max_output_tokens = max_output_tokens
    self.thinking_budget = thinking_budget
    self.system_prompt = SYSTEM_PROMPT
    self.safety_settings = [
      self._types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      self._types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      self._types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      self._types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    ]
    self.output_schema = _build_output_schema(self._types)

  def grade(self, bundled_input: str) -> Dict[str, Any]:
    user_msg = (bundled_input or "").strip()
    t = self._types
    contents = [t.Content(role="user", parts=[t.Part.from_text(text=user_msg)])]
    cfg = t.GenerateContentConfig(
      temperature=self.temperature,
      top_p=self.top_p,
      max_output_tokens=self.max_output_tokens,
      safety_settings=self.safety_settings,
      system_instruction=[t.Part.from_text(text=self.system_prompt)],
      thinking_config=t.ThinkingConfig(thinking_budget=self.thinking_budget, include_thoughts=True),
      response_mime_type="application/json",
      response_schema=self.output_schema,
    )
    out = self.client.models.generate_content(model=self.model_name, contents=contents, config=cfg)
    thought_summary = ""
    if hasattr(out, "candidates") and out.candidates:
      for candidate in out.candidates:
        if hasattr(candidate, "content") and candidate.content:
          if hasattr(candidate.content, "parts") and candidate.content.parts:
            for part in candidate.content.parts:
              if hasattr(part, "thought") and part.thought:
                if hasattr(part, "text") and part.text:
                  thought_summary += part.text
    if thought_summary.strip():
      print(f"{'=' * 80}")
      print("THOUGHT SUMMARY:")
      print(thought_summary.strip())
      print("=" * 80)
    text = (out.text or "").strip()
    try:
      return json.loads(text)
    except Exception:
      s = text[text.find("{") : text.rfind("}") + 1] if ("{" in text and "}" in text) else "{}"
      return json.loads(s)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--test", type=str, default=None, help="Test name, index, or 'all'.")
  parser.add_argument("--batch", type=int, default=None, help="Run all tests in batch N.")
  parser.add_argument("--model", type=str, default="gemini-flash-lite-latest")
  parser.add_argument("--max-output-tokens", type=int, default=256)
  parser.add_argument("--thinking-budget", type=int, default=0)
  parser.add_argument(
    "--check",
    action="store_true",
    help="After grading, require integer score to match expected output JSON. Exits non-zero on failure.",
  )
  args = parser.parse_args()

  if args.test is None and args.batch is None:
    print("Available test cases:")
    for i, tc in enumerate(TEST_CASES):
      batch = tc.get("batch")
      batch_s = str(batch) if isinstance(batch, int) else "—"
      print(f"  {i}: {tc.get('name')} (batch {batch_s})")
    return

  opt = ResolveDiscrepancyAccuracyCheckerOptimizer(
    model_name=args.model,
    max_output_tokens=args.max_output_tokens,
    thinking_budget=args.thinking_budget,
  )

  if args.batch is not None:
    cases = [(i, tc) for i, tc in enumerate(TEST_CASES) if int(tc.get("batch") or 0) == int(args.batch)]
    if not cases:
      raise SystemExit(f"No tests found for batch={args.batch}")
  elif (args.test or "").strip().lower() == "all":
    cases = list(enumerate(TEST_CASES))
  else:
    if (args.test or "").isdigit():
      idx = int(args.test)
      if idx < 0 or idx >= len(TEST_CASES):
        raise SystemExit(f"Test index out of range: {idx}")
      cases = [(idx, TEST_CASES[idx])]
    else:
      idx_tc = next(((i, t) for i, t in enumerate(TEST_CASES) if t.get("name") == args.test), None)
      if idx_tc is None:
        raise SystemExit(f"Unknown test: {args.test!r}")
      cases = [idx_tc]

  failures: list[str] = []

  for run_i, (case_index, tc) in enumerate(cases):
    if run_i:
      print(f"\n{_TEST_SEPARATOR}\n")
    batch = tc.get("batch")
    batch_s = str(batch) if isinstance(batch, int) else "—"
    print(f"# Test: {case_index}  {tc['name']}  (batch {batch_s})\n")
    print(f"\n{_SECTION_RULE}\n# LLM Checker Input\n{_SECTION_RULE}\n")
    print(tc["input"])
    result = opt.grade(tc["input"])
    print(f"\n{_SECTION_RULE}\n# LLM Checker Response\n{_SECTION_RULE}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.check:
      exp = None
      if tc.get("output"):
        try:
          exp = json.loads(tc.get("output"))
        except Exception:
          exp = None
      if not isinstance(exp, dict) or "score" not in exp:
        failures.append(f"{tc.get('name')}: invalid expected output JSON")
      else:
        try:
          got = int(result.get("score"))
          want = int(exp["score"])
        except Exception:
          failures.append(f"{tc.get('name')}: non-integer score")
        else:
          if got != want:
            failures.append(f"{tc.get('name')}: score {got} != expected {want}")
    if tc.get("output") is not None:
      print(f"\n{_SECTION_RULE}\n# Expected Output\n{_SECTION_RULE}\n")
      print(tc["output"])

  print(f"\n{_TEST_SEPARATOR}\n")
  print(f"# Total tests: {len(cases)}\n")
  if args.check and failures:
    print("# CHECK FAILURES\n")
    for line in failures:
      print(line)
    raise SystemExit(1)
  if args.check and not failures:
    print("# CHECK: all cases passed score match.\n")


if __name__ == "__main__":
  main()
