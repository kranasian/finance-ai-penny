"""
Resolve-discrepancy rubric optimizer — **completeness** only.

Grades only the **completeness** axis for ``Hr:ResolveDiscrepancyCompleteness`` checker templates.

Run from ``finance-ai-penny`` repo root:

  python3 active_experiments/hr_resolve_discrepancy_completeness_optimizer.py --test all
  python3 active_experiments/hr_resolve_discrepancy_completeness_optimizer.py --test all --check
  python3 active_experiments/hr_resolve_discrepancy_completeness_optimizer.py --batch 1 --check
  python3 active_experiments/hr_resolve_discrepancy_completeness_optimizer.py --model gemini-flash-lite-latest
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


SYSTEM_PROMPT = """You are a strict **completeness-only** grader for resolve-discrepancy outcomes. Return JSON `{score, notes}` (integer 1–5; one sentence `notes`).

**Bundle:** `<RESOLVE_CONTEXT>` … `<RESOLVE_OUTCOME>` … `<RESOLVE_TOOL_CALLS>` (this resolve run only: `# Round N`, then `## Invoked tools` per round). Grade only visible text; do not invent facts. **Ignore factual correctness of Yes/No** and whether resolution tool choice is ideal (accuracy).

**`<RESOLVE_CONTEXT>`** — Prior rationalize markdown plus `# Insight Metadata` and `# Discrepancy` (the Hermes user task).

**`<RESOLVE_OUTCOME>`** — Agent final markdown: exactly `# Is Discrepancy`, `# Reason`, `# Resolution` (no other top-level headings).

**`<RESOLVE_TOOL_CALLS>`** — Investigation + resolution tools from the run (`describe_categories`, `retrieve_user_*_by_sql`, `lookup_user_aggregate_spending`, `hide`, `update_score`, `update_forecast`).

**Completeness (workflow + shape):**
1. **Verify:** At least one **investigation** tool in `<RESOLVE_TOOL_CALLS>` before the final verdict when the discrepancy requires fresh evidence (direction/$/slug/timing). Re-stating `# Discrepancy` alone is **not** Verify.
2. **Verdict ↔ tools:** If `# Is Discrepancy` is **Yes**, `<RESOLVE_TOOL_CALLS>` must include **≥1** resolution tool (`hide`, `update_score`, `update_forecast`). If **No**, **no** resolution tools.
3. **Resolution section:** **No** → `# Resolution` must be exactly `No action.` **Yes** → one `## Resolution N` block per executed resolution tool, in call order, with `Tool`, `Args`, `Status`, `Message`. Never claim a resolution tool in markdown that is absent from tool calls.
4. **Structure:** Required headings only; no full-response code fence.

**Out of scope for completeness (do not penalize here):**
- Whether Yes/No is factually correct vs tools (accuracy).
- Whether `hide` vs `update_score` vs `update_forecast` was the best fix (accuracy).
- Magnitude wording disputes when the agent still ran Verify (accuracy handles verdict).

**Scoring anchors (worst match wins):**
- **1** — Yes without any resolution tool call; No with resolution tool(s); phantom `## Resolution` for a tool not in calls; or missing required headings.
- **2** — Multiple protocol breaks (e.g. skipped Verify **and** Yes without resolution).
- **3** — Single clear gap: skipped Verify, thin investigation when SQL/aggregates were needed, or minor Resolution formatting drift while tools match verdict.
- **4** — One minor shape issue (extra blank lines, weak Args line) while workflow otherwise holds.
- **5** — Verify → verdict → resolution/markdown all consistent with the rules above.

**`notes`:** One sentence; name the gap or affirm workflow alignment.
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
    "name": "complete_yes_verify_hide_and_resolution_blocks",
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
Home is flat at $2850 in both partial periods; the insight direction is unsupported.

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
    "output": '{"score": 5, "notes": "Verify used an investigation tool before verdict; Yes includes executed hide with a matching Resolution block."}',
  },
  {
    "name": "complete_no_verify_and_no_action",
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

## Drivers

*   **Home** is slightly up in May 2026 ($2850) compared to April ($2850).

# Insight Metadata

- Urgency Score: 2.4

# Discrepancy

The rationalize used the word slightly instead of significantly for Home; May and April partial totals are both $2850 and direction is flat.

</RESOLVE_CONTEXT>

<RESOLVE_OUTCOME>

# Is Discrepancy
No

# Reason
Only magnitude wording differs; $2850 matches in both partial months with no direction contradiction.

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
    "output": '{"score": 5, "notes": "Investigation before No; required headings present; No action with no resolution tools."}',
  },
  {
    "name": "yes_without_any_resolution_tool_calls",
    "batch": 1,
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

Home is slightly up at $2850 (vs April). Home is identical in April and May ($2850). The insight claim of an increase is factually incorrect.

</RESOLVE_CONTEXT>

<RESOLVE_OUTCOME>

# Is Discrepancy
Yes

# Reason
Home is flat; insight direction is wrong.

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
    "output": '{"score": 1, "notes": "Marked Yes but never called hide, update_score, or update_forecast while Resolution says No action."}',
  },
  {
    "name": "phantom_hide_claimed_in_markdown_not_executed",
    "batch": 2,
    "input": """<RESOLVE_CONTEXT>

# Rationalize What

Explain: Salary is significantly down the first 2 days of this week at $0. (partial week May 24-30, 2026)

# Rationalize Response

## Figures

*   **Week of May 24–30, 2026 (Salary):** from May 24–26, $0
*   **Week of May 17–23, 2026 (Salary):** from May 17–19, $0 · entire week $0

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
Misleading significantly down label on expected pre-payday $0.

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
  "limit": 20
}
```

</RESOLVE_TOOL_CALLS>
""",
    "output": '{"score": 1, "notes": "Outcome claims hide in Resolution but hide never appears in tool calls."}',
  },
  {
    "name": "skipped_verify_before_yes_and_hide",
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

Home is slightly up at $2850 (vs April). Home is identical in April and May ($2850). The insight claim of an increase is factually incorrect.

</RESOLVE_CONTEXT>

<RESOLVE_OUTCOME>

# Is Discrepancy
Yes

# Reason
Home direction contradicts flat totals.

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
    "output": '{"score": 3, "notes": "Yes with hide executed but no investigation tool ran before the verdict."}',
  },
  {
    "name": "no_verdict_with_resolution_tool_called",
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

Only wording intensity differs; totals are flat at $2850.

</RESOLVE_CONTEXT>

<RESOLVE_OUTCOME>

# Is Discrepancy
No

# Reason
Magnitude-only dispute; no factual contradiction.

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
  "periods": 2
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
    "output": '{"score": 1, "notes": "Marked No but hide was called, violating the No-resolution protocol."}',
  },
]


class ResolveDiscrepancyCompletenessCheckerOptimizer:
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

  opt = ResolveDiscrepancyCompletenessCheckerOptimizer(
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
