"""
Review next goal steps rubric optimizer — **accuracy** only.

Grades only the **accuracy** axis for ``Hr:ReviewNextGoalStepsAccuracy`` checker templates.

Run from ``finance-ai-penny`` repo root:

  python3 active_experiments/hr_review_next_goal_steps_accuracy_optimizer.py --test all
  python3 active_experiments/hr_review_next_goal_steps_accuracy_optimizer.py --test all --check
  python3 active_experiments/hr_review_next_goal_steps_accuracy_optimizer.py --batch 1 --check
  python3 active_experiments/hr_review_next_goal_steps_accuracy_optimizer.py --model gemini-flash-lite-latest
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


SYSTEM_PROMPT = """Grade **accuracy** only for ``review_next_goal_steps`` outcomes. Return JSON `{score, notes}` (integer 1–5).

**Bundle:** `<ACTIVE_GOALS>` … `<PROPOSED_GOAL_STEPS>` … `<ANALYZE_PROPOSAL>` … `<ANALYZE_TOOL_CALLS>` (`# Round N`, `## Invoked tools`). Grade only visible text.

**Notes:** List **every** distinct accuracy violation in `notes` (semicolon-separated); cite slug tokens in backticks. Score **5** → one brief affirmation only. Never mention missing tools, lookup count, or sectioning unless there is a factual contradiction.

**Out of scope for accuracy (completeness handles these — never lower the accuracy score for these alone):**
- Skipped or insufficient `lookup_user_aggregate_spending` calls.
- Missing propose tools, coherence count mismatches, or missing headings.
- Duplicate identical bullets.
- Do **not** mention or penalize absent tools / lookup count in `notes` unless there is a factual contradiction with visible proposal or tool args.

**Accuracy**
1. **Already-covered routing** — Drop to Already covered only when candidate **$** matches the matched `<ACTIVE_GOALS>` `target_amount` at the same granularity + category/topic. Different **$** → must keep as update, not Already covered. **Score 3** when this is the only accuracy issue (even if `<ANALYZE_TOOL_CALLS>` shows no propose tools — completeness owns that gap).
2. **Create vs update** — Existing covering row + different **$** → `propose_update_goal` with correct `existing_goal_id` and matched `goal_title`. No covering row → `propose_create_goal`. Never create when an update row exists.
3. **Category scope (updates)** — `category` on update **must equal** the active row slug; flag parent→leaf narrowing in **tool args only** (e.g. `leisure` → `leisure_entertainment`). Explanatory rationale/Finalized wording may reference the candidate's sub-topic (e.g. “entertainment”, “ticket expenses”) when tool args keep the matched parent slug — **do not** penalize that alone. Parent→leaf narrowing in tool args alone → **3** (clear fixable), not **4**.
4. **Grain** — Weekly candidates/goals → weekly lookups/horizon; monthly → monthly. Never cross grains in tool args vs active row.
5. **Amount grounding** — `target_amount` should be plausible vs lookup results in `<ANALYZE_TOOL_CALLS>` when present (partial/full spending fields). Flag clearly implausible amounts with no trail support.
6. **Rationale ↔ tool** — `Create new goal:` only on create; `Update existing goal: <goal_title>:` only on update; rationale **$** must match `target_amount`.
7. **Tool args** — `propose_create_goal` must not pass `existing_goal_id`; `propose_update_goal` must include it. Invalid category slug → cap **4**.

**Calibration:** **5** none · **4** minor wording/slug nit without routing break · **3** one clear fixable routing/scope/grounding issue · **2** several distinct accuracy violations · **1** wholesale failed routing (e.g. create when must update **and** wrong category **and** implausible amount). `score` reflects the worst **accuracy** band only — never **1**–**2** solely because propose tools or lookups are missing.
"""


def _build_output_schema(_types: Any) -> Any:
  return _types.Schema(
    type=_types.Type.OBJECT,
    required=["score", "notes"],
    properties={
      "score": _types.Schema(type=_types.Type.INTEGER, description="Integer 1–5."),
      "notes": _types.Schema(
        type=_types.Type.STRING,
        description="Semicolon-separated accuracy issues; score 5 = brief affirmation only.",
      ),
    },
  )


_SIX_WEEKLY_LEISURE_LOOKUPS_WITH_AMOUNTS = """
1. **`lookup_user_aggregate_spending`**
```json
{"granularity": "weekly", "category": ["leisure"], "date_in_range": "2026-04-06", "full_week_range_spending": 82.0}
```
2. **`lookup_user_aggregate_spending`**
```json
{"granularity": "weekly", "category": ["leisure"], "date_in_range": "2026-04-13", "full_week_range_spending": 74.0}
```
3. **`lookup_user_aggregate_spending`**
```json
{"granularity": "weekly", "category": ["leisure"], "date_in_range": "2026-04-20", "full_week_range_spending": 68.0}
```
4. **`lookup_user_aggregate_spending`**
```json
{"granularity": "weekly", "category": ["leisure"], "date_in_range": "2026-04-27", "full_week_range_spending": 71.0}
```
5. **`lookup_user_aggregate_spending`**
```json
{"granularity": "weekly", "category": ["leisure"], "date_in_range": "2026-05-04", "full_week_range_spending": 79.0}
```
6. **`lookup_user_aggregate_spending`**
```json
{"granularity": "weekly", "category": ["leisure"], "date_in_range": "2026-05-11", "full_week_range_spending": 65.0}
```
""".strip()


TEST_CASES: list[dict[str, Any]] = [
  {
    "name": "accurate_weekly_update_same_category",
    "batch": 1,
    "input": f"""<ACTIVE_GOALS>

1. goal_id=2751 | goal_title="Weekly Leisure Budget" | type=category | granularity=weekly | category=leisure | target_amount=$75

</ACTIVE_GOALS>

<PROPOSED_GOAL_STEPS>

1. Update existing goal: Weekly Leisure Budget to $50.

</PROPOSED_GOAL_STEPS>

<ANALYZE_PROPOSAL>

# Analysis output

## Finalized proposed goal steps

1. Update existing goal: Weekly Leisure Budget to $50.

## Already covered by existing goals

1. - (none)

</ANALYZE_PROPOSAL>

<ANALYZE_TOOL_CALLS>

# Round 1

## Invoked tools

{_SIX_WEEKLY_LEISURE_LOOKUPS_WITH_AMOUNTS}

# Round 2

## Invoked tools

1. **`propose_update_goal`**
```json
{{"existing_goal_id": 2751, "goal_title": "Weekly Leisure Budget", "goal_type": "spending_budget", "category": "leisure", "time_horizon": "weekly", "target_amount": 50, "rationale": "Update existing goal: Weekly Leisure Budget to $50."}}
```

</ANALYZE_TOOL_CALLS>
""",
    "output": '{"score": 5, "notes": "Correct update routing on goal_id 2751 with matching leisure slug and plausible weekly amount."}',
  },
  {
    "name": "wrong_already_covered_amount_mismatch",
    "batch": 1,
    "input": """<ACTIVE_GOALS>

1. goal_id=2751 | goal_title="Weekly Leisure Budget" | type=category | granularity=weekly | category=leisure | target_amount=$75

</ACTIVE_GOALS>

<PROPOSED_GOAL_STEPS>

1. Update existing goal: Weekly Leisure Budget to $50.

</PROPOSED_GOAL_STEPS>

<ANALYZE_PROPOSAL>

# Analysis output

## Finalized proposed goal steps

1. - (none)

## Already covered by existing goals

1. Update existing goal: Weekly Leisure Budget to $50. — Already covered by existing Weekly Leisure Budget goal.

</ANALYZE_PROPOSAL>

<ANALYZE_TOOL_CALLS>

# Round 1

## Invoked tools

1. (none)

</ANALYZE_TOOL_CALLS>
""",
    "output": '{"score": 3, "notes": "Candidate $50 ≠ active target_amount $75 so it must be kept as an update, not Already covered."}',
  },
  {
    "name": "parent_to_leaf_category_on_update",
    "batch": 1,
    "input": f"""<ACTIVE_GOALS>

1. goal_id=2751 | goal_title="Weekly Leisure Budget" | type=category | granularity=weekly | category=leisure | target_amount=$75

</ACTIVE_GOALS>

<PROPOSED_GOAL_STEPS>

1. Update existing goal: Weekly Leisure Budget to $50.

</PROPOSED_GOAL_STEPS>

<ANALYZE_PROPOSAL>

# Analysis output

## Finalized proposed goal steps

1. Update existing goal: Weekly Leisure Budget to $50.

## Already covered by existing goals

1. - (none)

</ANALYZE_PROPOSAL>

<ANALYZE_TOOL_CALLS>

# Round 1

## Invoked tools

{_SIX_WEEKLY_LEISURE_LOOKUPS_WITH_AMOUNTS}

# Round 2

## Invoked tools

1. **`propose_update_goal`**
```json
{{"existing_goal_id": 2751, "goal_title": "Weekly Leisure Budget", "goal_type": "spending_budget", "category": "leisure_entertainment", "time_horizon": "weekly", "target_amount": 50, "rationale": "Update existing goal: Weekly Leisure Budget to $50."}}
```

</ANALYZE_TOOL_CALLS>
""",
    "output": '{"score": 3, "notes": "Update narrowed parent `leisure` to leaf `leisure_entertainment` instead of keeping the active row slug."}',
  },
  {
    "name": "create_rerouted_parent_update_with_context_wording",
    "batch": 1,
    "input": f"""<ACTIVE_GOALS>

1. goal_id=2751 | goal_title="Weekly Leisure Budget 🎯" | type=category | granularity=weekly | category=leisure | target_amount=$25
2. goal_id=2741 | goal_title="Feast For 💖 Less" | type=category | granularity=monthly | category=top_meals | target_amount=$1100
3. goal_id=2740 | goal_title="Savvy Bites 🍽️" | type=category | granularity=weekly | category=meals_dining_out | target_amount=$150

</ACTIVE_GOALS>

<PROPOSED_GOAL_STEPS>

1. Create new goal: Establish a weekly Entertainment goal of $50 to account for the irregular timing of major ticket purchases like Fandango for leisure_entertainment.

</PROPOSED_GOAL_STEPS>

<ANALYZE_PROPOSAL>

# Analysis output

## Finalized proposed goal steps

1. Update existing goal: Weekly Leisure Budget 🎯 to $50 to better accommodate irregular entertainment and ticket expenses.

## Already covered by existing goals

1. - (none)

</ANALYZE_PROPOSAL>

<ANALYZE_TOOL_CALLS>

# Round 1

## Invoked tools

{_SIX_WEEKLY_LEISURE_LOOKUPS_WITH_AMOUNTS}

# Round 2

## Invoked tools

1. **`propose_update_goal`**
```json
{{"existing_goal_id": 2751, "goal_title": "Weekly Leisure Budget 🎯", "goal_type": "spending_budget", "category": "leisure", "time_horizon": "weekly", "target_amount": 50, "rationale": "Update existing goal: Weekly Leisure Budget 🎯 to $50 to better accommodate irregular entertainment and ticket expenses."}}
```

</ANALYZE_TOOL_CALLS>
""",
    "output": '{"score": 5, "notes": "Correct create→parent update reroute on goal_id 2751 with parent `leisure` slug; explanatory entertainment wording is fine."}',
  },
  {
    "name": "correct_already_covered_same_amount",
    "batch": 2,
    "input": """<ACTIVE_GOALS>

1. goal_id=22 | goal_title="Clothing" | type=category | granularity=monthly | category=shopping_clothing | target_amount=$60

</ACTIVE_GOALS>

<PROPOSED_GOAL_STEPS>

1. Create new goal: $60 monthly Clothing budget.

</PROPOSED_GOAL_STEPS>

<ANALYZE_PROPOSAL>

# Analysis output

## Finalized proposed goal steps

1. - (none)

## Already covered by existing goals

1. Create new goal: $60 monthly Clothing budget. — Already covered by existing Clothing monthly budget goal.

</ANALYZE_PROPOSAL>

<ANALYZE_TOOL_CALLS>

# Round 1

## Invoked tools

1. (none)

</ANALYZE_TOOL_CALLS>
""",
    "output": '{"score": 5, "notes": "Correctly dropped duplicate when candidate $60 matches the active Clothing goal."}',
  },
  {
    "name": "create_when_update_row_exists",
    "batch": 2,
    "input": f"""<ACTIVE_GOALS>

1. goal_id=2751 | goal_title="Weekly Leisure Budget" | type=category | granularity=weekly | category=leisure | target_amount=$75

</ACTIVE_GOALS>

<PROPOSED_GOAL_STEPS>

1. Update existing goal: Weekly Leisure Budget to $50.

</PROPOSED_GOAL_STEPS>

<ANALYZE_PROPOSAL>

# Analysis output

## Finalized proposed goal steps

1. Create new goal: $50 weekly leisure budget.

## Already covered by existing goals

1. - (none)

</ANALYZE_PROPOSAL>

<ANALYZE_TOOL_CALLS>

# Round 1

## Invoked tools

{_SIX_WEEKLY_LEISURE_LOOKUPS_WITH_AMOUNTS}

7. **`propose_create_goal`**
```json
{{"goal_title": "Weekly Leisure Budget", "goal_type": "spending_budget", "category": "leisure", "time_horizon": "weekly", "target_amount": 50, "rationale": "Create new goal: $50 weekly leisure budget."}}
```

</ANALYZE_TOOL_CALLS>
""",
    "output": '{"score": 1, "notes": "Used propose_create_goal though active goal_id 2751 already covers leisure weekly and the candidate is an update."}',
  },
  {
    "name": "create_leaf_when_parent_row_exists",
    "batch": 2,
    "input": f"""<ACTIVE_GOALS>

1. goal_id=2751 | goal_title="Weekly Leisure Budget 🎯" | type=category | granularity=weekly | category=leisure | target_amount=$25

</ACTIVE_GOALS>

<PROPOSED_GOAL_STEPS>

1. Create new goal: Establish a weekly Entertainment goal of $50 for leisure_entertainment.

</PROPOSED_GOAL_STEPS>

<ANALYZE_PROPOSAL>

# Analysis output

## Finalized proposed goal steps

1. Create new goal: Establish a weekly Entertainment goal of $50 for leisure_entertainment.

## Already covered by existing goals

1. - (none)

</ANALYZE_PROPOSAL>

<ANALYZE_TOOL_CALLS>

# Round 1

## Invoked tools

{_SIX_WEEKLY_LEISURE_LOOKUPS_WITH_AMOUNTS}

7. **`propose_create_goal`**
```json
{{"goal_title": "Weekly Entertainment Budget", "goal_type": "spending_budget", "category": "leisure_entertainment", "time_horizon": "weekly", "target_amount": 50, "rationale": "Create new goal: Establish a weekly Entertainment goal of $50 for leisure_entertainment."}}
```

</ANALYZE_TOOL_CALLS>
""",
    "output": '{"score": 1, "notes": "Created leaf `leisure_entertainment` though active parent `leisure` goal_id 2751 already covers the candidate topic."}',
  },
  {
    "name": "monthly_grain_on_weekly_goal",
    "batch": 2,
    "input": f"""<ACTIVE_GOALS>

1. goal_id=2751 | goal_title="Weekly Leisure Budget" | type=category | granularity=weekly | category=leisure | target_amount=$75

</ACTIVE_GOALS>

<PROPOSED_GOAL_STEPS>

1. Update existing goal: Weekly Leisure Budget to $50.

</PROPOSED_GOAL_STEPS>

<ANALYZE_PROPOSAL>

# Analysis output

## Finalized proposed goal steps

1. Update existing goal: Weekly Leisure Budget to $50.

## Already covered by existing goals

1. - (none)

</ANALYZE_PROPOSAL>

<ANALYZE_TOOL_CALLS>

# Round 1

## Invoked tools

1. **`lookup_user_aggregate_spending`**
```json
{{"granularity": "monthly", "category": ["leisure"], "date_in_range": "2026-05-01", "full_month_range_spending": 220.0}}
```

2. **`propose_update_goal`**
```json
{{"existing_goal_id": 2751, "goal_title": "Weekly Leisure Budget", "goal_type": "spending_budget", "category": "leisure", "time_horizon": "monthly", "target_amount": 50, "rationale": "Update existing goal: Weekly Leisure Budget to $50."}}
```

</ANALYZE_TOOL_CALLS>
""",
    "output": '{"score": 3, "notes": "Weekly active goal updated with monthly grain in lookup and propose_update_goal time_horizon."}',
  },
  {
    "name": "implausible_amount_vs_spending_trail",
    "batch": 2,
    "input": f"""<ACTIVE_GOALS>

1. goal_id=2751 | goal_title="Weekly Leisure Budget" | type=category | granularity=weekly | category=leisure | target_amount=$75

</ACTIVE_GOALS>

<PROPOSED_GOAL_STEPS>

1. Update existing goal: Weekly Leisure Budget to $50.

</PROPOSED_GOAL_STEPS>

<ANALYZE_PROPOSAL>

# Analysis output

## Finalized proposed goal steps

1. Update existing goal: Weekly Leisure Budget to $500.

## Already covered by existing goals

1. - (none)

</ANALYZE_PROPOSAL>

<ANALYZE_TOOL_CALLS>

# Round 1

## Invoked tools

{_SIX_WEEKLY_LEISURE_LOOKUPS_WITH_AMOUNTS}

7. **`propose_update_goal`**
```json
{{"existing_goal_id": 2751, "goal_title": "Weekly Leisure Budget", "goal_type": "spending_budget", "category": "leisure", "time_horizon": "weekly", "target_amount": 500, "rationale": "Update existing goal: Weekly Leisure Budget to $500."}}
```

</ANALYZE_TOOL_CALLS>
""",
    "output": '{"score": 3, "notes": "target_amount $500 is implausible vs weekly leisure trail band roughly $65–$82."}',
  },
]


class ReviewNextGoalStepsAccuracyCheckerOptimizer:
  def __init__(
    self,
    model_name: str = "gemini-flash-lite-latest",
    *,
    max_output_tokens: int = 512,
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
  parser.add_argument("--max-output-tokens", type=int, default=512)
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

  opt = ReviewNextGoalStepsAccuracyCheckerOptimizer(
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
