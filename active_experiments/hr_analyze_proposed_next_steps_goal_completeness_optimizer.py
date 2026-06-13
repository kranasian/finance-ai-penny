"""
Analyze-proposed-next-steps-goal rubric optimizer — **completeness** only.

Grades only the **completeness** axis for ``Hr:AnalyzeProposedNextStepsGoalCompleteness`` checker templates.

Run from ``finance-ai-penny`` repo root:

  python3 active_experiments/hr_analyze_proposed_next_steps_goal_completeness_optimizer.py --test all
  python3 active_experiments/hr_analyze_proposed_next_steps_goal_completeness_optimizer.py --test all --check
  python3 active_experiments/hr_analyze_proposed_next_steps_goal_completeness_optimizer.py --batch 1 --check
  python3 active_experiments/hr_analyze_proposed_next_steps_goal_completeness_optimizer.py --model gemini-flash-lite-latest
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


SYSTEM_PROMPT = """You are a strict **completeness-only** grader for ``analyze_proposed_next_steps_goal`` outcomes. Return JSON `{score, notes}` (integer 1–5; one sentence `notes`).

**Bundle:** `<PROPOSE_SOURCE>` (optional prior propose markdown) … `<ACTIVE_GOALS>` … `<PROPOSED_GOAL_STEPS>` … `<ANALYZE_PROPOSAL>` … `<ANALYZE_TOOL_CALLS>` (this analyze run only: `# Round N`, then `## Invoked tools` per round). Grade only visible text; do not invent facts. **Ignore factual routing correctness and whether dollar amounts are ideal** (accuracy).

**Completeness (workflow + coverage):**
1. **Disposition coverage** — Every create/update goal bullet in `<PROPOSED_GOAL_STEPS>` must appear in **either** `## Finalized proposed goal steps` **or** `## Already covered by existing goals` (paraphrase OK). No silent drops. **Create→update reroute is valid disposition:** when a parent active goal covers a **Create** candidate's leaf topic and amounts differ, the kept bullet may read `Update existing goal: <goal_title> to $…` instead of `Create new goal:` — that still satisfies coverage (do **not** score 1 solely because the verb changed from create to update).
2. **Investigation (required for kept steps)** — For **each kept** candidate (a bullet under `## Finalized proposed goal steps`), `<ANALYZE_TOOL_CALLS>` must show **≥6** `lookup_user_aggregate_spending` calls for that candidate's slug **before** its `propose_create_goal` / `propose_update_goal` call(s), at the candidate's grain (weekly/monthly). Widen to 9–12 is acceptable. **Dropped / already-covered** candidates do **not** require lookups or propose tools. On create→update reroute, lookups use the **matched active row's** parent ``category`` slug (e.g. ``leisure``), not the candidate's leaf slug.
3. **Tool ↔ Finalized coherence** — `len(## Finalized proposed goal steps bullets) == len(propose_create_goal) + len(propose_update_goal)`. Zero kept → Finalized is only `1. - (none)`.
4. **Structure** — Top-level `# Analysis output`; sections `## Finalized proposed goal steps` and `## Already covered by existing goals` only; **no** `## Open items`. Finalized bullets should match turn-1 propose-tool `rationale` text.
5. **Tool family** — Creates → `propose_create_goal`; updates → `propose_update_goal`.

**Out of scope for completeness (do not penalize here):**
- Whether create vs update routing is factually correct vs `<ACTIVE_GOALS>` (accuracy).
- Whether `target_amount` matches spending trail (accuracy).
- Category parent↔leaf scope on updates (accuracy).

**Scoring anchors (worst match wins):**
- **1** — Kept propose without any prior lookup for that candidate; propose tools with no matching Proposed bullets; or a primary candidate wholly unaddressed.
- **2** — Multiple protocol breaks (e.g. skipped investigation **and** coherence break).
- **3** — One clear gap: fewer than 6 lookups before a kept propose; missing Already-covered entry for a dropped duplicate; wrong propose-tool count; wrong tool family (create vs update).
- **4** — Minor shape issue (thin Already covered wording) while workflow otherwise holds.
- **5** — Full disposition; ≥6 lookups before each kept propose; coherence + structure correct.

**`notes`:** One sentence; name the gap or affirm disposition, investigation, and coherence.
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


_SIX_WEEKLY_LEISURE_LOOKUPS = """
1. **`lookup_user_aggregate_spending`**
```json
{"granularity": "weekly", "category": ["leisure"], "date_in_range": "2026-04-06"}
```
2. **`lookup_user_aggregate_spending`**
```json
{"granularity": "weekly", "category": ["leisure"], "date_in_range": "2026-04-13"}
```
3. **`lookup_user_aggregate_spending`**
```json
{"granularity": "weekly", "category": ["leisure"], "date_in_range": "2026-04-20"}
```
4. **`lookup_user_aggregate_spending`**
```json
{"granularity": "weekly", "category": ["leisure"], "date_in_range": "2026-04-27"}
```
5. **`lookup_user_aggregate_spending`**
```json
{"granularity": "weekly", "category": ["leisure"], "date_in_range": "2026-05-04"}
```
6. **`lookup_user_aggregate_spending`**
```json
{"granularity": "weekly", "category": ["leisure"], "date_in_range": "2026-05-11"}
```
""".strip()


TEST_CASES: list[dict[str, Any]] = [
  {
    "name": "complete_update_with_six_lookups",
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

{_SIX_WEEKLY_LEISURE_LOOKUPS}

# Round 2

## Invoked tools

1. **`propose_update_goal`**
```json
{{"existing_goal_id": 2751, "goal_title": "Weekly Leisure Budget", "goal_type": "spending_budget", "category": "leisure", "time_horizon": "weekly", "target_amount": 50, "rationale": "Update existing goal: Weekly Leisure Budget to $50."}}
```

</ANALYZE_TOOL_CALLS>
""",
    "output": '{"score": 5, "notes": "Candidate kept with six weekly leisure lookups before propose_update_goal; coherence and sections are correct."}',
  },
  {
    "name": "create_rerouted_to_parent_update",
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

{_SIX_WEEKLY_LEISURE_LOOKUPS}

# Round 2

## Invoked tools

1. **`propose_update_goal`**
```json
{{"existing_goal_id": 2751, "goal_title": "Weekly Leisure Budget 🎯", "goal_type": "spending_budget", "category": "leisure", "time_horizon": "weekly", "target_amount": 50, "rationale": "Update existing goal: Weekly Leisure Budget 🎯 to $50 to better accommodate irregular entertainment and ticket expenses."}}
```

</ANALYZE_TOOL_CALLS>
""",
    "output": '{"score": 5, "notes": "Create candidate rerouted to parent-row update with six leisure lookups, matching rationale, and coherence."}',
  },
  {
    "name": "insufficient_lookups_before_kept_propose",
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

1. **`lookup_user_aggregate_spending`**
```json
{{"granularity": "weekly", "category": ["leisure"], "date_in_range": "2026-04-06"}}
```
2. **`lookup_user_aggregate_spending`**
```json
{{"granularity": "weekly", "category": ["leisure"], "date_in_range": "2026-04-13"}}
```
3. **`lookup_user_aggregate_spending`**
```json
{{"granularity": "weekly", "category": ["leisure"], "date_in_range": "2026-04-20"}}
```
4. **`lookup_user_aggregate_spending`**
```json
{{"granularity": "weekly", "category": ["leisure"], "date_in_range": "2026-04-27"}}
```
5. **`lookup_user_aggregate_spending`**
```json
{{"granularity": "weekly", "category": ["leisure"], "date_in_range": "2026-05-04"}}
```

# Round 2

## Invoked tools

1. **`propose_update_goal`**
```json
{{"existing_goal_id": 2751, "goal_title": "Weekly Leisure Budget", "goal_type": "spending_budget", "category": "leisure", "time_horizon": "weekly", "target_amount": 50, "rationale": "Update existing goal: Weekly Leisure Budget to $50."}}
```

</ANALYZE_TOOL_CALLS>
""",
    "output": '{"score": 3, "notes": "Kept update proposed after only five weekly leisure lookups, not the required six."}',
  },
  {
    "name": "silent_drop_create_candidate",
    "batch": 1,
    "input": """<ACTIVE_GOALS>

1. goal_id=2751 | goal_title="Weekly Leisure Budget 🎯" | type=category | granularity=weekly | category=leisure | target_amount=$25

</ACTIVE_GOALS>

<PROPOSED_GOAL_STEPS>

1. Create new goal: Establish a weekly Entertainment goal of $50 for leisure_entertainment.

</PROPOSED_GOAL_STEPS>

<ANALYZE_PROPOSAL>

# Analysis output

## Finalized proposed goal steps

1. - (none)

## Already covered by existing goals

1. - (none)

</ANALYZE_PROPOSAL>

<ANALYZE_TOOL_CALLS>

# Round 1

## Invoked tools

1. (none)

</ANALYZE_TOOL_CALLS>
""",
    "output": '{"score": 1, "notes": "Create candidate wholly unaddressed in both Finalized and Already covered."}',
  },
  {
    "name": "skipped_investigation_before_propose",
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

1. Update existing goal: Weekly Leisure Budget to $50.

## Already covered by existing goals

1. - (none)

</ANALYZE_PROPOSAL>

<ANALYZE_TOOL_CALLS>

# Round 1

## Invoked tools

1. **`propose_update_goal`**
```json
{"existing_goal_id": 2751, "goal_title": "Weekly Leisure Budget", "goal_type": "spending_budget", "category": "leisure", "time_horizon": "weekly", "target_amount": 50, "rationale": "Update existing goal: Weekly Leisure Budget to $50."}
```

</ANALYZE_TOOL_CALLS>
""",
    "output": '{"score": 1, "notes": "Kept update proposed without any lookup_user_aggregate_spending investigation."}',
  },
  {
    "name": "coherence_mismatch_two_bullets_one_tool",
    "batch": 1,
    "input": f"""<ACTIVE_GOALS>

1. goal_id=2751 | goal_title="Weekly Leisure Budget" | type=category | granularity=weekly | category=leisure | target_amount=$75
2. goal_id=2750 | goal_title="Monthly Health Budget" | type=category | granularity=monthly | category=health | target_amount=$235

</ACTIVE_GOALS>

<PROPOSED_GOAL_STEPS>

1. Update existing goal: Weekly Leisure Budget to $50.
2. Update existing goal: Monthly Health Budget to $210.

</PROPOSED_GOAL_STEPS>

<ANALYZE_PROPOSAL>

# Analysis output

## Finalized proposed goal steps

1. Update existing goal: Weekly Leisure Budget to $50.
2. Update existing goal: Monthly Health Budget to $210.

## Already covered by existing goals

1. - (none)

</ANALYZE_PROPOSAL>

<ANALYZE_TOOL_CALLS>

# Round 1

## Invoked tools

{_SIX_WEEKLY_LEISURE_LOOKUPS}

7. **`propose_update_goal`**
```json
{{"existing_goal_id": 2751, "goal_title": "Weekly Leisure Budget", "goal_type": "spending_budget", "category": "leisure", "time_horizon": "weekly", "target_amount": 50, "rationale": "Update existing goal: Weekly Leisure Budget to $50."}}
```

</ANALYZE_TOOL_CALLS>
""",
    "output": '{"score": 3, "notes": "Two Proposed bullets but only one propose_update_goal call."}',
  },
  {
    "name": "duplicate_dropped_to_already_covered",
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
    "output": '{"score": 5, "notes": "Duplicate dropped to Already covered with no propose tools; no investigation required for a dropped candidate."}',
  },
  {
    "name": "missing_already_covered_entry_for_duplicate",
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

1. - (none)

</ANALYZE_PROPOSAL>

<ANALYZE_TOOL_CALLS>

# Round 1

## Invoked tools

1. (none)

</ANALYZE_TOOL_CALLS>
""",
    "output": '{"score": 3, "notes": "Duplicate candidate dropped silently without an Already covered line."}',
  },
  {
    "name": "wrong_tool_family_create_on_update",
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

{_SIX_WEEKLY_LEISURE_LOOKUPS}

7. **`propose_create_goal`**
```json
{{"existing_goal_id": 2751, "goal_title": "Weekly Leisure Budget", "goal_type": "spending_budget", "category": "leisure", "time_horizon": "weekly", "target_amount": 50, "rationale": "Update existing goal: Weekly Leisure Budget to $50."}}
```

</ANALYZE_TOOL_CALLS>
""",
    "output": '{"score": 3, "notes": "Update candidate used propose_create_goal instead of propose_update_goal."}',
  },
  {
    "name": "mixed_reroute_and_duplicate_drop",
    "batch": 2,
    "input": f"""<ACTIVE_GOALS>

1. goal_id=2751 | goal_title="Weekly Leisure Budget 🎯" | type=category | granularity=weekly | category=leisure | target_amount=$25
2. goal_id=22 | goal_title="Clothing" | type=category | granularity=monthly | category=shopping_clothing | target_amount=$60

</ACTIVE_GOALS>

<PROPOSED_GOAL_STEPS>

1. Create new goal: Establish a weekly Entertainment goal of $50 for leisure_entertainment.
2. Create new goal: $60 monthly Clothing budget.

</PROPOSED_GOAL_STEPS>

<ANALYZE_PROPOSAL>

# Analysis output

## Finalized proposed goal steps

1. Update existing goal: Weekly Leisure Budget 🎯 to $50 to better accommodate irregular entertainment and ticket expenses.

## Already covered by existing goals

1. Create new goal: $60 monthly Clothing budget. — Already covered by existing Clothing monthly budget goal.

</ANALYZE_PROPOSAL>

<ANALYZE_TOOL_CALLS>

# Round 1

## Invoked tools

{_SIX_WEEKLY_LEISURE_LOOKUPS}

# Round 2

## Invoked tools

1. **`propose_update_goal`**
```json
{{"existing_goal_id": 2751, "goal_title": "Weekly Leisure Budget 🎯", "goal_type": "spending_budget", "category": "leisure", "time_horizon": "weekly", "target_amount": 50, "rationale": "Update existing goal: Weekly Leisure Budget 🎯 to $50 to better accommodate irregular entertainment and ticket expenses."}}
```

</ANALYZE_TOOL_CALLS>
""",
    "output": '{"score": 5, "notes": "Both candidates dispositioned: create rerouted to parent update with six lookups; duplicate dropped to Already covered without tools."}',
  },
]


class AnalyzeProposedNextStepsGoalCompletenessCheckerOptimizer:
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

  opt = AnalyzeProposedNextStepsGoalCompletenessCheckerOptimizer(
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
