"""
Propose-next-steps rubric optimizer — **completeness only**.

Grades only the **completeness** axis.

Run from `finance-ai-penny` repo root:

  python3 active_experiments/hr_propose_next_steps_completeness_optimizer.py --test all
  python3 active_experiments/hr_propose_next_steps_completeness_optimizer.py --test all --check
  python3 active_experiments/hr_propose_next_steps_completeness_optimizer.py --batch 1
  python3 active_experiments/hr_propose_next_steps_completeness_optimizer.py --batch 1 --check
  # Batches 1–4 partition fixtures (see each test case's "batch" field).
  python3 active_experiments/hr_propose_next_steps_completeness_optimizer.py --model gemini-flash-lite-latest
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


def _print_section_banner(title: str) -> None:
  print(f"\n{_SECTION_RULE}\n{title}\n{_SECTION_RULE}\n")


def _parse_expected_output(raw: str | None) -> dict[str, Any] | None:
  if not raw:
    return None
  try:
    return json.loads(raw)
  except Exception:
    return None


TEST_CASES: list[dict[str, Any]] = [
  {
    "name": "good_costco_rule_automated",
    "batch": 1,
    "input": """<RATIONALIZE_1>

# Rationalize What

Explain: Groceries is significantly down this month at $120. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

- Groceries: $120 this month (Apr 1–30, 2026).

## Drivers

Several Costco transactions appear miscategorized as Miscellaneous.

## Next steps

Create a categorization rule so Costco is always groceries.

</RATIONALIZE_1>

<PROPOSAL>

# Proposal

## Proposed next steps

1. **Add a categorization rule** so merchant name matches “Costco” map to Groceries (`ai_category_id` for groceries), applied to past and future.

## Open items (not addressed)

1. **Review** whether any non-Costco groceries are still landing in Miscellaneous.

</PROPOSAL>

<PROPOSAL_TOOL_CALLS>

# Round 1

## Invoked tools

1. **`propose_create_categorization_rule`**

```json
{
  "rule": {
    "name_sub_eq": "costco"
  },
  "ai_category_id": 4,
  "scope": "future_and_past",
  "rationale": "Costco should count as groceries."
}
```

</PROPOSAL_TOOL_CALLS>
""",
    "output": '{"score": 5, "notes": "Covers the rationalize next step with a concrete proposal, matching tool automation, and a sensible open item."}',
  },
  {
    "name": "bad_unrelated_merchant_recategorize",
    "batch": 1,
    "input": """<RATIONALIZE_1>

# Rationalize What

Explain: Groceries is significantly down this month at $120. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

- Groceries: $120 this month (Apr 1–30, 2026).

## Drivers

Several Costco transactions appear miscategorized as Miscellaneous.

## Next steps

Create a categorization rule so Costco is always groceries.

</RATIONALIZE_1>

<PROPOSAL>

# Proposal

## Proposed next steps

1. **Recategorize** the $500 Whole Foods charge on 2026-04-02 from Dining to Groceries.

## Open items (not addressed)

1. None.

</PROPOSAL>

<PROPOSAL_TOOL_CALLS>

# Round 1

## Invoked tools

1. **`propose_recategorize_transactions`**

```json
{
  "transaction_ids": [
    99999
  ],
  "target_category": "groceries",
  "rationale": "fix"
}
```

</PROPOSAL_TOOL_CALLS>
""",
    "output": '{"score": 1, "notes": "Does not address the Costco rule next step and substitutes unrelated recategorization work."}',
  },
  {
    "name": "uncategorized_ambiguous_user_open_items_five",
    "batch": 2,
    "input": """<RATIONALIZE_1>

# Rationalize What

Explain: Uncategorized “general merchandise / mixed basket” spend this month. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

- $210 across a handful of charges still uncategorized after Penny review.

## Drivers

Merchant strings are vague or describe many goods/services; Penny cannot assign a single category without guessing.

## Next steps

1. Leave those uncategorized charges for the user to categorize manually where Penny cannot infer one category.
2. Add a narrow merchant rule only for merchants with an unambiguous category (e.g. “PUREGYM MONTHLY” → Health & Fitness).

</RATIONALIZE_1>

<PROPOSAL>

# Proposal

## Proposed next steps

1. **Create** a categorization rule mapping merchant substring “PUREGYM MONTHLY” to Health & Fitness for past and future.

## Open items (not addressed)

1. **User** manually categorizes the remaining uncategorized “general merchandise / mixed basket” charges Penny flagged as too ambiguous for automatic categorization.

</PROPOSAL>

<PROPOSAL_TOOL_CALLS>

# Round 1

## Invoked tools

1. **`propose_create_categorization_rule`**

```json
{
  "rule": {
    "name_sub_eq": "puregym monthly"
  },
  "ai_category_id": 12,
  "scope": "future_and_past",
  "rationale": "Unambiguous gym subscription string."
}
```

</PROPOSAL_TOOL_CALLS>
""",
    "output": '{"score": 5, "notes": "Ambiguous uncategorized transactions correctly parked for user categorization under Open items; unambiguous rule automated with matching calls."}',
  },
  {
    "name": "verify_categorizations_in_proposed_only_four",
    "batch": 2,
    "input": """<RATIONALIZE_1>

# Rationalize What

Explain: Electronics and streaming subscription spend this month. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

- Electronics: $340; streaming subscriptions: $48.

## Drivers

Several large electronics charges were auto-tagged; streaming merchants look consistent.

## Next steps

1. Verify that recent electronics and streaming subscription transactions are categorized correctly before locking month-end views.

</RATIONALIZE_1>

<PROPOSAL>

# Proposal

## Proposed next steps

1. **Have Penny** systematically verify that all recent electronics and streaming subscription line items are categorized correctly and adjust any mislabels found.

## Open items (not addressed)

1. None.

</PROPOSAL>

<PROPOSAL_TOOL_CALLS>

# Round 1

## Invoked tools

_No tool calls this round._

</PROPOSAL_TOOL_CALLS>
""",
    "output": '{"score": 4, "notes": "Verification is not a Penny capability and belongs under Open items for the user; placing it only under Proposed is a minor structure gap."}',
  },
  {
    "name": "historical_metric_in_open_not_proposed_four",
    "batch": 3,
    "input": """<RATIONALIZE_1>

# Rationalize What

Explain: Coffee spend creeping up. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

- Coffee: $95 this month vs ~$60 typical in prior months.

## Drivers

More café visits mid-month.

## Next steps

1. From the last 12 months of historical transactions, compute median monthly coffee spend to size a realistic reduction target.
2. Propose a monthly coffee spending goal informed by that median (not above it).

</RATIONALIZE_1>

<PROPOSAL>

# Proposal

## Proposed next steps

1. **Create** a monthly Coffee spending goal of $120 without recomputing history.

## Open items (not addressed)

1. **Compute** median monthly coffee spend from the trailing 12 months of historical transactions so the cap is evidence-based.

</PROPOSAL>

<PROPOSAL_TOOL_CALLS>

# Round 1

## Invoked tools

1. **`propose_create_goal`**

```json
{
  "category": "dining_coffee",
  "goal_type": "spending_budget",
  "goal_title": "Cap Coffee",
  "target_amount": 120,
  "time_horizon": "monthly",
  "rationale": "Reduce café drift."
}
```

</PROPOSAL_TOOL_CALLS>
""",
    "output": '{"score": 5, "notes": "Historical median computation correctly parked under Open as non-Penny work; the evidence-based goal is proposed with a matching tool call."}',
  },
  {
    "name": "incomplete_dropped_rationalize_steps_two",
    "batch": 3,
    "input": """<RATIONALIZE_1>

# Rationalize What

Explain: Household bills and subscriptions snapshot. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

- Utilities up slightly; three streaming services renewed same week.

## Next steps

1. Create a categorization rule so “CITYPOWER UTIL” always maps to Utilities.
2. Review whether any streaming charges should be split or merged across months.
3. Set a monthly combined cap goal for streaming subscriptions for the next quarter.

</RATIONALIZE_1>

<PROPOSAL>

# Proposal

## Proposed next steps

1. **Add** a categorization rule for merchant substring “CITYPOWER UTIL” → Utilities, past and future.

## Open items (not addressed)

1. None.

</PROPOSAL>

<PROPOSAL_TOOL_CALLS>

# Round 1

## Invoked tools

1. **`propose_create_categorization_rule`**

```json
{
  "rule": {
    "name_sub_eq": "citypower util"
  },
  "ai_category_id": 33,
  "scope": "future_and_past",
  "rationale": "Stabilize utilities tagging."
}
```

</PROPOSAL_TOOL_CALLS>
""",
    "output": '{"score": 2, "notes": "Delivers only the utilities rule while omitting streaming review (Open) and the streaming cap goal (Proposed); Open items falsely list none despite two remaining steps."}',
  },
  {
    "name": "service_fees_budget_and_reviews_open_five",
    "batch": 4,
    "input": """<RATIONALIZE_1>

# Rationalize What

Explain: Service Fees are significantly down this month. (credit card interest charges) (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

- Service Fees / interest charges: down vs last month.

## Drivers

Credit card interest charges decreased.

## Next steps

1. Consider setting a budget for service fees to keep interest charges low.
2. Review APR / statement details to confirm why interest changed.

</RATIONALIZE_1>

<PROPOSAL>

# Proposal

## Proposed next steps

1. **Create** a monthly spending budget of $250 for Service Fees to support the current downward trend in credit card interest charges.

## Open items (not addressed)

1. **Review** credit card account statements and APR details to determine if the interest reduction resulted from a lower average daily balance or a rate change.
2. **Explore** debt repayment strategies or balance transfer options to eliminate remaining recurring interest fees.

</PROPOSAL>

<PROPOSAL_TOOL_CALLS>

# Round 1

## Invoked tools

1. **`propose_create_goal`**

```json
{
  "category": "bills_service_fees",
  "goal_type": "spending_budget",
  "rationale": "Based on the recent reduction in credit card interest charges, a $250 monthly budget for service fees will help sustain this positive trend and encourage continued debt management.",
  "time_horizon": "monthly",
  "goal_title": "Limit Service Fees",
  "target_amount": 250
}
```

</PROPOSAL_TOOL_CALLS>
""",
    "output": '{"score": 5, "notes": "Creates a concrete budget with matching goal tool call and parks human APR and debt follow-ups under Open items."}',
  },
]


def _build_output_schema(_types: Any) -> Any:
  return _types.Schema(
    type=_types.Type.OBJECT,
    required=["score", "notes"],
    properties={
      "score": _types.Schema(type=_types.Type.INTEGER, description="Integer 1–5."),
      "notes": _types.Schema(type=_types.Type.STRING, description="One short sentence."),
    },
  )


SYSTEM_PROMPT = """You are a strict completeness-only grader. Return JSON `{score, notes}` (integer 1–5; one sentence `notes`).

**Bundle:** `<RATIONALIZE_N>` … `<PROPOSAL>` … `<PROPOSAL_TOOL_CALLS>` (this propose run only: `# Round N`, then `## Invoked tools` per round). Grade only visible text; do not invent facts. Ignore factual accuracy, goal dollar amounts, and whether tool choice is ideal.

**Penny capabilities (only these belong under Proposed next steps with matching tools):**
- **Track budgets** — spending caps via `propose_create_goal` (`spending_budget`).
- **Track saving goals** — savings targets via `propose_create_goal` (saving-goal types).
- **Re-categorize transactions** — `propose_recategorize_transactions`.
- **Set categorization rules** — `propose_create_categorization_rule`.

**Completeness:**
1. **Coverage:** Every `## Next steps` line appears in Proposed or Open (paraphrase OK); combine all `<RATIONALIZE_N>` bodies; no silent drops.
2. **Automation:** If **Proposed next steps** assign Penny one of the **four capabilities** above, `<PROPOSAL_TOOL_CALLS>` must include matching tool invocations **across rounds** (scan every `# Round N` block). Steps that are **not** Penny capabilities do **not** need tools and belong under **Open items**.
3. **Structure:** Only the four Penny capabilities belong in **Proposed**; everything else in **Open**.

**Historical statistic vs goal:** Rationalize asks for (A) a **historical statistic** and (B) a **goal informed by** it. (A) is **not** a Penny capability—park it under **Open items** (complete and correct). (B) under **Proposed** with matching goal tools is expected even when the goal amount is provisional or not yet tied to (A). **Do not** score down because the goal omits recomputing history or uses a placeholder amount. If (A) is **missing** from PROPOSAL entirely, coverage is weaker (**often 3**).

**Sectioning vs coverage:** A rationalize step that appears **only** under **Proposed** but describes **non-Penny** work is **covered** but **mis-sectioned** → **4**, not **3**, when no other bullets are missing. Do **not** treat non-Penny steps parked under Proposed as “missing automation”, “missing tools”, or partial coverage. Example: “verify categorizations are correct” only under Proposed with Open **none** → **4** (belongs in Open for the user), not **3**.

**Scoring anchors (worst match wins):**
- **1** — Primary rationalize next step **wholly unaddressed** while PROPOSAL pursues **different** transaction subjects (e.g. rationalize names a **Costco rule** but PROPOSAL only **recategorizes Whole Foods**), **or** primary step replaced by unrelated work—**use 1, not 2**, even if recategorize tools exist.
- **2** — Whole context thread dropped, **or** **≥2** `## Next steps` bullets missing from both Proposed and Open while Open falsely says **none** (e.g. rationalize lists utilities rule + streaming review + streaming cap goal but PROPOSAL only delivers the rule and Open says none)—**use 2, not 3**.
- **3** — Proposed Penny-capability work without matching `<PROPOSAL_TOOL_CALLS>`, **or** a **primary** Penny-capability ask only under Open with **no** Proposed mirror, **or** exactly **one** rationalize bullet missing from both sections (not the ≥2 + Open-none case).
- **4** — Single sectioning issue, threads otherwise present: **(a)** non-Penny work (verification, retrieval, historical stats, etc.) framed as **only** Penny work under **Proposed** when it belongs under **Open**; **(b)** thin wording only.
- **5** — Full coverage; tools match Proposed Penny-capability work; non-capability steps correctly in **Open**, including **user** categorization for ambiguous uncategorized charges.

**`notes`:** One sentence (semicolons allowed); name gaps or affirm coverage, tools, and split. For multi-`<RATIONALIZE_N>` **5**, name each context theme checked.
"""


class ProposeNextStepsCompletenessCheckerOptimizer:
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

  opt = ProposeNextStepsCompletenessCheckerOptimizer(
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
    _print_section_banner("# LLM Checker Input")
    print(tc["input"])
    result = opt.grade(tc["input"])
    _print_section_banner("# LLM Checker Response")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.check:
      exp = _parse_expected_output(tc.get("output"))
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
      _print_section_banner("# Expected Output")
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

