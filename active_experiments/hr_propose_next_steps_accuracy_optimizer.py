"""
Propose-next-steps rubric optimizer — **accuracy only**.

Grades only the **accuracy** axis.

Run from `finance-ai-penny` repo root:

  python3 active_experiments/hr_propose_next_steps_accuracy_optimizer.py --test all
  python3 active_experiments/hr_propose_next_steps_accuracy_optimizer.py --test good_aligned
  python3 active_experiments/hr_propose_next_steps_accuracy_optimizer.py --model gemini-flash-lite-latest
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

TEST_CASES: list[dict[str, Any]] = [
  {
    "name": "good_aligned",
    "batch": 1,
    "input": """<CONTEXTS>

<CONTEXT index="1">

<RATIONALIZE>

# Rationalize What

Explain: Groceries is significantly down this month at $120. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

- Groceries: $120 this month (Apr 1–30, 2026).

## Drivers

Several Costco transactions appear miscategorized as Miscellaneous.

## Next steps

Create a categorization rule so Costco is always groceries.

</RATIONALIZE>

</CONTEXT>

</CONTEXTS>

<PROPOSAL>

# Proposal

## Proposed next steps

1. **Add a categorization rule** so merchant name matches “Costco” map to Groceries (`ai_category_id` for groceries), applied to past and future.

## Open items (not addressed)

1. **Review** whether any non-Costco groceries are still landing in Miscellaneous.

</PROPOSAL>

<CALLS>

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

</CALLS>
""",
    "output": '{"score": 5, "notes": "Proposal is grounded in the rationalize context and the tool call matches the text."}',
  },
  {
    "name": "bad_hallucinated_merchant",
    "batch": 1,
    "input": """<CONTEXTS>

<CONTEXT index="1">

<RATIONALIZE>

# Rationalize What

Explain: Groceries is significantly down this month at $120. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

- Groceries: $120 this month (Apr 1–30, 2026).

## Drivers

Several Costco transactions appear miscategorized as Miscellaneous.

## Next steps

Create a categorization rule so Costco is always groceries.

</RATIONALIZE>

</CONTEXT>

</CONTEXTS>

<PROPOSAL>

# Proposal

## Proposed next steps

1. **Recategorize** the $500 Whole Foods charge on 2026-04-02 from Dining to Groceries.

## Open items (not addressed)

1. None.

</PROPOSAL>

<CALLS>

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

</CALLS>
""",
    "output": '{"score": 1, "notes": "Invents a Whole Foods charge and tool usage is inconsistent with the rationalize context."}',
  },
  {
    "name": "real_service_fees_goal_two_rounds",
    "batch": 1,
    "input": """<CONTEXTS>

<CONTEXT index="1">

<RATIONALIZE>

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

</RATIONALIZE>

</CONTEXT>

</CONTEXTS>

<PROPOSAL>

# Proposal

## Proposed next steps

1. **Create** a monthly spending budget of $250 for Service Fees to support the current downward trend in credit card interest charges.

## Open items (not addressed)

1. **Review** credit card account statements and APR details to determine if the interest reduction resulted from a lower average daily balance or a rate change.
2. **Explore** debt repayment strategies or balance transfer options to eliminate remaining recurring interest fees.

</PROPOSAL>

<CALLS>

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

# Round 2
_No tool calls this round._

</CALLS>
""",
    "output": '{"score": 5, "notes": "Proposal and tool call are grounded in the rationalize context and are actionable for Penny."}',
  },
  {
    "name": "multi_two_contexts_synthesized",
    "batch": 1,
    "input": """<CONTEXTS>

<CONTEXT index="1">

<RATIONALIZE>

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

</RATIONALIZE>

</CONTEXT>

<CONTEXT index="2">

<RATIONALIZE>

# Rationalize What

Explain: Kids education spending this month. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

- Kids education: $180 this month.

## Next steps

Review recurring tutoring charges for consistency.

</RATIONALIZE>

</CONTEXT>

</CONTEXTS>

<PROPOSAL>

# Proposal

## Proposed next steps

1. **Create** a monthly spending budget of $250 for Service Fees (credit card interest) to sustain the downward trend.
2. **Review** recurring tutoring charges under Kids education for consistency and miscategorization.

## Open items (not addressed)

1. **Confirm** whether tutoring amounts align with expected subscription cadence after categorization cleanup.

</PROPOSAL>

<CALLS>

# Round 1
## Invoked tools

1. **`propose_create_goal`**

```json
{
  "category": "bills_service_fees",
  "goal_type": "spending_budget",
  "goal_title": "Limit Service Fees",
  "target_amount": 250,
  "time_horizon": "monthly",
  "rationale": "Align with reduced interest narrative."
}
```

</CALLS>
""",
    "output": '{"score": 5, "notes": "Grounded in both contexts; propose_create_goal supports the service-fees thread."}',
  },
  {
    "name": "multi_drops_second_context",
    "batch": 1,
    "input": """<CONTEXTS>

<CONTEXT index="1">

<RATIONALIZE>

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

</RATIONALIZE>

</CONTEXT>

<CONTEXT index="2">

<RATIONALIZE>

# Rationalize What

Explain: Kids education spending this month. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

- Kids education: $180 this month.

## Next steps

Review recurring tutoring charges for consistency.

</RATIONALIZE>

</CONTEXT>

</CONTEXTS>

<PROPOSAL>

# Proposal

## Proposed next steps

1. **Create** a monthly spending budget of $250 for Service Fees.

## Open items (not addressed)

1. None.

</PROPOSAL>

<CALLS>

# Round 1
## Invoked tools

1. **`propose_create_goal`**

```json
{
  "category": "bills_service_fees",
  "goal_type": "spending_budget",
  "goal_title": "Limit Service Fees",
  "target_amount": 250,
  "time_horizon": "monthly",
  "rationale": "Align with reduced interest narrative."
}
```

</CALLS>
""",
    "output": '{"score": 4, "notes": "Remaining content is grounded but incomplete versus dual-context evidence."}',
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


SYSTEM_PROMPT = """Grade **accuracy** only.

You are a **strict rubric grader** for the checker bundle below (XML-style role wrappers; do not require table/column names).

You receive **evidence**, then **`<PROPOSAL>`**, then **`<CALLS>`**.

**Evidence:**

- **`<CONTEXTS>` … `</CONTEXTS>`**: one or more **`<CONTEXT index="N">`** blocks. Each **`<CONTEXT>`** contains **`<RATIONALIZE>` … `</RATIONALIZE>`** (figures, drivers, next steps). Optionally **`<RATIONALIZE_CALLS>` … `</RATIONALIZE_CALLS>`** holds **that rationalize run’s** stored LLM trace — **not** the propose run. Alternatively, the bundle may use a single top-level **`<RATIONALIZE>` … `</RATIONALIZE>`** with no **`<CONTEXTS>`** wrapper (same markdown as inside a context).

For **grounding**, treat **all** **`<RATIONALIZE>`** bodies as the **combined** evidence; the proposal should synthesize across contexts.

**Then:**

1. **`<PROPOSAL>` … `</PROPOSAL>`** — Markdown from the **propose** outcome (`agent_outcome`): normally **only** the **`# Proposal`** block with **`## Proposed next steps`** and **`## Open items (not addressed)`** (or equivalent **`##`** headings if `# Proposal` was omitted).

2. **`<CALLS>` … `</CALLS>`** — Markdown listing **LLM round-trips** from **this same propose / propose-multi run** (`calls` field): `# Round N`, optional metrics, **`## Invoked tools`** with numbered tools and fenced argument blocks. **`latency_ms` / `input_tokens` / `output_tokens`** appear only when present in source data.

**Do not** confuse **`<RATIONALIZE_CALLS>`** inside a **`<CONTEXT>`** with **`<CALLS>`** at the end. Grade tool consistency for the propose run using **`<CALLS>`** only.

If **`<CALLS>`** says there are **no rounds**, or **no tools** were invoked in any round, score the **tool-trace** aspect of **`accuracy` = 3** with `notes` stating no usable tool trace (do not infer tool order from proposal text alone). You may still score grounding of proposal text vs **all** rationalize evidence when clear.

Grade **only** what is in the message. Do not invent missing data.

**Axis (return ONLY `{score, notes}`):**

**Accuracy** — Is **`<PROPOSAL>`** both **grounded** and **helpful for Penny to execute**?
  - Grounding: No invented merchants/amounts/dates; no contradictions vs **any** **`<RATIONALIZE>`** body.
  - Actionability for Penny: Proposed steps should align with Penny’s capabilities (create goals/budgets, create categorization rules, recategorize transactions) and avoid recommending actions irrelevant to the rationalize context(s).
  - Tool consistency: When **`<CALLS>`** documents invocations, they must support the proposal (correct tools, sensible args; **retrieve** before **propose_recategorize_transactions** when ids are not in rationalize text). Apply the **score 3** rule above when no usable tool trace exists.

**Calibration:** **5** = no meaningful gap on that axis. **4** = one minor gap. **3** = clear but fixable issue. **2** = several problems. **1** = axis largely failed.

Return **only** the JSON object matching the schema (`score`, `notes`).
"""


class ProposeNextStepsAccuracyCheckerOptimizer:
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
  args = parser.parse_args()

  if args.test is None and args.batch is None:
    print("Available test cases:")
    for i, tc in enumerate(TEST_CASES):
      batch = tc.get("batch")
      batch_s = str(batch) if isinstance(batch, int) else "—"
      print(f"  {i}: {tc.get('name')} (batch {batch_s})")
    return

  opt = ProposeNextStepsAccuracyCheckerOptimizer(
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
    if tc.get("output") is not None:
      _print_section_banner("# Expected Output")
      print(tc["output"])

  print(f"\n{_TEST_SEPARATOR}\n")
  print(f"# Total tests: {len(cases)}\n")


if __name__ == "__main__":
  main()

