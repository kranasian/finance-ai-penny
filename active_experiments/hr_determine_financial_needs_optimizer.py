"""
Determine-financial-needs rubric optimizer — **quality** (holistic).

Grades ``Hr:DetermineFinancialNeeds`` checker templates.

Run from ``finance-ai-penny`` repo root:

  python3 active_experiments/hr_determine_financial_needs_optimizer.py --test all
  python3 active_experiments/hr_determine_financial_needs_optimizer.py --test all --check
  python3 active_experiments/hr_determine_financial_needs_optimizer.py --batch 1 --check
  python3 active_experiments/hr_determine_financial_needs_optimizer.py --model gemini-flash-lite-latest
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


def _build_output_schema(_types: Any) -> Any:
  return _types.Schema(
    type=_types.Type.OBJECT,
    required=["score", "notes"],
    properties={
      "score": _types.Schema(type=_types.Type.INTEGER, description="Integer 1–5."),
      "notes": _types.Schema(type=_types.Type.STRING, description="One short sentence."),
    },
  )


SYSTEM_PROMPT = """Grade **determine_financial_needs** outcomes for holistic **quality**. Return JSON `{score, notes}` (integer 1–5; one sentence `notes`).

**Bundle:** `<AGENT_OUTCOME>` … `<NEEDS_TOOL_CALLS>`.

- **`<AGENT_OUTCOME>`** — `# Task Input` (spending/income, depository, credit) plus `# Financial Needs` response.
- **`<NEEDS_TOOL_CALLS>`** — This run's tool trace (`# Round N`, `## Invoked tools`).

Grade **only** visible text. Do not invent missing data.

**Agent job:** Triage **1–3** near-term needs (e.g. **Settle debt**, **Build emergency fund**, **Save money**, **Stabilize cash flow**, **Reduce interest drag**, **Stay on track**) — not a full multi-year strategy.

**Required response shape** under `# Financial Needs`:
- `## Summary`
- `## Primary needs` (1–3 numbered; Title Case labels from taxonomy)
- `## Evidence` (3–6 bullets)
- `## Immediate Things to Do` (2–4 steps, 7–30 days)
- `## Next Set of Milestones to Aspire` (2–4 measurable targets, 1–6 months)

**Tool protocol:**
1. **`lookup_user_account_interest_rates`** and **`lookup_user_account_payment_schedule`** should appear in `<NEEDS_TOOL_CALLS>` before the final answer when credit/liquidity matters.
2. Do **not** require re-fetching monthly **Total Spending** / **Total Income** already in `# Task Input` — citing Task Input is correct.
3. Transaction SQL only when a single merchant/charge explains a need.

**Grounding:** Every **$** in the response must be traceable to `# Task Input` or tool outputs (whole dollars). Low score for invented balances, interest, or spend.

**Triage quality:** Needs must match evidence (high card balance + interest → debt/interest drag; thin liquid vs monthly spend → emergency fund; income ≈ outflows → stabilize cash flow). Wrong primary need or >3 needs → lower score.

**Actionability:** Immediate steps and milestones must be **specific and measurable** ($ targets, months of expenses, paydown amounts) — not vague advice.

**Scoring (worst gap wins):**
- **1** — Invented $; missing `# Financial Needs` structure; no mandatory tools when credit is material; wholly wrong triage.
- **2** — Multiple gaps (skipped interest/payment tools, vague milestones, weak evidence).
- **3** — Single clear gap (one need mis-ranked, one milestone without $, thin Evidence).
- **4** — Minor issue (wording, one slightly vague step) while triage and grounding are sound.
- **5** — Correct triage, grounded Evidence, mandatory tools when warranted, concrete immediate steps and milestones.

Return **only** the JSON object (`score`, `notes`).
"""


TEST_CASES: list[dict[str, Any]] = [
  {
    "name": "good_debt_and_interest_triage",
    "batch": 1,
    "input": """<AGENT_OUTCOME>

# Task Input

## Spending and income (last 3 completed months)

| Period | Total Spending | Total Income |
| --- | --- | --- |
| Mar 01, 2026 – Mar 31, 2026 | $4,200 | $4,100 |
| Feb 01, 2026 – Feb 28, 2026 | $4,050 | $4,100 |
| Jan 01, 2026 – Jan 31, 2026 | $3,980 | $4,050 |

## Credit

| id | name | balance | limit | pay1 | pay2 | pay3 |
| --- | --- | --- | --- | --- | --- | --- |
| 12 | Venture | $8,400 | $20,000 | $180 on 2026-03-15 ($8,200 bal) | $175 on 2026-02-12 ($8,100 bal) | $170 on 2026-01-10 ($7,900 bal) |

# Financial Needs

## Summary
You carry about **$8,400** on Venture with spending near income (~**$4,100**/mo). Interest charges are material.

## Primary needs
1. **Reduce interest drag** — Paid **$312** in card interest over 90 days while liquid cash is thin.
2. **Settle debt** — Venture balance **$8,400** is rising versus three months ago (**$7,900**).

## Evidence
- Mar spending **$4,200** vs income **$4,100** (Task Input).
- Venture balance **$8,400** (Task Input credit table).
- Interest tool: **$312** total interest paid last 90 days on Venture.
- Next card due **2026-04-18** per payment schedule tool.

## Immediate Things to Do
1. Pay **$500** extra on Venture before the **Apr 18** due date.
2. Pause discretionary dining until May spend is under **$3,800**.

## Next Set of Milestones to Aspire
1. Cut Venture below **$7,500** within 60 days.
2. Build **$2,000** combined checking+savings (~half a month of spend).

</AGENT_OUTCOME>

<NEEDS_TOOL_CALLS>

# Round 1

## Invoked tools

1. **`lookup_user_account_interest_rates`**

```json
{"lookback_days": 90}
```

2. **`lookup_user_account_payment_schedule`**

```json
{}
```

</NEEDS_TOOL_CALLS>
""",
    "output": '{"score": 5, "notes": "Grounded debt/interest triage with mandatory tools, evidence, and measurable near-term steps."}',
  },
  {
    "name": "bad_invented_interest_no_tools",
    "batch": 1,
    "input": """<AGENT_OUTCOME>

# Task Input

## Spending and income (last 3 completed months)

| Period | Total Spending | Total Income |
| --- | --- | --- |
| Mar 01, 2026 – Mar 31, 2026 | $3,500 | $4,200 |

## Credit

| id | name | balance | limit |
| --- | --- | --- | --- |
| 12 | Venture | $5,000 | $20,000 |

# Financial Needs

## Summary
You paid **$900** in interest last month and should focus on debt.

## Primary needs
1. **Settle debt** — Balance **$5,000**.

## Evidence
- Interest **$900** last month.
- Balance **$5,000**.

## Immediate Things to Do
1. Pay off the card.

## Next Set of Milestones to Aspire
1. Be debt free.

</AGENT_OUTCOME>

<NEEDS_TOOL_CALLS>

# Round 1

_No tool calls this round._

</NEEDS_TOOL_CALLS>
""",
    "output": '{"score": 1, "notes": "Invented $900 interest, skipped mandatory interest/payment tools, and milestones are vague."}',
  },
  {
    "name": "good_emergency_fund_thin_liquid",
    "batch": 1,
    "input": """<AGENT_OUTCOME>

# Task Input

## Spending and income (last 3 completed months)

| Period | Total Spending | Total Income |
| --- | --- | --- |
| Mar 01, 2026 – Mar 31, 2026 | $5,100 | $5,200 |

## Depository

| id | name | kind | balance | min low (date) | max high (date) |
| --- | --- | --- | --- | --- | --- |
| 1 | Checking | checking | $1,200 | $800 (2026-03-05) | $2,100 (2026-03-28) |
| 2 | Savings | savings | $400 | $350 (2026-02-10) | $450 (2026-01-15) |

# Financial Needs

## Summary
Income covers spend, but combined liquid cash (~**$1,600**) is well under one month of **$5,100** spending.

## Primary needs
1. **Build emergency fund** — Only **~$1,600** liquid vs **~$5,100**/mo spend (<1 month).

## Evidence
- Mar spend **$5,100**; income **$5,200** (Task Input).
- Checking **$1,200** + savings **$400** = **$1,600** liquid.
- Payment schedule: no urgent credit due dates in next 14 days.

## Immediate Things to Do
1. Auto-transfer **$300**/mo to savings after each paycheck.
2. Set a Penny savings goal for **$5,000** emergency fund.

## Next Set of Milestones to Aspire
1. Reach **$2,500** liquid (~2 weeks of spend) in 90 days.
2. Reach **$5,100** (1 month expenses) in 6 months.

</AGENT_OUTCOME>

<NEEDS_TOOL_CALLS>

# Round 1

## Invoked tools

1. **`lookup_user_account_interest_rates`**

```json
{"lookback_days": 90}
```

2. **`lookup_user_account_payment_schedule`**

```json
{}
```

</NEEDS_TOOL_CALLS>
""",
    "output": '{"score": 5, "notes": "Correct emergency-fund triage grounded in Task Input balances and spend with concrete milestones."}',
  },
  {
    "name": "medium_vague_milestones",
    "batch": 2,
    "input": """<AGENT_OUTCOME>

# Task Input

## Spending and income (last 3 completed months)

| Period | Total Spending | Total Income |
| --- | --- | --- |
| Mar 01, 2026 – Mar 31, 2026 | $3,800 | $3,750 |

# Financial Needs

## Summary
Cash flow is tight with spend slightly above income in March.

## Primary needs
1. **Stabilize cash flow** — Mar spend **$3,800** vs income **$3,750**.

## Evidence
- Mar spend **$3,800** vs income **$3,750**.

## Immediate Things to Do
1. Review subscriptions this week.
2. Align bill due dates with payday.

## Next Set of Milestones to Aspire
1. Feel more comfortable with monthly bills.
2. Save more when possible.

</AGENT_OUTCOME>

<NEEDS_TOOL_CALLS>

# Round 1

## Invoked tools

1. **`lookup_user_account_payment_schedule`**

```json
{}
```

</NEEDS_TOOL_CALLS>
""",
    "output": '{"score": 3, "notes": "Cash-flow triage is grounded but milestones lack $ targets and interest tool was skipped for a thin-cash user."}',
  },
  {
    "name": "bad_four_needs_over_limit",
    "batch": 2,
    "input": """<AGENT_OUTCOME>

# Task Input

## Spending and income (last 3 completed months)

| Period | Total Spending | Total Income |
| --- | --- | --- |
| Mar 01, 2026 – Mar 31, 2026 | $4,000 | $4,500 |

# Financial Needs

## Summary
Several areas need work.

## Primary needs
1. **Save money**
2. **Build emergency fund**
3. **Settle debt**
4. **Stabilize cash flow**

## Evidence
- Spend **$4,000**; income **$4,500**.

## Immediate Things to Do
1. Do everything at once.

## Next Set of Milestones to Aspire
1. Improve finances.

</AGENT_OUTCOME>

<NEEDS_TOOL_CALLS>

# Round 1

## Invoked tools

1. **`lookup_user_account_interest_rates`**

```json
{}
```

2. **`lookup_user_account_payment_schedule`**

```json
{}
```

</NEEDS_TOOL_CALLS>
""",
    "output": '{"score": 2, "notes": "Lists four primary needs (max 3), vague steps/milestones, and surplus cash-flow does not support all four needs equally."}',
  },
]


class DetermineFinancialNeedsCheckerOptimizer:
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

  opt = DetermineFinancialNeedsCheckerOptimizer(
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
