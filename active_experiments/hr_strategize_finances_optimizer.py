"""
Strategize-finances rubric optimizer — **quality** (holistic).

Grades ``Hr:StrategizeFinances`` checker templates.

Run from ``finance-ai-penny`` repo root:

  python3 active_experiments/hr_strategize_finances_optimizer.py --test all
  python3 active_experiments/hr_strategize_finances_optimizer.py --test all --check
  python3 active_experiments/hr_strategize_finances_optimizer.py --batch 1 --check
  python3 active_experiments/hr_strategize_finances_optimizer.py --model gemini-flash-lite-latest
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


SYSTEM_PROMPT = """Grade **strategize_finances** outcomes for holistic **quality**. Return JSON `{score, notes}` (integer 1–5; one sentence `notes`).

**Bundle:** `<AGENT_OUTCOME>` … `<STRATEGIZE_TOOL_CALLS>`.

- **`<AGENT_OUTCOME>`** — `# Task Input` → `## Accounts` plus strategist response sections.
- **`<STRATEGIZE_TOOL_CALLS>`** — This run's tool trace (`# Round N`, `## Invoked tools`).

Grade **only** visible text. Do not invent missing data.

**Agent job:** Long-term financial strategy (debt/credit, fixed vs flexible spend, income vs plan) — not transaction listing alone.

**Mandatory tool pass** (should appear in `<STRATEGIZE_TOOL_CALLS>` before final answer):
1. `retrieve_user_spending_forecasts_by_sql` and `retrieve_user_income_forecasts_by_sql`
2. `lookup_user_aggregate_spending` (monthly; weekly optional for crunch)
3. `retrieve_user_spending_transactions_by_sql` and/or `retrieve_user_income_transactions_by_sql` for recent drivers

**Required response sections:**
- `## Situation (grounded)` — 4–8 bullets; each with **$ amount or balance** and **time window**
- `## Strategic priorities` — 3–7 numbered items; highest impact first; one concrete metric each
- `## Next 30–90 days` — ordered checklist

**Grounding:** Account balances/limits must come from `## Accounts` in `# Task Input`. Transaction/forecast $ must come from tools — never invent merchants, amounts, or accounts.

**Strategic quality:** Priorities should be ranked by impact and apply the right lens (debt utilization, fixed vs flexible, forecast gaps, bill timing).

**Product boundaries:** Do not penalize for failing to execute transfers; steps should be framed as user/Penny actions, not claims that actions were already performed.

**Scoring (worst gap wins):**
- **1** — Invented balances/$; missing all mandatory tools; no required sections.
- **2** — Multiple gaps (no forecast SQL, generic priorities without metrics, Situation bullets without $).
- **3** — Single clear gap (skipped transaction SQL, one priority without metric, thin Situation).
- **4** — Minor issue while tools, grounding, and structure are otherwise sound.
- **5** — Full tool pass, grounded Situation, ranked priorities with metrics, executable 30–90 day checklist.

Return **only** the JSON object (`score`, `notes`).
"""


TEST_CASES: list[dict[str, Any]] = [
  {
    "name": "good_grounded_debt_strategy",
    "batch": 1,
    "input": """<AGENT_OUTCOME>

# Task Input

## Accounts

| id | name | type | balance | limit |
| --- | --- | --- | --- | --- |
| 10 | Checking | deposit_checking | $2,400 | |
| 11 | Venture | credit_card | $6,800 | $20,000 |
| 12 | Savings | deposit_savings | $900 | |

## Situation (grounded)
- Venture balance **$6,800** on a **$20,000** limit (~34% utilization) as of Task Input.
- Forecast tools show **$3,900**/mo committed spend vs **$4,200**/mo expected income over the next 3 months.
- Mar aggregate spending: **$4,050** total; **$620** on dining (weekly drill-down).
- Recent transactions include **$285** Venture interest charge on **2026-03-12**.

## Strategic priorities
1. **Cut interest drag** — Target **$0** interest charges within 90 days by paying Venture below statement balance each cycle.
2. **Rebuild liquidity** — Grow combined checking+savings from **$3,300** to **$5,000** (~1.2 months of forecast spend).
3. **Cap flexible dining** — Hold dining under **$450**/mo (down from **$620** in Mar).

## Next 30–90 days
1. Pay **$800** extra on Venture by **Apr 30**.
2. Set a Penny monthly dining budget goal at **$450**.
3. Auto-transfer **$200**/paycheck to savings.
4. Review subscriptions before **May 15** bill cluster.

</AGENT_OUTCOME>

<STRATEGIZE_TOOL_CALLS>

# Round 1

## Invoked tools

1. **`retrieve_user_spending_forecasts_by_sql`**

```json
{"sql_query": "SELECT month_date, SUM(amount) FROM user_spending_forecasts GROUP BY 1"}
```

2. **`retrieve_user_income_forecasts_by_sql`**

```json
{"sql_query": "SELECT month_date, SUM(amount) FROM user_income_forecasts GROUP BY 1"}
```

3. **`lookup_user_aggregate_spending`**

```json
{"granularity": "monthly", "date_in_range": "2026-03-01..2026-03-31"}
```

4. **`retrieve_user_spending_transactions_by_sql`**

```json
{"sql_query": "SELECT date, merchant, amount FROM transactions ORDER BY date DESC LIMIT 30"}
```

</STRATEGIZE_TOOL_CALLS>
""",
    "output": '{"score": 5, "notes": "Mandatory tools ran; Situation and priorities are grounded with $; checklist is concrete."}',
  },
  {
    "name": "bad_invented_balance_no_forecasts",
    "batch": 1,
    "input": """<AGENT_OUTCOME>

# Task Input

## Accounts

| id | name | type | balance | limit |
| --- | --- | --- | --- | --- |
| 10 | Checking | deposit_checking | $2,400 | |
| 11 | Venture | credit_card | $6,800 | $20,000 |

## Situation (grounded)
- You have **$12,000** in checking and heavy debt.
- Spending seems high lately.

## Strategic priorities
1. **Get organized**
2. **Spend less**
3. **Save more**

## Next 30–90 days
1. Think about budget.
2. Talk to advisor.

</AGENT_OUTCOME>

<STRATEGIZE_TOOL_CALLS>

# Round 1

## Invoked tools

1. **`lookup_user_aggregate_spending`**

```json
{"granularity": "monthly"}
```

</STRATEGIZE_TOOL_CALLS>
""",
    "output": '{"score": 1, "notes": "Invented $12,000 checking, skipped forecast SQL, and priorities lack metrics."}',
  },
  {
    "name": "medium_missing_transaction_sql",
    "batch": 1,
    "input": """<AGENT_OUTCOME>

# Task Input

## Accounts

| id | name | type | balance | limit |
| --- | --- | --- | --- | --- |
| 10 | Checking | deposit_checking | $1,100 | |

## Situation (grounded)
- Checking **$1,100** (Task Input).
- Forecast spend **$3,200**/mo vs income **$3,400**/mo next quarter.

## Strategic priorities
1. **Stabilize cash** — Keep checking above **$1,500** before large bills.
2. **Trim subscriptions** — Cut **$80**/mo recurring spend.

## Next 30–90 days
1. List subscriptions and cancel one by **Apr 20**.
2. Move **$150**/mo to savings.

</AGENT_OUTCOME>

<STRATEGIZE_TOOL_CALLS>

# Round 1

## Invoked tools

1. **`retrieve_user_spending_forecasts_by_sql`**

```json
{"sql_query": "SELECT SUM(amount) FROM user_spending_forecasts"}
```

2. **`retrieve_user_income_forecasts_by_sql`**

```json
{"sql_query": "SELECT SUM(amount) FROM user_income_forecasts"}
```

3. **`lookup_user_aggregate_spending`**

```json
{"granularity": "monthly"}
```

</STRATEGIZE_TOOL_CALLS>
""",
    "output": '{"score": 3, "notes": "Forecasts and aggregates present but no transaction SQL for recent drivers; Situation is thin."}',
  },
  {
    "name": "good_fixed_vs_flexible_lens",
    "batch": 2,
    "input": """<AGENT_OUTCOME>

# Task Input

## Accounts

| id | name | type | balance | limit |
| --- | --- | --- | --- | --- |
| 20 | Checking | deposit_checking | $800 | |
| 21 | Mortgage | loan_mortgage | $240,000 | |

## Situation (grounded)
- Checking **$800** with mortgage payment **$2,100** due **2026-04-01** (transaction SQL).
- Forecast committed outflows **$3,600**/mo vs income **$4,000**/mo.
- Fixed shelter+debt service ~**$2,900**/mo; flexible spend averaged **$700**/mo (aggregate).

## Strategic priorities
1. **Protect fixed obligations** — Never let checking drop below **$2,200** before the **$2,100** mortgage on the 1st.
2. **Flex the $700** — Cut discretionary **$200**/mo until checking holds **$2,500**.

## Next 30–90 days
1. Build a **$2,500** checking floor by **May 31**.
2. Pause non-essential shopping until checking recovers.
3. Set Penny reminder 3 days before mortgage due date.

</AGENT_OUTCOME>

<STRATEGIZE_TOOL_CALLS>

# Round 1

## Invoked tools

1. **`retrieve_user_spending_forecasts_by_sql`**

```json
{"sql_query": "SELECT month_date, SUM(amount) FROM user_spending_forecasts GROUP BY 1"}
```

2. **`retrieve_user_income_forecasts_by_sql`**

```json
{"sql_query": "SELECT month_date, SUM(amount) FROM user_income_forecasts GROUP BY 1"}
```

3. **`lookup_user_aggregate_spending`**

```json
{"granularity": "monthly"}
```

4. **`retrieve_user_spending_transactions_by_sql`**

```json
{"sql_query": "SELECT date, merchant, amount FROM transactions WHERE amount > 500 ORDER BY date DESC LIMIT 20"}
```

5. **`retrieve_user_income_transactions_by_sql`**

```json
{"sql_query": "SELECT date, amount FROM income_transactions ORDER BY date DESC LIMIT 10"}
```

</STRATEGIZE_TOOL_CALLS>
""",
    "output": '{"score": 5, "notes": "Fixed-vs-flexible framing with grounded $, full tool pass, and actionable 30–90 day steps."}',
  },
  {
    "name": "bad_claims_transfer_executed",
    "batch": 2,
    "input": """<AGENT_OUTCOME>

# Task Input

## Accounts

| id | name | type | balance | limit |
| --- | --- | --- | --- | --- |
| 10 | Checking | deposit_checking | $3,000 | |
| 12 | Savings | deposit_savings | $500 | |

## Situation (grounded)
- Checking **$3,000**; savings **$500** (Task Input).

## Strategic priorities
1. **Grow savings** — Target **$2,000** in savings.

## Next 30–90 days
1. I already moved **$1,000** to savings for you this week.

</AGENT_OUTCOME>

<STRATEGIZE_TOOL_CALLS>

# Round 1

## Invoked tools

1. **`retrieve_user_spending_forecasts_by_sql`**

```json
{}
```

2. **`retrieve_user_income_forecasts_by_sql`**

```json
{}
```

3. **`lookup_user_aggregate_spending`**

```json
{"granularity": "monthly"}
```

4. **`retrieve_user_spending_transactions_by_sql`**

```json
{"sql_query": "SELECT 1"}
```

</STRATEGIZE_TOOL_CALLS>
""",
    "output": '{"score": 2, "notes": "Claims an executed transfer Penny cannot perform; thin Situation and only one priority."}',
  },
]


class StrategizeFinancesCheckerOptimizer:
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

  opt = StrategizeFinancesCheckerOptimizer(
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
