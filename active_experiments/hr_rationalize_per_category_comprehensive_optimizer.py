"""
Rationalize rubric checker optimizer: Comprehensive.

Use this to iterate on the system_prompt that will be stored in `penny_templates`
for the checker template `Chk:RationalizePerCategoryComprehensive`.

Run from `finance-ai-penny` repo root:

  python3 active_experiments/hr_rationalize_per_category_comprehensive_optimizer.py --test all
  python3 active_experiments/hr_rationalize_per_category_comprehensive_optimizer.py --batch 1 --check
  python3 active_experiments/hr_rationalize_per_category_comprehensive_optimizer.py --test all --model gemini-flash-lite-latest

**Recommended minimal generation settings** (re-validate `--test all --check`; scores match fixture `output`):

- **model:** `gemini-flash-lite-latest`
- **temperature:** `0` · **top_p:** `0.95`
- **thinking_budget:** `0` (internal reasoning; `include_thoughts=False` so JSON output stays reliable)
- **max_output_tokens:** `128`
- **response:** `application/json` + **response_schema** for `{score, notes}`

**Rubric:** Grade **`## Figures`** vs **`# Rationalize What`**. Read **`## Drivers`** only for **What–Figures** discrepancy acknowledgment. **`notes` always non-empty** (including score 5). **9 fixtures** (batches **1–9**); use **`--batch N --check`**.

**Input:** a single markdown **`str`**—`# Rationalize What` then `# Rationalize Response` (same shape as `ai_agent_outcomes.agent_outcome` / insightful optimizer).
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
from google import genai
from google.genai import types

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


OUTPUT_SCHEMA = types.Schema(
  type=types.Type.OBJECT,
  required=["score", "notes"],
  properties={
    "score": types.Schema(
      type=types.Type.INTEGER,
      description=(
        "1-5 weakest-claim impact. Trend at claim grain only. Grain-family mix → ≤3. Thin trend sole defect: 2 periods=4, 1 period=3. Parent-only or parent+sub partial sub breakdown, or parent+sub missing claimed sub → ≤3."
      ),
    ),
    "notes": types.Schema(
      type=types.Type.STRING,
      description=(
        "Always non-empty; semicolon-separated. score=5: '{Category} {scope} claim; three {grain} periods; {detail}; figure label and {grain} grain OK.' score≤4: issues only—never cite acknowledged discrepancies; use parent+sub OR parent-only template, not both. Parent+sub missing claimed sub: '{Sub} and {Parent} claims present but Figures omit {Sub} row.' Parent+sub partial unclaimed subs: '{Parent} parent+sub claim but Figures show parent row plus partial unclaimed subs; {missing} subcategory missing from breakdown.' Parent-only incomplete: '{Parent} parent-only claim but Figures show only sub rows without parent row; {missing} subcategory missing.' or '{Parent} parent-only claim but Figures show parent row plus partial subs; {missing} subcategory missing from breakdown.' Never fault absent unclaimed subs when Figures contain only parent row plus claimed sub rows."
      ),
    ),
  },
)


SYSTEM_PROMPT = """Grade **`## Figures`** vs **`# Rationalize What`**. Evidence = Figures only (labels, windows, amounts, derivations). Skim **Drivers** only per **Discrepancies**; ignore Drivers narrative and **Next steps**.

**Notes:** semicolon-separated; use "three monthly periods" / "month grain" (not monthly). score 5 uses template above; score ≤4 issues only. Name the specific category in notes.

## Category taxonomy (parent → subcategories)
Synonyms allowed. Parent rolls up subs; sub ≠ parent for labeling.
- **Bills**: Connectivity, Insurance, Taxes, Service Fees
- **Donations & Gifts**
- **Education**: Kids Activities, Tuition
- **Health**: Medical & Pharmacy, Gym & Wellness, Personal Care
- **Income**: Salary, Side-Gig, Business, Interest
- **Leisure**: Entertainment, Travel & Vacations
- **Meals/Food**: Dining Out, Delivered Food, Groceries
- **Miscellaneous**
- **Shelter**: Home, Utilities, Upkeep
- **Shopping**: Clothing, Gadgets, Kids, Pets
- **Transfer**
- **Transport**: Car & Fuel, Public Transit
- **Uncategorized**

## What scope
Classify from **What text only**—Figure rows never change scope.
- **Parent-only:** parent node named—no sub tokens in What.
- **Subcategory-only:** only sub(s) named—sub name alone is never parent-only.
- **Parent + sub:** parent token **and** sub token both appear in What—never reclassify as parent-only when both are named.

**Amounts:** rounded $ vs What is OK. **Sign:** spending negative = net inflow; Income negative = net outflow—valid, never a defect or Drivers discrepancy.

## Figures scope
**Categories**
- **Parent-only What:** **parent row only** (sub rows may be absent—never fault missing subs when a parent row is present), or **parent row plus all taxonomy subs** together. **Sub rows without parent row → score 3—every taxonomy sub required; partial subs never satisfy parent-only.** **Parent row plus partial subs → score 3—when subs appear, all taxonomy subs must accompany the parent row.**
- **Subcategory-only What:** claimed sub rows only; no parent rows.
- **Parent + sub What:** parent row when parent is claimed; each **claimed sub** needs its own figure row—a parent rollup row does **not** substitute for a claimed sub. Permitted shapes: **parent row plus claimed sub rows only**, or **parent row plus all taxonomy subs**. **Parent row plus claimed subs plus partial unclaimed subs → score 3.** Absent unclaimed subs are valid when Figures contain only the parent row and claimed sub rows. **Do not apply parent-only completeness gates to parent+sub What.**

**Grain:** week claim → week family only (~7-day/WTD); month claim → month family only (calendar/MTD). **If Figures contain both families on a week/month What → score ≤3 (grain mix beats thin-trend 4).** Month rows never satisfy week-claim depth. Unlike lengths within same family are OK.

**Trend:** count periods **at claim grain only** by date windows (each amount clause counts, regardless of sign). Cross-family rows do not count toward depth. ≥3 at claim grain satisfies depth. Sole defect: 2 periods → **4**; 1 period → **3**. Never assign 3 when count is exactly 2 and no other defect.

**Stated math:** derivations must match taxonomy; partial subs cannot substitute for parent or missing subs.

**Labels:** each row maps to one taxonomy target. Synonyms allowed. Parent rollup labels valid for parent rows; sub labels valid for sub rows. **Defects (~3):** one row label coordinating parent and sub → ambiguous; add misread-risk clause for label-only faults.

**Discrepancies:** $/direction/window gaps acknowledged in **Figures or Drivers** (including narrative figure rows) → no fault, omit from notes. Sign alone never counts. Fault only unacknowledged mismatches.

## Scores (1–5, weakest claim)
5 = all claims supported; **never 5** if parent-only What has partial sub Figures (subs without parent row, or parent row with incomplete subs); **never 5** if parent+sub What has partial unclaimed sub Figures (some but not all unclaimed subs present); **never 5** if claim grain and figure rows mix week/month families. 4 = one minor issue (**exactly 2 periods** at claim grain = thin trend). 3 = moderate. 2 = major. 1 = critical.

## Process
1. **Enumerate Figures rows**—list every category row present before scoring; never fault a missing row without confirming absence in Figures.
2. Classify scope from What text only—both parent and sub named → **parent+sub** (skip parent-only gate).
3. **Parent-only only:** **parent row only** (no sub rows) → category OK, continue. If **sub rows present:** require **parent row plus all taxonomy subs**; **no parent row** or **any missing sub → 3, stop**.
4. **Parent+sub only:** parent row when parent is claimed; figure row per claimed sub—parent row does not replace a missing claimed sub → ≤3 if any claimed sub absent. If **only parent and claimed sub rows** → unclaimed subs need not appear. If **any unclaimed sub row appears** → require **parent row plus all taxonomy subs**; **any missing sub → 3, stop**.
5. **Grain:** detect family mix first—both week and month rows present → **3**. Count trend at claim grain only.
6. Thin trend sole defect: 2 periods → **4**. Weakest claim. Discrepancy OK if in Figures or Drivers.

Return only JSON `{score, notes}`.
"""


TEST_CASES: list[dict[str, Any]] = [
  {
    "name": "business_figures_negative_amounts_acceptable",
    "batch": 1,
    "output": (
      '{"score": 5, "notes": "Business subcategory-only claim; three monthly periods; '
      'negative May net reflects business expenses; figure label and month grain OK."}'
    ),
    "input": """# Rationalize What

Explain: Business income is up this month at $1,850. (2026-06-01 to 2026-06-30)

# Rationalize Response

## Figures

* **Business (Jun 1–30, 2026):** $1,850.00 vs -$320.00 (May 1–31, 2026) vs $2,100.00 (Apr 1–30, 2026).

## Drivers

June client invoices posted; May net was negative after equipment and supply purchases.

## Next steps

1. Track income_business vs business expense tags.
""",
  },
  {
    "name": "transport_figures_parent_only_acceptable",
    "batch": 2,
    "output": (
      '{"score": 5, "notes": "Total Transport is an acceptable parent Transport label; '
      'three monthly periods; parent-only What scope OK."}'
    ),
    "input": """# Rationalize What

Explain: Transport spending is up this month at $680. (2026-06-01 to 2026-06-30)

# Rationalize Response

## Figures

* **Total Transport (Jun 1–30, 2026):** $680.00 vs $412.00 (May 1–31, 2026) vs $390.00 (Apr 1–30, 2026).

## Drivers

Fuel and a maintenance bill posted in June.

## Next steps

1. Review transportation_car spend.
""",
  },
  {
    "name": "food_dining_out_parent_sub_figures_acceptable",
    "batch": 3,
    "output": (
      '{"score": 5, "notes": "Food (Total) and Dining Out cover parent+sub claims; '
      'three monthly periods; parent+sub What scope OK."}'
    ),
    "input": """# Rationalize What

Explain: Dining Out is significantly up this month at $320.  Food is thus up this month to $785. (2026-06-01 to 2026-06-30)

# Rationalize Response

## Figures

* **Food (Total) (Jun 1–30, 2026):** $785.00 vs $590.00 (May 1–31, 2026) vs $545.00 (Apr 1–30, 2026).
* **Dining Out (Jun 1–30, 2026):** $320.00 vs $210.00 (May 1–31, 2026) vs $195.00 (Apr 1–30, 2026).

## Drivers

More restaurant spend drove June Food and Dining Out totals.

## Next steps

1. Review meals_dining_out budget.
""",
  },
  {
    "name": "food_figures_partial_sub_breakdown_unacceptable",
    "batch": 4,
    "output": (
      '{"score": 3, "notes": "Food parent-only claim but Figures show only Dining Out and '
      'Groceries; Delivered Food subcategory missing from parent breakdown."}'
    ),
    "input": """# Rationalize What

Explain: Food spending is up this month at $785. (2026-06-01 to 2026-06-30)

# Rationalize Response

## Figures

* **Dining Out (Jun 1–30, 2026):** $320.00 vs $210.00 (May 1–31, 2026) vs $195.00 (Apr 1–30, 2026).
* **Groceries (Jun 1–30, 2026):** $465.00 vs $380.00 (May 1–31, 2026) vs $350.00 (Apr 1–30, 2026).

## Drivers

More restaurant and grocery spend in June.

## Next steps

1. Review meals_dining_out and meals_groceries budgets.
""",
  },
  {
    "name": "groceries_figures_two_periods_thin_trend",
    "batch": 5,
    "output": (
      '{"score": 4, "notes": "thin trend for Groceries—only two monthly periods, '
      'no third period for trend comparison."}'
    ),
    "input": """# Rationalize What

Explain: Groceries spending is up this month at $465. (2026-06-01 to 2026-06-30)

# Rationalize Response

## Figures

* **Groceries (Jun 1–30, 2026):** $465.00 vs $380.00 (May 1–31, 2026).

## Drivers

More grocery trips and a Costco run posted in June.

## Next steps

1. Review meals_groceries budget.
""",
  },
  {
    "name": "food_weekly_what_monthly_figures_grain_mismatch",
    "batch": 6,
    "output": (
      '{"score": 3, "notes": "Food weekly claim but Figures mix two week-grain periods with '
      'month-family rows (Jun, May, Apr); thin weekly trend and off-scope monthly data."}'
    ),
    "input": """# Rationalize What

Explain: Food spending is up this week at $125. (2026-06-09 to 2026-06-15)

# Rationalize Response

## Figures

* **Food (Total) (Jun 9–15, 2026):** $125.00 vs $86.00 (Jun 2–8, 2026).
* **Food (Total) (Jun 1–30, 2026):** $785.00 vs $590.00 (May 1–31, 2026) vs $545.00 (Apr 1–30, 2026).

## Drivers

More dining and delivery orders posted this week; Figures mix two weekly windows with three monthly calendar-month totals.

## Next steps

1. Review meals spending week over week.
""",
  },
  {
    "name": "income_sidegig_salary_figures_missing_salary",
    "batch": 7,
    "output": (
      '{"score": 3, "notes": "Side-Gig and Income claims present but Figures omit Salary '
      'row; Total Income cannot support Salary claim without Salary figures."}'
    ),
    "input": """# Rationalize What

Explain: Side-Gig is significantly up this month at $2,158.  Salary is slightly down this month at $2,532.  Income is thus up this month to $4,690. (2026-06-01 to 2026-06-30)

# Rationalize Response

## Figures

* **Side-Gig (Jun 1–30, 2026):** $2,158.00 vs $1,420.00 (May 1–31, 2026) vs $980.00 (Apr 1–30, 2026).
* **Total Income (Jun 1–30, 2026):** $4,690.00 vs $3,850.00 (May 1–31, 2026) vs $3,620.00 (Apr 1–30, 2026).

## Drivers

Frequent kiosk deposits drove June Side-Gig earnings; Salary payroll posted as expected.

## Next steps

1. Set aside 20–30% of Side-Gig income for taxes.
""",
  },
  {
    "name": "clothing_shopping_weekly_what_monthly_figures",
    "batch": 8,
    "output": (
      '{"score": 3, "notes": "Shopping parent+sub weekly claim but Figures mix one week-grain '
      'period with month-family rows; thin weekly trend and off-scope monthly data."}'
    ),
    "input": """# Rationalize What

Explain: Clothing received refunds last week, totaling $168.  Shopping thus received significantly more in refunds last week, totaling $168.

# Rationalize Response

## Figures

*   **Shopping (Clothing) Weekly Total**: $-168.37 (Apr 26–May 2, 2026).
*   **Monthly Shopping (Clothing) Total**: $-111.19 (Apr 1–30, 2026).
*   **Monthly Shopping (Clothing) Total**: $218.16 (Mar 1–31, 2026).
*   **Current Month Shopping (Clothing) Total**: $0.00 (May 1–31, 2026).

## Drivers

The refund activity last week was primarily driven by a single transaction: Kohl's: $-215.17 on April 30. This was partially offset by a charge at Macy's: $46.80 on April 27, resulting in the net $-168.37 weekly figure.
This large refund shifted your total April category spending to $-111.19, a significant reversal from the $218.16 spent in March.

## Next steps

1.  Create a recurring "Clothing" budget of $150 to account for these fluctuations.
2.  If you would like to track future returns more consistently, we can set up a tag rule for "Kohl's" to ensure all transactions remain categorized correctly under "shopping_clothing".
""",
  },
  {
    "name": "salary_income_figures_missing_salary_thin_trend",
    "batch": 9,
    "output": (
      '{"score": 3, "notes": "Salary and Income claims present but Figures omit Salary row; '
      'thin trend for Income—only two monthly periods, no third period for trend comparison."}'
    ),
    "input": """# Rationalize What

Explain: Salary is slightly up last month at $638.  Income is thus up last month to $638.

# Rationalize Response

## Figures

*   **Total Income (Apr 1–30, 2026):** $638.15
*   **Total Income (Mar 1–31, 2026):** $2,138.91
*   **Income Change:** Your total income actually decreased by $1,500.76 in April compared to March, contrary to the suggestion that it increased.

## Drivers

The reported figure of $638.15 for April represents the total income for that month, not an increase.
In March, your total income was significantly higher at $2,138.91. The "Labor Force Group Payroll" transactions indicate that your income fluctuates based on the number and timing of paychecks received within each calendar month, which explains why the monthly totals vary so substantially.

## Next steps

1.  **Budget Review:** Given the volatility in your monthly income, I recommend setting a "minimum monthly income" baseline based on your lowest recent month to better manage your spending.
2.  **Income Goal:** Create a savings goal specifically for the higher-income months (like March) to create a buffer for months with fewer paychecks.
""",
  },
]


class CheckerOptimizer:
  def __init__(
    self,
    model_name: str = "gemini-flash-lite-latest",
    *,
    max_output_tokens: int = 512,
    thinking_budget: int = 256,
  ):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set.")
    self.client = genai.Client(api_key=api_key)
    self.model_name = model_name
    self.temperature = 0.0
    self.top_p = 0.95
    self.max_output_tokens = max_output_tokens
    self.thinking_budget = thinking_budget
    self.system_prompt = SYSTEM_PROMPT
    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    ]

  def _build_config(self, *, thinking_budget: int | None = None) -> types.GenerateContentConfig:
    budget = self.thinking_budget if thinking_budget is None else thinking_budget
    return types.GenerateContentConfig(
      temperature=self.temperature,
      top_p=self.top_p,
      max_output_tokens=self.max_output_tokens,
      safety_settings=self.safety_settings,
      system_instruction=[types.Part.from_text(text=self.system_prompt)],
      thinking_config=types.ThinkingConfig(
        thinking_budget=budget,
        include_thoughts=False,
      ),
      response_mime_type="application/json",
      response_schema=OUTPUT_SCHEMA,
    )

  def _parse_json_response(self, text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
      raise ValueError("Empty response from model. Check API key and model availability.")
    try:
      parsed = json.loads(raw)
    except Exception:
      start, end = raw.find("{"), raw.rfind("}")
      if start < 0 or end <= start:
        raise ValueError(f"No JSON object in model response: {raw[:300]!r}")
      parsed = json.loads(raw[start : end + 1])
    if not isinstance(parsed, dict) or "score" not in parsed:
      raise ValueError(f"Missing score in model response: {parsed!r}")
    return parsed

  def grade(self, agent_outcome: str) -> Dict[str, Any]:
    user_msg = (agent_outcome or "").strip()
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_msg)])]
    output_text = ""
    thought_summary = ""

    for chunk in self.client.models.generate_content_stream(
      model=self.model_name,
      contents=contents,
      config=self._build_config(),
    ):
      if chunk.text is not None:
        output_text += chunk.text

      if hasattr(chunk, "candidates") and chunk.candidates:
        for candidate in chunk.candidates:
          if hasattr(candidate, "content") and candidate.content:
            if hasattr(candidate.content, "parts") and candidate.content.parts:
              for part in candidate.content.parts:
                if hasattr(part, "thought") and part.thought:
                  if hasattr(part, "text") and part.text:
                    if thought_summary:
                      thought_summary += part.text
                    else:
                      thought_summary = part.text

    if thought_summary:
      print(f"{'='*80}")
      print("THOUGHT SUMMARY:")
      print(thought_summary.strip())
      print("=" * 80)

    text = output_text.strip()
    try:
      if text:
        return self._parse_json_response(text)
    except ValueError:
      pass
    resp = self.client.models.generate_content(
      model=self.model_name,
      contents=contents,
      config=self._build_config(thinking_budget=0),
    )
    return self._parse_json_response((resp.text or "").strip())


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--test", type=str, default=None, help="Test name, index, or 'all'.")
  parser.add_argument("--batch", type=int, default=None, help="Run all tests in batch N.")
  parser.add_argument("--model", type=str, default="gemini-flash-lite-latest")
  parser.add_argument("--max-output-tokens", type=int, default=512)
  parser.add_argument("--thinking-budget", type=int, default=256)
  parser.add_argument(
    "--check",
    action="store_true",
    help="Require integer score to match expected JSON per test; exit non-zero on mismatch.",
  )
  args = parser.parse_args()

  if args.test is None and args.batch is None:
    print("Available test cases:")
    for i, tc in enumerate(TEST_CASES):
      batch = tc.get("batch")
      batch_s = str(batch) if isinstance(batch, int) else "—"
      print(f"  {i}: {tc.get('name')} (batch {batch_s})")
    return

  opt = CheckerOptimizer(
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
    if tc.get("output") is not None:
      _print_section_banner("# Expected Output")
      print(tc["output"])
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

  print(f"\n{_TEST_SEPARATOR}\n")
  print(f"# Total tests: {len(cases)}\n")
  if args.check and failures:
    print("# CHECK FAILURES\n")
    for line in failures:
      print(line)
    raise SystemExit(1)
  if args.check and not failures and cases:
    print("# CHECK: all scores matched expected.\n")


if __name__ == "__main__":
  main()
