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

**Rubric:** Grade **`## Figures`** vs **`# Rationalize What`**. Read **`## Drivers`** only for **What–Figures** discrepancy acknowledgment. **`notes` always non-empty** (including score 5). **11 fixtures** (batches **1–11**); use **`--batch N --check`**.

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
        "1-5 by impact magnitude only: how much the data issues would mislead reader."
      ),
    ),
    "notes": types.Schema(
      type=types.Type.STRING,
      description=(
        "Always non-empty. score=5: one concise phrase on-scope rows, period count, label fit, scope OK; note Drivers/Figures $-gap acknowledgment if relevant (no “fully supported” filler). score≤4: semicolon-separated issues (all, regardless of impact); name each claim when the What lists several. No praise fluff; never mention Next steps."
      ),
    ),
  },
)


SYSTEM_PROMPT = """Grade **`## Figures`** against **`# Rationalize What`**. Evidence is **Figures** only (row labels, windows, amounts, stated derivations). Skim **`## Drivers`** only where **Discrepancies** allows; ignore **Drivers** narrative and **`## Next steps`** for scoring.

For **`notes`**: if **score 5**, state adequacy—label fit, period depth, scope, grain—always non-empty. If **score ≤4**, **issues only**: semicolon-separated defects; do **not** mention what passed, satisfied trend, acceptable labels, scope OK, or other mitigating context. Use defect vocabulary where relevant: **mash**, **conflated**, **ambiguous**, **thin trend**, **off-scope**, **invalid derivation**.

## Category taxonomy (parent → subcategories)
Synonyms may appear in inputs. A **parent** rolls up its **subcategories**; a subcategory is **not** interchangeable with its parent for labeling rules.
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

## What Scope (parent vs subcategory)
Use the **Category taxonomy** section.
- **Parent-Only What:** Claim/s name a **parent** only (**no** subcategories).
- **Subcategory-Only What:** Claim/s name a **subcategory/ies** only (**no** parent categories).
- **Parent + Subcategory What:** Claims name parent and **subcategory/ies** (not necessarily all).

**Amounts**
- $ amounts are rounded, and may not be exactly equal to Figures. This is acceptable.

## Figures Scope:
**Categories**
- **Parent-Only What:** Figures should show (1) **parent row**, (2) all **subcategories** under that parent (even subcategories **not** named in the What), or (3) **both**.
- **Subcategory-Only What:** **No** Parent rows allowed. Figures should be about **claimed** subcategories. Flag row when category referred to is not claimed in the What.
- **Parent + Subcategory What:** Figures should show either (1) **parent** row AND **claimed subcategory/ies** row/s, or (2) **parent** row, **claimed subcategory/ies** row/s, and all **subcategories** under the parent (even subcategories **not** named in the What).

**Time Grain**
- **Week** Claim → Figures must stay in the **week family** (~7-day weekly and/or WTD windows only). **Current WTD compared to prior full ~7-day weeks is valid**—unlike window lengths within the week family are **not** a fault. **Month**-family rows on a week What → major fault (~2).
- **Month** Claim → Figures must stay in the **month family** (calendar months and/or MTD windows only). **Current MTD compared to prior full calendar months is valid**—unlike window lengths within the month family are **not** a fault. **Week**-family rows on a month What → major fault (~2).

**Trend Depth**
- ≥3 distinct periods at claim grain. WTD/MTD each count as **one** period at that grain. **MTD focal period + two prior full calendar months**, or **WTD focal period + two prior full ~7-day weeks**, satisfies depth—do **not** list partial-vs-full compares as an issue. Fewer than 3 periods at claim grain → thin trend (~3–4). Two periods only → thin trend (**minor**) when trend language applies. Three weekly rows for the claim (including **$0**) = satisfied.

**Stated Math**
- When Figures show that pulled data was used to compute a row, computation should be logical given the category taxonomy
- Parent Figures should always be a sum of **ALL** subcategories and not only those identified in What and/or Figures.

**Labels**
- Each figure row label must map to **one** taxonomy target for the claim in the What: claimed **parent**, claimed **subcategory**, or a **standalone** line (no subs in taxonomy).
- **Standalone lines** (no subcategories listed under that node in taxonomy): alternate wording is an **acceptable synonym** when it cannot be read as a different taxonomy node—**not** a label defect, **not ambiguous**. When the What names that **standalone** node and the figure label maps **only** to it (cannot map to any **parent-with-subs** node), overall **5** on labels—**never 4** for synonym wording alone. Informal shortenings are **acceptable**—never downgrade for "non-standard taxonomy string" or missing namespace alone.
- **Parent-only What — acceptable:** (a) parent name or clear parent synonym; (b) **Total** (or equivalent whole-parent rollup qualifier) **+ parent family name only**—no listed sub token in the label; (c) for **Income**, earnings-style wording that denotes **full Income parent**, not one sub-line; (d) **only when What is parent-only:** **parent token**, then **single space**, then **one listed sub token**—**or** parent and sub **stacked with no coordinator**—**no** **and**/**&**/**+**/**/comma between them: **acceptable parent-facing label** and **parent-claim figure row**. **(d) never applies to subcategory-only What.** Same **parent + space + sub** surface shape as sub path-style, but on **parent-only What** read it as **parent claim row**, **not** subcategory-only label misfit. When **(d)** applies: overall **5** if month grain and **≥3** periods satisfied and no other rubric axis fails.
- **Subcategory-only What — acceptable:** (a) exact sub name or clear sub synonym; (b) **Total** **immediately before** the **claimed sub** name—no second category token, no coordinator; (c) **path-style**: parent-family namespace token, **single space**, **claimed sub name only**—ledger/slug style for **that sub row**; applies **only** when What is **subcategory-only**; **not** a parent mash.
- **Subcategory-only What — moderate label defect:** label **coordinates** or **pairs** parent-branch scope with the claimed sub using **and**, **&**, **+**, **/**, or comma-separated dual categories—**mash** / **ambiguous**; **any coordinator present overrides path-style or synonym readings**; overall **3**, **never 4 or 5**, when this coordinator mash is the sole defect.
- **Parent-only What — moderate label defect:** applies **only** when a **coordinator** (**and**, **&**, **+**, **/**, comma) **joins** parent token to one listed sub token—**conflated**; **adjacent-only** parent+sub compounds are **excluded**; overall **3**, **never 4**, when this is the sole defect.
- When score **≤4** for label reasons only, **`notes`** use **mash**, **conflated**, or **ambiguous**; for **~3** label faults add a second **issue** clause on **misread risk** (subset vs full parent, or **subcategory-only** vs widened scope). Do **not** cite scope, trend, or grain in **`notes`** when the only problem is label form.

**Discrepancies**
- If **Drivers** acknowledge any mismatch between **What** claims and **Figures** (amounts, direction/trend language, or date window), **do not fault**—overall **5** when Figures otherwise satisfy scope, grain, trend depth, labels, and stated math. **Do not** score for "factually incorrect claim," "invalid derivation," or unstated trend direction when **Drivers** disclose the gap.
- **Fault** only when the mismatch is **not** acknowledged in **Drivers** or **Figures**.

## Scores (1–5, weakest claim)
- **5** — Every claim supported; scope/trend/labels/derivations OK. What-vs-Figures discrepancy **acknowledged in Drivers** is OK at **5**.
- **4** — Trustworthy core; **one minor** issue
- **3** — **Moderate** issue
- **2** — **Major** issue
- **1** — **Critical** issue

## Process
1. Classify What scope. Read Figures. If misaligned with the What, check if acknowledged in Drivers. Ignore issue if acknowledged.
2. Check each claim: categories, time grain, trend depth, stated math, labels.
3. **`notes`:** **Always non-empty** (never `""`). If **score 5**: one concise phrase on-scope rows, period count, label fit, scope OK, What-vs-Figures discrepancy (if any) acknowledged in Drivers; no need to note What vs. Figures discrepancy acknowledgment if because of rounding (no “fully supported” filler). If **score ≤4**: semicolon-separated issues (all, regardless of impact); name each claim when the What lists several.
4. **`score`:** Weakest-claim impact.

Return only JSON `{score, notes}`.
"""


TEST_CASES: list[dict[str, Any]] = [
  {
    "name": "donations_gifts_figures_labeled_giving_acceptable",
    "batch": 1,
    "output": (
      '{"score": 5, "notes": "Giving is an acceptable synonym for Donations & Gifts; '
      'three monthly periods; figure label and month grain OK."}'
    ),
    "input": """# Rationalize What

Explain: Donations & Gifts are up this month at $240. (2026-06-01 to 2026-06-30)

# Rationalize Response

## Figures

* **Giving (Jun 1–30, 2026):** $240.00 vs $180.00 (May 1–31, 2026) vs $95.00 (Apr 1–30, 2026).

## Drivers

A one-time charity pledge and holiday gifts drove June.

## Next steps

1. Review donations_gifts budget.
""",
  },
  {
    "name": "dining_out_figures_food_and_dining_out_label_unacceptable",
    "batch": 2,
    "output": (
      '{"score": 3, "notes": "Food & Dining Out mashes parent Meals/Food with Dining Out sub; '
      'ambiguous vs Dining Out subcategory-only claim."}'
    ),
    "input": """# Rationalize What

Explain: Dining Out is significantly up this month at $420. (2026-06-01 to 2026-06-30)

# Rationalize Response

## Figures

* **Food & Dining Out (Jun 1–30, 2026):** $420.00 vs $280.00 (May 1–31, 2026) vs $260.00 (Apr 1–30, 2026).

## Drivers

More restaurant and café spend in June.

## Next steps

1. Cap meals_dining_out if needed.
""",
  },
  {
    "name": "transport_figures_total_transport_acceptable",
    "batch": 3,
    "output": (
      '{"score": 5, "notes": "Total Transport is an acceptable parent Transport label; '
      'three monthly periods; parent-only What scope OK."}'
    ),
    "input": """# Rationalize What

Explain: Transport spending is up this month at $680. (2026-06-01 to 2026-06-30)

# Rationalize Response

## Figures

* **Total Transport (Jun 1–30, 2026):** $680.40 vs $412.00 (May 1–31, 2026) vs $390.00 (Apr 1–30, 2026).

## Drivers

Fuel and a maintenance bill posted in June.

## Next steps

1. Review transportation_car spend.
""",
  },
  {
    "name": "income_figures_total_earnings_acceptable",
    "batch": 4,
    "output": (
      '{"score": 5, "notes": "Total Earnings is an acceptable paraphrase for parent Income; '
      'three monthly periods; parent-only What scope OK."}'
    ),
    "input": """# Rationalize What

Explain: Income is up this month at $8,400. (2026-06-01 to 2026-06-30)

# Rationalize Response

## Figures

* **Total Earnings (Jun 1–30, 2026):** $8,400.00 vs $7,200.00 (May 1–31, 2026) vs $7,050.00 (Apr 1–30, 2026).

## Drivers

Salary and interest posted as expected; side-gig picked up slightly.

## Next steps

1. Track income_salary vs income_sidegig tags.
""",
  },
  {
    "name": "bills_figures_bills_and_connectivity_unacceptable",
    "batch": 5,
    "output": (
      '{"score": 3, "notes": "Bills and Connectivity reads as parent Bills conflated with one sub; '
      'misleading for a Bills parent-only claim."}'
    ),
    "input": """# Rationalize What

Explain: Bills are up this month at $1,120. (2026-06-01 to 2026-06-30)

# Rationalize Response

## Figures

* **Bills and Connectivity (Jun 1–30, 2026):** $1,120.00 vs $980.00 (May 1–31, 2026) vs $940.00 (Apr 1–30, 2026).

## Drivers

Phone and internet rose with a plan change.

## Next steps

1. Review bills_connectivity charges.
""",
  },
  {
    "name": "education_figures_education_tuition_acceptable",
    "batch": 6,
    "output": (
      '{"score": 5, "notes": "Education Tuition is an acceptable label for Education parent spending; '
      'three monthly periods; parent-only What scope OK."}'
    ),
    "input": """# Rationalize What

Explain: Education spending is up this month at $2,400. (2026-06-01 to 2026-06-30)

# Rationalize Response

## Figures

* **Education Tuition (Jun 1–30, 2026):** $2,400.00 vs $1,550.00 (May 1–31, 2026) vs $1,480.00 (Apr 1–30, 2026).

## Drivers

Summer camp deposits and a tuition installment hit in June.

## Next steps

1. Split education_kids_activities vs education_tuition in review.
""",
  },
  {
    "name": "dining_out_figures_total_dining_out_acceptable",
    "batch": 7,
    "output": (
      '{"score": 5, "notes": "Total Dining Out is acceptable sub-row rollup for Dining Out; '
      'three monthly periods; figure label and month grain OK."}'
    ),
    "input": """# Rationalize What

Explain: Dining Out is significantly up this month at $320. (2026-06-01 to 2026-06-30)

# Rationalize Response

## Figures

* **Total Dining Out (Jun 1–30, 2026):** $320.00 vs $210.00 (May 1–31, 2026) vs $195.00 (Apr 1–30, 2026).

## Drivers

More restaurant spend in June.

## Next steps

1. Review meals_dining_out budget.
""",
  },
  {
    "name": "delivered_food_figures_meals_delivered_food_acceptable",
    "batch": 8,
    "output": (
      '{"score": 5, "notes": "Meals Delivered Food is acceptable path-style label for Delivered Food; '
      'three monthly periods; figure label and month grain OK."}'
    ),
    "input": """# Rationalize What

Explain: Delivered Food is up this month at $180. (2026-06-01 to 2026-06-30)

# Rationalize Response

## Figures

* **Meals Delivered Food (Jun 1–30, 2026):** $180.00 vs $95.00 (May 1–31, 2026) vs $88.00 (Apr 1–30, 2026).

## Drivers

More delivery orders in June.

## Next steps

1. Review meals_delivered_food spending.
""",
  },
  {
    "name": "groceries_food_parent_sub_week_drivers_ack_direction",
    "batch": 9,
    "output": (
      '{"score": 5, "notes": "Groceries and Food (Total) cover parent+sub claims; three weekly '
      'periods; partial-week vs full-week grain OK; What direction vs Figures acknowledged in Drivers."}'
    ),
    "input": """# Rationalize What

Explain: Groceries is significantly up this week at $52.  Food is thus significantly up this week to $52. (2026-05-17 to 2026-05-18)

Category Taxonomy: parent Food (meals: 1), with leaf categories Dining Out (meals_dining_out: 2), Delivered Food (meals_delivered_food: 3), Groceries (meals_groceries: 4)

# Rationalize Response

## Figures

* **Groceries:** $52.20 this week (May 17–18) vs. $86.79 last week (May 10–16) and $0.00 two weeks ago (May 3–9).
* **Food (Total):** $52.20 this week (May 17–18) vs. $125.04 last week (May 10–16) and $0.00 two weeks ago (May 3–9).

## Drivers

Groceries and total Food spending are not "significantly up" this week; rather, they have decreased significantly compared to the $86.79 spent on groceries and $125.04 total food spending from last week. The sole transaction for this period is:
* **Trader Joe's:** $52.20 (2026-05-17)

## Next steps

1. **Monitor weekly trends:** Since spending fluctuates (from $0 to $125), track if your weekly grocery visits are becoming more consistent or if this is just a single trip.
2. **Set a weekly grocery budget:** Given your recent activity, consider setting a budget target of $70/week to manage these variations.
""",
  },
  {
    "name": "income_sidegig_salary_parent_sub_mtd_thin_trend",
    "batch": 10,
    "output": (
      '{"score": 4, "notes": "thin trend for Side-Gig, Salary, and Total Income—only MTD '
      'periods, no prior months for trend comparison."}'
    ),
    "input": """# Rationalize What

Explain: Side-Gig is significantly up this month at $2158.  Salary is slightly down this month at $2532.  Income is thus up this month to $4690. (2026-05-01 to 2026-05-18)

Category Taxonomy: parent Income (income: 47), with leaf categories Salary (income_salary: 36), Side-Gig (income_sidegig: 37), Business (income_business: 38), Interest (income_interest: 39)

# Rationalize Response

## Figures

* **Side-Gig ($2,158.95):** May 1–18, 2026.
* **Salary ($2,531.87):** May 1–18, 2026.
* **Total Income ($4,690.82):** May 1–18, 2026.

## Drivers

The primary driver for your increased total income this month is a significant surge in **Side-Gig** earnings, which reached $2,158.95 so far. This category is exceptionally active with frequent deposits:
* **Kiosk Deposit:** $475.00 (2026-05-17)
* **Kiosk Deposit:** $300.00 (2026-05-18)
* **Kiosk Deposit:** $250.00 (2026-05-09)

**Salary** remains consistent at $2,531.87 for the month to date, reflecting two standard payroll deposits (each $1,265.93 or $1,265.94) on May 1st and May 15th, which is slightly lower than previous cycles where three payroll deposits occasionally occurred within a calendar month.

## Next steps

1. **Budget:** Consider setting aside 20-30% of your Side-Gig income for taxes, as this revenue is significantly higher than usual and typically lacks automatic withholding.
2. **Goal:** Use the extra Side-Gig income to accelerate your primary savings or debt reduction goal.
""",
  },
  {
    "name": "bills_service_fees_taxes_parent_sub_mtd_thin_trend",
    "batch": 11,
    "output": (
      '{"score": 4, "notes": "thin trend for Service Fees and Taxes (single May MTD row each); '
      'Bills (Total) has three monthly periods; Drivers acknowledge partial-month vs full-month comparison."}'
    ),
    "input": """# Rationalize What

Explain: Service Fees is significantly down this month at $170.  Taxes is significantly down this month at $100.  Bills is thus significantly down this month to $471. (2026-05-01 to 2026-05-18)

Category Taxonomy: parent Bills (bills: 9), with leaf categories Connectivity (bills_connectivity: 10), Insurance (bills_insurance: 11), Taxes (bills_tax: 12), Service Fees (bills_service_fees: 13)

# Rationalize Response

## Figures

* **Service Fees:** $170.00 (May 2026)
* **Taxes:** $100.00 (May 2026)
* **Bills (Total):** $470.66 (May 2026) vs. $1,239.21 (April 2026) vs. $1,322.99 (March 2026)

## Drivers

Service Fees and Taxes are lower for the first 18 days of May compared to full prior months, contributing to a significantly lower total in the Bills parent category ($470.66). The spending in Service Fees this month primarily consists of recurring overdraft coverage fees and small charges.

**May 2026 Service Fees and Taxes Transactions:**
* **ACH Withdrawal: IRS:** $100.00 (2026-05-15)
* **Overdraft Coverage Fee:** $20.00 (2026-05-11, 2026-05-08, 2026-05-06, 2026-05-05, 2026-05-04)

## Next steps

1. Review account activity to address the frequency of **Overdraft Coverage Fees**, which are driving the Service Fees total.
2. Set a monthly budget for **Bills** at $500 based on this month's current trend, or establish a goal to reduce overdraft charges by 50% next month.
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

    for chunk in self.client.models.generate_content_stream(
      model=self.model_name,
      contents=contents,
      config=self._build_config(),
    ):
      if chunk.text is not None:
        output_text += chunk.text

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
