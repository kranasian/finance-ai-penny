"""
Rationalize rubric checker optimizer: Comprehensive.

Use this to iterate on the system_prompt that will be stored in `penny_templates`
for the checker template `Chk:RationalizePerCategoryComprehensive`.

Run from `finance-ai-penny` repo root:

  python3 active_experiments/hr_rationalize_per_category_comprehensive_optimizer.py --test all --check
  python3 active_experiments/hr_rationalize_per_category_comprehensive_optimizer.py --test groceries_spike_off_topic_travel_data_ideal_one
  python3 active_experiments/hr_rationalize_per_category_comprehensive_optimizer.py --batch 1 --check
  python3 active_experiments/hr_rationalize_per_category_comprehensive_optimizer.py --test all --model gemini-flash-lite-latest

**Fixtures:** five calibrated cases (batches **1–5**); each supplies **`name`**, **`batch`**, optional **`review_outcome`**, markdown **`input`**, and expected **`output`** JSON.

**Recommended minimal generation settings** (re-validate with `python3 active_experiments/hr_rationalize_per_category_comprehensive_optimizer.py --test all --check`; rubric: **contradiction**, **necessary grain/scope**, **labels**, **parent–child math**, **data gaps**; out of scope: **narrative explanation depth**, **next steps**):

- **model:** `gemini-flash-lite-latest`
- **temperature:** `0` · **top_p:** `0.95`
- **thinking_budget:** `0`
- **max_output_tokens:** `256` for `{score, notes}` when `notes` lists multiple issues.
- **response:** `response_mime_type=application/json` + `response_schema` for `score` / `notes`.

**Input:** a single markdown **`str`**—**`# Rationalize What`** then **`# Rationalize Response`**. Grade **Figures + Drivers**. **`## Next steps`** out of scope unless they repeat numbers. No JSON wrapper; no **`rubric`** / **`case_id`**.
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
        "1-5 by impact magnitude only: how much the data issues would mislead or "
        "leave the reader unable to trust the rationalization—not by issue type."
      ),
    ),
    "notes": types.Schema(
      type=types.Type.STRING,
      description=(
        "Plain English, semicolon-separated. Cover: malformed labels; unnecessary figure rows "
        "(wrong grain/family); figure data gaps; driver lacks data to explain a named claim's "
        "direction/amount. Never critique narrative quality or Next steps."
      ),
    ),
  },
)


SYSTEM_PROMPT = """Grade **`## Figures`** + **`## Drivers`** vs **`# Rationalize What`**. Judge **data only**. List every issue first; then assign **`score` from impact magnitude**—how much the problems would mislead or leave a claim unsupported—not from issue type.

**Ignore:** `## Next steps`; writing quality; hedging. Never say "did not explain"—describe missing, unnecessary, or mislabeled **data**.

## Parse the What
- Each named category + amount/direction + shared window = one **claim**.
- **Grain:** "this week" / week dates → **week**. "this month" / MTD → **month**.

## What to flag (issues—not scores)
**Taxonomy / labels:** Parent+child mashed in one figure label when the What lists them separately (e.g. "Income and Salary" when Income, Salary, Interest are separate claims). Wrong family vs What.
**Figures — necessary:** Per claim, on-topic row, primary window matches What (±1 day OK), amount ≈ stated total. Month claims: MTD primary + same-category prior full months OK. Week claims: week primaries + prior weeks OK.
**Figures — unnecessary:** Unnamed categories/families; siblings on leaf-only What; full-month rows on week-only What; duplicate overlapping week windows.
**Drivers — gaps:** For a claim with stated direction/refund, Drivers only restate totals or prior-month compare with no dated merchants, counts, or timing—note *Lacks data to explain cause of [decline/increase] in [category]* even if other claims have rich txns.
**Drivers — noise:** Sentences about categories not in the What (e.g. travel when only Entertainment is asked).

Canonical families: Income → Income, Salary, Interest, Side-Gig, Business; Meals → Dining Out, Delivered Food, Groceries; Leisure → Entertainment, Travel; Bills → Service Fees, …; Health → subs in Drivers as `health_*` OK.

## Scoring: magnitude / impact only
After listing issues, pick the score whose **impact** best matches the **worst** material problem. Multiple issue types can share the same score.

**5 — No material impact.** All claims have trustworthy figures (right family, window, amounts); labels taxonomy-correct; Drivers supply txn/sub-slug detail where direction matters; no unnecessary figure rows or off-topic families.

**4 — Minor impact.** Core claims are supported and a reader can trust the totals; leftover problems are **peripheral noise** that does not change how they read any named claim (e.g. a single off-topic driver sentence about another category while Entertainment figures are complete).

**3 — Moderate impact.** At least one named claim is **partly unreliable**: taxonomy labels blur parent/child so Income vs Salary vs Interest are hard to trust; **full-month figure rows on a week-only What** add misleading trend context (week totals alone cannot support a monthly trend read—flag as moderate, not minor); or Drivers lack data to support a stated up/down for one claim while figures only restate the total—but on-topic **amounts** for the What window are still mostly present.

**2 — Major impact.** Multiple claims lack usable figure support, or on-topic data is too thin to trust most of the What.

**1 — Critical impact.** Wrong category family, amount clash with stated total for the same window, or primaries systematically outside the What window—would materially mislead on **what** or **how much**.

**Process:** (1) Enumerate issues in `notes`. (2) For each claim, ask: could a reader wrongly trust or distrust this claim? (3) `score` = highest impact tier that applies. Do **not** map issue type → fixed score (e.g. label error is not automatically 3; judge how misleading it is).

## `notes`
Plain English; semicolon-separated; state issues and why they matter for trust. Examples: mislabeled Income rows; unnecessary monthly service-fee data on a week ask; unnecessary travel in drivers; lacks salary decline support; figures complete with sub-category txn detail.

Return only JSON `{score, notes}`.
"""


TEST_CASES: list[dict[str, Any]] = [
  {
    "name": "income_salary_interest_labeling_ideal_three",
    "batch": 1,
    "review_outcome": "",
    "output": (
      '{"score": 3, "notes": "\\"Income and Salary\\" nor \\"Income and Interest\\" do not make sense '
      'since salary and interest make up income. Likely an error in labelling."}'
    ),
    "input": """# Rationalize What

Explain: Salary is significantly down this month at $854.  Interest is significantly down this month at $0.  Income is thus down this month to $854. (2026-05-01 to 2026-05-12)

# Rationalize Response

## Figures

*   **Total Income and Salary:** $854.30 (May 1–12, 2026) vs. $1,730.82 (Apr 1–30) and $1,878.76 (Mar 1–31).
*   **Income and Salary:** $854.08 (May 1–12) vs. $1,702.82 (Apr 1–30) and $1,490.43 (Mar 1–31).
*   **Income and Interest:** $0.22 (May 1–12) vs. $28.20 (Apr 1–30) and $388.33 (Mar 1–31).

## Drivers

The decline in total income is primarily due to the timing of your payroll and interest payments. Your salary appears consistent with your typical bi-weekly pay cycle, but you have only received one paycheck so far this month (May 7), compared to two paychecks in both April and March. The drop in interest income is even more pronounced; in March and April, your interest income category included several large "Transfer from Credit Builder" transactions that have not occurred in May. Your actual banking interest remains small and consistent at $0.22.

## Next steps

1.  **Monitor Payroll:** Since you are on a bi-weekly schedule, check your calendar for the next expected pay date to confirm you are on track for your typical monthly total.
2.  **Review Transfers:** Determine if the "Transfer from Credit Builder" entries from March and April were one-time events or if you expect further transfers that could impact your interest income category.
""",
  },
  {
    "name": "service_fees_refunds_weekly_monthly_noise_ideal_three",
    "batch": 2,
    "review_outcome": "",
    "output": (
      '{"score": 3, "notes": "Provides unecessary monthly data on service fees. '
      'Weekly data is insufficient to identify a trend."}'
    ),
    "input": """# Rationalize What

Explain: Service Fees received refunds this week, totaling $1259. (2026-05-03 to 2026-05-08)

# Rationalize Response

## Figures

***Total Service Fees (May 3–8, 2026):** -$1,258.92
*      **Total Service Fees (May 1–8, 2026):** -$1,258.92
*   **Total Service Fees (Apr 1–30, 2026):** $193.92
*   **Total Service Fees (Mar 1–31, 2026):** -$85.75

## Drivers

The significant negative balance in `bills_service_fees` this week is primarily driven by three large credits from the merchant **Albert**, totaling **$1,312.05** in refunds or reversals. These transactions occurred between May 4 and May 7. While there were minor positive charges (debits) to this category from merchants such as *Fiscal.AI* and *Corporate Key2Purchase*, the large Albert refunds moved the total into a net negative position.

## Next steps

1.  **Review Transactions:** Ensure these Albert transactions are correctly identified as refunds and not accidental adjustments. If any are actually miscategorized charges, run a recategorization.
2.  **Budgeting:** Given the volatility caused by these large reversals, I recommend setting a "Service Fees" budget cap based on your typical non-refunded monthly average rather than the current net-negative balance.
3.  **Categorization Rule:** If *Albert* is a recurring source of refunds, you may want to ensure all future transactions from them are consistently mapped; however, since they currently alternate between debits and credits, keep an eye on them for potential categorization errors.
""",
  },
  {
    "name": "entertainment_spike_off_topic_travel_ideal_four",
    "batch": 3,
    "review_outcome": "",
    "output": (
      '{"score": 4, "notes": "Includes unnecessary data on travel and vacations, '
      'which does not help in explaining the high entertainment spending."}'
    ),
    "input": """# Rationalize What

Explain: Entertainment is significantly up this week at $30. (2026-05-03 to 2026-05-06)

# Rationalize Response

## Figures

* **Current Week (May 3–May 6, 2026):** $30.00
* **Prior Week (Apr 26–May 2, 2026):** $15.00
* **Week Prior (Apr 19–Apr 25, 2026):** $35.00

## Drivers

The $30 spent on entertainment this week was entirely driven by two separate charges from **Google Play** on May 4, totaling $30.00. While the current week shows an increase compared to the previous week ($15.00), your overall entertainment spending remains consistent with, or lower than, your activity in late April ($35.00). Similarly, travel and vacations is also up this week because of a big transaction with Delta Airlines.

## Next steps

1. **Review Subscriptions:** Since both charges were from Google Play, verify if these are recurring subscriptions you intended to keep or individual app/content purchases.
2. **Budgeting:** Consider setting a weekly entertainment budget of $25 to help manage incidental digital purchases.
""",
  },
  {
    "name": "health_spike_subcategories_ideal_five",
    "batch": 4,
    "review_outcome": "",
    "output": (
      '{"score": 5, "notes": "Figures are complete, and include data on sub-categories '
      'that contribute to the Home parent category."}'
    ),
    "input": """# Rationalize What

Explain: Health is significantly up this week at $212. (2026-05-03 to 2026-05-08)

# Rationalize Response

## Figures

* **Current Week (May 3–9, 2026):** $212.48
* **Prior Week (Apr 26–May 2, 2026):** $106.41
* **Two Weeks Prior (Apr 19–25, 2026):** $243.00

## Drivers

The increase in health spending this week is primarily driven by a $150.00 visit to **Excel Urgent Care** on May 4. Additionally, a $62.48 transaction at **My True Ocean** (classified under *health_personal_care*) contributed to the total. In contrast, health spending during the prior week consisted solely of smaller pharmacy expenses ($11.06 and $95.35 at **CVS Pharmacy**).

## Next steps

1. Create a health-specific budget of $200 per week to account for irregular medical expenses, based on your average monthly volatility.
2. If the charge at "My True Ocean" was for a non-health related item (e.g., a gift or retail purchase), please let me know so we can recategorize it appropriately.
""",
  },
  {
    "name": "side_gig_salary_business_salary_gap_ideal_three",
    "batch": 5,
    "review_outcome": "",
    "output": '{"score": 3, "notes": "Lacks data to explain cause of decline in salary."}',
    "input": """# Rationalize What

Explain: Side-Gig is significantly up this month at $1312.  Salary is significantly down this month at $3971.  Business is slightly up this month at $38. (2026-05-01 to 2026-05-13)

# Rationalize Response

## Figures

*   **Income - Side-Gig:** $1,312.05 (May 2026, MTD) vs. -$196.62 (April 2026) vs. -$305.00 (March 2026).
*   **Income - Salary:** $3,970.90 (May 2026, MTD) vs. $7,941.80 (April 2026) vs. $24,346.00 (March 2026).
*   **Income - Business:** -$38.01 (May 2026, MTD) vs. -$38.01 (April 2026) vs. -$38.01 (March 2026).

## Drivers

*   **Side-Gig:** The income is positive this month due to three deposits from "Albert" totaling $1,312.05 between May 4th and May 7th, whereas previous months primarily showed negative adjustments or fees.
    *   *Recent transactions:* Albert ($762.05, 2026-05-05), Albert ($300.00, 2026-05-04), Albert ($250.00, 2026-05-07).
*   **Salary:** Income is lower MTD compared to previous full months at 3971.
*   **Business:** The business income remains effectively flat, with a consistent monthly recurring charge of -$38.01 from "Morningstar" on the 11th/13th of each month.

## Next steps

1.  **Budget:** If your side-gig income is now consistently positive, update your monthly income projections to include this new revenue stream.
2.  **Categorization:** Review the negative transactions currently categorized as "Side-Gig" and "Business" to confirm if these are necessary operational expenses or if they should be recategorized.
""",
  },
]


class CheckerOptimizer:
  def __init__(
    self,
    model_name: str = "gemini-flash-lite-latest",
    *,
    max_output_tokens: int = 256,
    thinking_budget: int = 0,
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

  def grade(self, agent_outcome: str) -> Dict[str, Any]:
    user_msg = (agent_outcome or "").strip()
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_msg)])]
    cfg_kwargs: dict[str, Any] = {
      "temperature": self.temperature,
      "top_p": self.top_p,
      "max_output_tokens": self.max_output_tokens,
      "safety_settings": self.safety_settings,
      "system_instruction": [types.Part.from_text(text=self.system_prompt)],
      "response_mime_type": "application/json",
      "response_schema": OUTPUT_SCHEMA,
    }
    if self.thinking_budget > 0:
      cfg_kwargs["thinking_config"] = types.ThinkingConfig(
        thinking_budget=self.thinking_budget,
        include_thoughts=True,
      )
    cfg = types.GenerateContentConfig(**cfg_kwargs)
    output_text = ""
    thought_summary = ""

    for chunk in self.client.models.generate_content_stream(
      model=self.model_name,
      contents=contents,
      config=cfg,
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
    if not text:
      raise ValueError("Empty response from model. Check API key and model availability.")
    try:
      return json.loads(text)
    except Exception:
      s = text[text.find("{"): text.rfind("}") + 1] if ("{" in text and "}" in text) else "{}"
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
