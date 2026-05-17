"""
Rationalize rubric checker optimizer: Comprehensive.

Use this to iterate on the system_prompt that will be stored in `penny_templates`
for the checker template `Chk:RationalizePerCategoryComprehensive`.

Run from `finance-ai-penny` repo root:

  python3 active_experiments/hr_rationalize_per_category_comprehensive_optimizer.py --test all --check
  python3 active_experiments/hr_rationalize_per_category_comprehensive_optimizer.py --test groceries_spike_off_topic_travel_data_ideal_one
  python3 active_experiments/hr_rationalize_per_category_comprehensive_optimizer.py --batch 1 --check
  python3 active_experiments/hr_rationalize_per_category_comprehensive_optimizer.py --test all --model gemini-flash-lite-latest

**Fixtures:** nine calibrated cases (batches **1–9**); each supplies **`name`**, **`batch`**, optional **`review_outcome`**, markdown **`input`**, and expected **`output`** JSON.

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
      description="Integer 1-5 rubric score.",
    ),
    "notes": types.Schema(
      type=types.Type.STRING,
      description=(
        "Semicolon-separated data-only clauses: unnecessary rows, window, figure data gaps, "
        "contradiction—never 'did not explain'."
      ),
    ),
  },
)


SYSTEM_PROMPT = """Grade **`## Figures`** + **`## Drivers`** against **`# Rationalize What`**. Core: was the **right data pulled**—**necessary** to support each What insight, or **directly connected** to a category/claim named there?

**STEP 1 (stop if true):** Count **distinct category/sub claims** in the What (each named line with amount/direction = 1). If **≥2** and **every** figure row’s **primary** period ends **after** the What’s **end** → **`score` = 1** only (e.g. Groceries + Delivered Food + Dining Out + Food, all primaries May 3–9, What ends May 5). **Never 3** for this pattern.

**Out of scope (never score, never in notes):** **`## Next steps`**; whether Drivers “explain,” narrate, hedge, or pivot; cause depth. Flag **missing on-topic numbers in Figures** (**figure data gap**), never “did not explain” or “drivers failed to.”

**Per What claim** (each category + window + stated amount/direction if any):
1. **Connected figure?** On-topic labeled row, primary in the What window, amount matches the stated total → **amounts pulled** (do **not** call a figure-amount gap).
2. **No figure row** for that claim → **figure data gap** (note: *no [grain] amounts pulled for [claim]*).
3. **Transaction gap (narrow):** Only when the What names **≥2** categories and **another** named claim has a **dated merchant charge** in Drivers (e.g. `$142.50 at Capital Grille on May 12`, `DoorDash` orders) while this claim’s Drivers **only restate the figure total** with no merchants/dates → note *no transactions pulled to support the $X [claim]* (**≤3**). **Do not** apply on **leaf-only** Whats or when **every** claim lacks dated charges. Never say “no week amounts” when Figures already show the stated total.
4. **Figure row** for a category **not** named in the What → **unnecessary data** (note the row/slug).

**Judge order:** wrong family / amount clash → **hard window** → **unnecessary disconnected rows** → **figure data gaps** → draft score.

**Connected (necessary):** Figures for every category **named** in the What; parent-total rows (**Total Meals**, **Shelter (total)**, **Total Bills**) satisfy parent claims (**Food**, **Shelter**, **Bills**) even if labels differ; **all children under a named parent** (e.g. **Home** + **Utilities** when **Shelter** is named); **MoM/WoW comparison legs** (full April/March vs partial May)—never treat comparison legs as window fails. **Unnecessary:** sibling/parent rows on **leaf-only** What; same-leaf **month** row on **week-only** leaf ask; unrelated families.

**Contradiction (→1):** Wrong family; primary **amount** for the **same** window **conflicts** with the What’s stated dollars (e.g. What **$79**, figure **$200**). **Never** contradiction for: “up/down” vs a prior month when the **stated MTD/week amount matches** the primary row; parent “down” vs an anomalous prior month when **May MTD matches** the What; subs that **sum to** the parent in the same window.

**Grain:** **Week** ask → week-labeled primaries required; same-leaf **month** row = unnecessary. **Month/MTD** ask → MTD/partial month + prior full months for **named** lines are **necessary** (never “unnecessary”).

**Window:** **Primary** period per row only—not comparison legs. **Hard 1:** **≥2** What lines and **every** primary ends **after** the What’s **end** (e.g. May 3–9 vs What ending May 5)—**window mismatch only** (figures exist but wrong window—**not** a figure data gap). Single leaf, one bucket slightly past end → **3**. Week leaf + unnecessary month row + window slip on primary → **3**.

**Parent–child math:** For the **same primary window**, named subs should **approximately sum** to the parent (rounding OK). Do **not** score math fails from comparing May MTD to a different April total.

**Taxonomy:** Meals → Dining Out, Delivered Food, Groceries; Leisure → Entertainment, Travel; Shopping → Pets, Clothing, Gadgets, Kids; Bills → Connectivity, Insurance, Tax, Service Fees; Shelter → Home, Utilities, Upkeep; Education → Kids Activities, Tuition; Transportation → Car, Public; Health → Medical and Pharmacy, Gym and Wellness, Personal Care; Income → Salary, Sidegig, Business, Interest; Donations and Gifts, Miscellaneous, Transfers. Omitting **unmentioned** children is **not** a gap.

**Hard outputs (mandatory):** (1) Wrong family / zero on-topic figure rows → **1**. (2) **≥2** What claims, all **primaries** end after What end → **1**. (3) Week ask, no week labels → **1**. (4) **One** category, week primary past What end + unnecessary month/parent row → **3**. (5) **Transaction gap** (rule 3) for any named claim → **≤3**. (6) **Leaf-only** What + **any** unnecessary sibling/parent **figure** row → **4** max (**never 5**). (7) Month/MTD parent + named subs + MoM, no gaps, no (6) → **5**.

**Scores (1–5):** **5** all named claims have connected figure data; no unnecessary rows; window OK. **4** leaf primary matches; only unnecessary extra **figure** rows (no window slip). **3** **figure data gap** for ≥1 named claim; **or** **single** leaf What with unnecessary month row **and** primary window slip. **2** thin on-topic. **1** wrong family; **≥2** What lines with all primaries past What end; week ask without week labels; amount clash.

**`notes`:** Semicolon-separated **data-only**: `figure data gap for X: no [grain] amounts pulled`; `no transactions pulled to support the $X [claim]` when Figures show the total but Drivers lack txn/merchant lines peers have; `unnecessary: …`; `window`; `contradiction`. **Banned:** “did not explain,” “no week amounts” when Figures match the stated total, “driver pivot,” “narrative.”

**Before JSON (order):** **STEP 1** multi-line window → **1**. **STEP 2** single leaf week past end → **3**. **STEP 3:** **≥2** claims, peers have dated merchant charges in Drivers, but one claim (e.g. **Groceries $90**) only restates the figure → **≤3**, note *no transactions pulled to support the $X [claim]*. **STEP 4:** What names **only one** category (e.g. `bills_service_fees` alone) and Figures include **other** sibling slugs (insurance, tax, connectivity, delivered_food, groceries) → **`score` = 4**. Else if all clean → **5**. Never “did not explain” or “no week amounts” when Figures show the total.

Return only JSON `{score, notes}`. STEP 1 → **1**; STEP 4 → **4** (never 5).
"""


TEST_CASES: list[dict[str, Any]] = [
  {
    "name": "groceries_spike_off_topic_travel_data_ideal_one",
    "batch": 1,
    "review_outcome": "",
    "output": "{\"score\": 1, \"notes\": \"Wrong category family (leisure_travel) vs meals_groceries What; no on-topic grocery figures or drivers; off-topic data only.\"}",
    "input": """# Rationalize What

Explain: **`meals_groceries`** is **much higher this April** than usual—what drove it? (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

*   **`leisure_travel` monthly:** $2,400.00 (Apr 1–30, 2026); $1,900.00 (Mar 1–31, 2026); $2,100.00 (Feb 1–28, 2026).

## Drivers

April **travel** is elevated versus prior months on **airfare and lodging** booked for the month; that is the main swing shown in these **leisure_travel** totals.

## Next steps

1. Review upcoming trips if you want to smooth travel spend into May.
""",
  },
  {
    "name": "groceries_weekly_what_monthly_and_parent_meals_noise_ideal_three",
    "batch": 2,
    "review_outcome": "",
    "output": "{\"score\": 3, \"notes\": \"April monthly groceries row is unnecessary coarser grain for a weekly-only leaf What; Total Meals parent roll-up is unnecessary parent zoom vs groceries-only ask; ISO week label spans May 3–May 9 vs What May 3–May 5 window mismatch on the weekly bucket.\"}",
    "input": """# Rationalize What

Explain: **`meals_groceries`** is **down this week** at **$79**—what explains it? **(2026-05-03 to 2026-05-05)**

# Rationalize Response

## Figures

*   **`meals_groceries` (ISO week of May 3, 2026):** $78.56 (May 3–May 9, 2026) vs. $112.01 (Apr 26–May 2) vs. $98.20 (Apr 19–Apr 25)
*   **`meals_groceries` (April 2026 month):** $610.00 (Apr 1–30, 2026); $540.00 (Mar 1–31, 2026) (monthly context for grocery trend)
*   **Total Meals (May 3–May 9, 2026):** $130.59 vs. $175.55 (Apr 26–May 2) (parent roll-up for context)

## Drivers

Groceries is softer this week versus the prior ISO week; the April monthly grocery line also shows you were elevated last month, and total Meals moved similarly week-over-week.

## Next steps

1. Meal-plan for the short week.
""",
  },
  {
    "name": "dining_out_leaf_with_sibling_noise_ideal_three",
    "batch": 3,
    "review_outcome": "",
    "output": "{\"score\": 4, \"notes\": \"Unnecessary figure rows: meals_delivered_food and meals_groceries not named in What; on-topic meals_dining_out figures pulled for the April claim.\"}",
    "input": """# Rationalize What

Explain: **`meals_dining_out`** is **much higher this April** than usual—what drove it? (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

*   **`meals_dining_out`:** $640.00 (Apr 1–30, 2026); $310.00 (Mar 1–31, 2026); $290.00 (Feb 1–28, 2026).
*   **`meals_delivered_food`:** $420.00 (Apr); $260.00 (Mar); $240.00 (Feb) (for context on overall food spend).
*   **`meals_groceries`:** $510.00 (Apr); $440.00 (Mar); $400.00 (Feb) (for context on overall food spend).

## Drivers

Dining out is up in April; delivered and groceries also rose, so **overall Meals** is higher month-over-month.

## Next steps

1. Set a monthly cap for dining out.
""",
  },
  {
    "name": "meals_subs_week_window_mismatch_ideal_one",
    "batch": 4,
    "review_outcome": "",
    "output": "{\"score\": 1, \"notes\": \"Focal buckets run through May 9 while the What ends May 5—window mismatch; Total Meals parent line bundles siblings vs per-sub claims in the What (unnecessary parent aggregation for the stated sub-lines).\"}",
    "input": """# Rationalize What

Explain: Groceries is significantly down this week at $79. Delivered Food is significantly up this week at $46. Dining Out is significantly down this week at $6. Food is thus significantly down this week to $131. (2026-05-03 to 2026-05-05)

# Rationalize Response

## Figures

*   **Total Meals Spend:** $130.59 (May 3–May 9) vs. $175.55 (Apr 26–May 2) vs. $133.55 (Apr 19–Apr 25)
*   **Groceries:** $78.56 (May 3–May 9) vs. $112.01 (Apr 26–May 2)
*   **Delivered Food:** $45.61 (May 3–May 9) vs. $29.77 (Apr 26–May 2)
*   **Dining Out:** $6.42 (May 3–May 9) vs. $33.77 (Apr 26–May 2)

## Drivers

The decrease in overall food spending is driven primarily by a $33.45 drop in **Groceries** and a $27.35 drop in **Dining Out** compared to the previous week.
While **Delivered Food** increased by $15.84, it was not enough to offset the reductions in the other categories, leading to a net decline in total weekly food spending.
The current spend of $130.59 is lower than last week's $175.55 and remains slightly below the levels seen in late April.

## Next steps

1. **Budget Adjustment:** Consider setting a weekly food budget of $145 based on your average spending over the last three weeks to help manage these shifts between delivery and groceries.
2. **Dining Out Goal:** If you would like to keep dining out spend low consistently, we can set a monthly goal of $120 for the `meals_dining_out` category.
3. **Recategorization:** Review recent transactions to ensure all grocery store visits are accurately tagged as `meals_groceries` and not mislabeled, as this helps maintain the accuracy of these category breakdowns.
""",
  },
  {
    "name": "shelter_parent_subs_mtd_zero_timing_ideal_five",
    "batch": 5,
    "review_outcome": "",
    "output": "{\"score\": 5, \"notes\": \"Parent Shelter figures match May 1–13; per-sub Utilities and Home supported with prior-month context and dated txn drivers explaining $0 MTD as timing.\"}",
    "input": """# Rationalize What

Explain: Utilities is significantly down this month at $0.
 Home is significantly down this month at $0.
 Shelter is thus significantly down this month to $0.
(2026-05-01 to 2026-05-13)

# Rationalize Response

## Figures

* **Shelter (Total):** $0 (May 1–13, 2026) vs $268.11 (April 2026) vs $4,248.49 (March 2026)
* **Shelter Utilities:** $0 (May 1–13, 2026) vs $240.36 (April 15, 2026)
* **Shelter Home:** $0 (May 1–13, 2026) vs $0 (April 2026) vs $3,895.57 (March 2026)

## Drivers

Your spending in **Shelter** is $0 for May, which is significantly down compared to April ($268.11) and March ($4,248.49).
This is driven primarily by the lack of any transactions posted so far this month for both **Shelter Utilities** and **Shelter Home**.
In April, your only shelter-related expense was a single **Utilities** payment, while March included both your rent/home payments and utility bills:

* **NGRID:** $240.36 (2026-04-15)
* **AVA Commons:** $2,887.62 (2026-03-09)
* **AVA Commons:** $1,007.95 (2026-03-05)

The significant drop to $0 in May is likely due to timing—your regular recurring housing and utility payments have simply not posted for this month yet.

## Next steps

1. **Verify billing cycles:** Check your next scheduled payment dates for your rent (AVA Commons) and utility provider (National Grid) to confirm they are expected later this month.
2. **Budget check:** Ensure you have enough funds allocated for your upcoming rent and utilities, as the total for the month will increase once these transactions post.
""",
  },
  {
    "name": "bills_parent_subs_april_anomaly_ideal_five",
    "batch": 6,
    "review_outcome": "",
    "output": "{\"score\": 5, \"notes\": \"Parent Bills figures match May 1–11; per-sub Connectivity and Service Fees supported with MoM context and drivers explaining parent drop vs April anomaly.\"}",
    "input": """# Rationalize What

Explain: Connectivity is significantly up this month at $249.
 Service Fees is significantly up this month at $131.
 Bills is thus significantly down this month to $380.
(2026-05-01 to 2026-05-11)

# Rationalize Response

## Figures

* **Total Bills (Category 'bills' + children):** $380 in May 1–11, 2026 vs. $9,434 in Apr 2026 vs. $375 in Mar 2026.
* **Connectivity (bills_connectivity):** $249 in May 1–11, 2026 vs. $227 in Apr 2026 vs. $162 in Mar 2026.
* **Service Fees (bills_service_fees):** $131 in May 1–11, 2026 vs. $315 in Apr 2026 vs. $213 in Mar 2026.

## Drivers

The "significant down" trend in Bills is due to a comparison against an anomalous April total ($9,434), which was skewed by a large $8,900+ spike that is not present in May.
Your current May spending of $380 is actually in line with typical monthly patterns seen in March ($375).
Connectivity is slightly higher this month primarily due to recurring AT&T and T-Mobile charges totaling $229.98 by May 7th.
Service Fees are driven by a $111 charge from "Smart Payment Solutions" on May 10th.

## Next steps

1. **Categorization Rule:** Create a rule to consistently track "Smart Payment Solutions" as `bills_service_fees`.
2. **Budgeting:** Set a monthly target of $400 for total `bills` to reflect your recent consistent spending levels.
3. **Review:** Examine the "Namecheap.com" ($229.96) charge from April if you believe it was miscategorized, as it was a significant driver of that month's elevated Service Fees.
""",
  },
  {
    "name": "service_fees_leaf_sibling_bills_noise_ideal_four",
    "batch": 7,
    "review_outcome": "",
    "output": "{\"score\": 4, \"notes\": \"On-topic Service Fees figures match the What; unnecessary sibling rows for Insurance, Taxes, and Connectivity do not help explain Service Fees.\"}",
    "input": """# Rationalize What

Explain: **`bills_service_fees`** is **significantly up this month** at **$186**. (2026-05-01 to 2026-05-15)

# Rationalize Response

## Figures

* **`bills_service_fees`:** $186.40 (May 1–15, 2026) vs. $94.20 (Apr 1–30, 2026) vs. $112.05 (Mar 1–31, 2026)
* **`bills_insurance`:** $420.00 (May 1–15, 2026) vs. $420.00 (Apr 1–30, 2026) vs. $415.00 (Mar 1–31, 2026)
* **`bills_tax`:** $0.00 (May 1–15, 2026) vs. $0.00 (Apr 1–30, 2026) vs. $1,240.00 (Mar 1–31, 2026)
* **`bills_connectivity`:** $249.18 (May 1–15, 2026) vs. $227.04 (Apr 1–30, 2026) vs. $162.33 (Mar 1–31, 2026)

## Drivers

Service Fees are elevated in May primarily due to two **Smart Payment Solutions** charges ($111.00 on May 10 and $75.40 on May 14), versus a quieter April with $94.20 total in the category.

## Next steps

1. Create a categorization rule for Smart Payment Solutions under `bills_service_fees`.
""",
  },
  {
    "name": "home_and_shelter_mom_context_ideal_five",
    "batch": 8,
    "review_outcome": "",
    "output": "{\"score\": 5, \"notes\": \"Home figures support the Home increase claim; Shelter parent totals with prior-month context support the Shelter increase claim; labels and MTD window match.\"}",
    "input": """# Rationalize What

Explain: **Home** is **significantly up this month** at **$3,200**.
 **Shelter** is thus **significantly up this month** to **$3,450**.
(2026-05-01 to 2026-05-15)

# Rationalize Response

## Figures

* **`shelter` (total):** $3,450.00 (May 1–15, 2026) vs. $2,680.00 (Apr 1–30, 2026) vs. $4,248.49 (Mar 1–31, 2026)
* **`shelter_home`:** $3,200.00 (May 1–15, 2026) vs. $2,450.00 (Apr 1–30, 2026) vs. $3,895.57 (Mar 1–31, 2026)
* **`shelter_utilities`:** $250.00 (May 1–15, 2026) vs. $230.00 (Apr 1–30, 2026) vs. $352.92 (Mar 1–31, 2026)

## Drivers

**Home** is up in May because your **AVA Commons** rent payment on May 5 was **$3,200.00**, higher than April’s **$2,450.00** single payment.
That increase in **Home** is the main driver of the rise in total **Shelter** to **$3,450** for the period, with **Utilities** roughly flat week-over-week in May.

## Next steps

1. Confirm whether the May rent amount reflects a lease change or a one-time adjustment.
""",
  },
  {
    "name": "meals_subs_week_groceries_figure_gap_ideal_three",
    "batch": 9,
    "review_outcome": "",
    "output": "{\"score\": 3, \"notes\": \"Figure data gap for Groceries: no transactions pulled to support the $90 claim; Dining Out and Delivered Food have on-topic weekly figures.\"}",
    "input": """# Rationalize What

Explain: **Dining Out** is **significantly up this week** at **$234**.
 **Delivered Food** is **significantly up this week** at **$100**.
 **Groceries** is **significantly up this week** at **$90**.
(2026-05-10 to 2026-05-16)

# Rationalize Response

## Figures

* **`meals_dining_out` (week of May 10, 2026):** $234.18 (May 10–16, 2026) vs. $98.40 (May 3–9, 2026) vs. $112.05 (Apr 26–May 2, 2026)
* **`meals_delivered_food` (week of May 10, 2026):** $100.12 (May 10–16, 2026) vs. $52.30 (May 3–9, 2026) vs. $61.88 (Apr 26–May 2, 2026)
* **`meals_groceries` (week of May 10, 2026):** $90.00 (May 10–16, 2026) vs. $50.00 (May 3–9, 2026) vs. $62.88 (Apr 26–May 2, 2026)

## Drivers

**Dining Out** increased this week largely because of a **$142.50** dinner at **Capital Grille** on May 12. **Delivered Food** rose with more frequent **DoorDash** and **Uber Eats** orders across the week (six deliveries vs. three last week). **Groceries** is higher than the previous week at **$90**.

## Next steps

1. Set a weekly cap for `meals_dining_out` if you want to smooth restaurant spikes.
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
