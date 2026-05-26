"""
Rationalize rubric checker optimizer: Insightful.

Use this to iterate on the system_prompt that will be stored in `penny_templates`
for the checker template `Chk:RationalizePerCategoryInsightful`.

Run from `finance-ai-penny` repo root:

  python3 active_experiments/hr_rationalize_per_category_insightful_optimizer.py --test all
  python3 active_experiments/hr_rationalize_per_category_insightful_optimizer.py --batch 1 --check
  python3 active_experiments/hr_rationalize_per_category_insightful_optimizer.py --test all --model gemini-flash-lite-latest

**Recommended minimal generation settings** (re-validate `--test all`; scores match `ideal_response` **5 / 1**):

- **model:** `gemini-flash-lite-latest`
- **temperature:** `0` · **top_p:** `0.95`
- **thinking_budget:** `256` (internal reasoning; `include_thoughts=False` so JSON output stays reliable)
- **max_output_tokens:** `384`
- **response:** `application/json` + **response_schema** for `{score, notes}`

**Rubric:** Grade **only** **`## Drivers`** vs **`# Rationalize What`** (movement insight, clarity, coverage). **`## Figures`** and **`## Next steps`** are **out of scope**—never score figure tables, labels, or next steps. **What** is ground truth; **What vs Response number mismatches are out of scope**. **19 fixtures** in batches **1–5**; **`--batch N --check`** (batches **1–4** for iteration).

**Input:** a single markdown **`str`**—`# Rationalize What` then `# Rationalize Response` (same shape as `ai_agent_outcomes.agent_outcome` / comprehensive optimizer).
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
        "One sentence on **## Drivers only**: why each What move changed vs a relevant prior period or trend, clarity, and coverage. Do not cite Figures-table quality or What-vs-Response amount mismatches."
      ),
    ),
  },
)


SYSTEM_PROMPT = """Grade **only** **`## Drivers`** for **insightfulness** vs **`# Rationalize What`**.

## Out of scope (never affects score)
- **`## Figures`** — do not grade figure tables, labels, grain, or whether figures match the What.
- **`## Next steps`**
- **What vs Response data discrepancies** — different amounts, windows, or totals between What and Figures/Drivers are **not** rubric issues. Do not mention or penalize them in `notes`.
- Whether the What “should” differ from underlying data

## In scope
**`# Rationalize What`** = ground truth for each move (category, direction, amount, dates). Judge whether **`## Drivers`** explains **why each move happened** vs a **relevant comparison**: immediate prior week/month **or** a **multi-period trend / baseline** across past periods.

## Period windows (What dates vs comparisons)
- Dates on the What line may be a **subset** of the period described (e.g. “this month” with data through May 19). That is valid.
- Drivers may compare to **full prior months** (Apr 1–30) **or** **matching subsets** (Apr 1–19) when that supports the movement story. Do not penalize use of full-month priors when the What uses a partial month.
- Score from the **weakest** What move when several are listed.

## Relevant comparison (prior period **or** trend)
- **Immediate prior period** (last week, last month, matching partial-month slice) is a common anchor when it best explains the What.
- **Multi-period trend / baseline** is equally valid when Drivers cite **several past periods** or a clear baseline (e.g. “up vs late 2025,” “elevated over the last 6 months”) **with evidence** (totals, merchants, dates) to explain the What’s up/down/flat direction.
- Do **not** mark down Drivers for emphasizing a **trend** story when it **does** explain the What’s stated direction. Do **not** require a vs-**immediate-prior-month** explanation if trend + mechanisms already account for the move.
- Optional contrast to the prior month (e.g. “down vs April 1–25”) **strengthens** insight but is **not required** when trend + merchants explain the What.

## Insight = movement in Drivers (not composition alone)
Insightful **Drivers** explain **why spend/income changed** vs a relevant comparison—prior period **or** trend across past periods—not only how the focal total was built.

**Each What line:** Drivers must explain **that line’s stated direction** (up/down/flat/$0) using a **relevant comparison** (prior period **or** cited trend/baseline)—not only what merchants/charges make up the focal total.

- **Naming focal merchants without any movement story** (no vs-prior, no trend, no baseline) → partial (~3), not 5. Listing this week’s charges (or May MTD composition) **without** why the pattern **changed** vs prior period **or** trend is **not** fully insightful. Contrast: explaining a cleared hold vs prior-week posted **is** movement insight (can be 5); explaining “slightly up” via recurring rent + higher misc volume vs a **late-2025 baseline** **is** movement insight (can be 5).
- **$0 focal period:** If the What move is **$0** and prior periods had activity, stating **no transactions / no new charges / no activity** in the focal window is a **complete** explanation—do not demand extra mechanisms.
- **Refunds in the What:** treat as net credit aggregates; name mechanisms in Drivers when stated.
- **Prior-period narrative alone** in Drivers (e.g. last month’s trip/gym charges) **without** why the **focal** window is $0/up/down → **weak (~2)** for that move; cannot score **5**.
- **Denying the What in Drivers** (e.g. “not really down”) → **weak insight (~2)**; reserve **1** for no causal story at all.
- **Hollow tautology** in Drivers (“up because you spent more”) with no vs-prior contrast → **~2**; **1** only when Drivers **only** restate What/amounts with **no** “because” attempt.
- **Several What lines:** score from the **weakest**; a strong explanation for one category (e.g. Taxes) does **not** excuse another (e.g. Service Fees “**down** at $117”) where Drivers **only** name May interest charges with **no why lower than** the prior month → that line is **partial (~3)**; overall **≤3**.

## Scoring process
1. Parse each What move and its date window.
2. Read **only** Drivers: does each move get a **why** story vs prior period **or** cited trend/baseline?
3. One integer **1–5** from holistic impact (weakest move + severity). No fixed mapping from issue labels to scores.

## Scores (integer 1–5)
- **5** — Every What move: clear **why-it-changed** in Drivers (merchants/dates, pause/restart, absence of activity for $0, etc.).
- **4** — Strong; one move slightly thin or vague in Drivers.
- **3** — **Partial:** Drivers **name focal-period merchants/charges** (or recite totals) for a move but **not why that move’s direction changed** vs prior period **or** trend—**not** a 5. *Naming merchants without any movement story (no vs-prior, no trend) → ~3.*
- **2** — **Weak:** **focal move skipped** (only prior-period merchants, never why focal $0/up/down); **denies What**; **circular “because”** with no vs-prior story; **generic timing/vague** when Drivers could cite specific charges; reciting prior totals with a hollow “because.”
- **1** — Drivers **only** echo What headlines and **recite dollar amounts** across periods—**no** merchants, timing, pause, absence, or real mechanism (e.g. “down to $12; May was $48, April was $52” only).

**Weakest-move cap:** Score from the weakest move. If **any** move is **partial** (focal merchants/charges named, no vs-prior **or** trend **direction** story) → overall **≤3** (e.g. Connectivity up: Verizon overage + Xfinity listed, no why overage missing before). **Focal skip** (prior-period merchants only, never why focal $0/up/down) is **weak (~2)**, not partial. Trend + merchants explaining the What’s direction is **not** partial.

## `notes`
One sentence on **Drivers-only** insight: movement explanation, clarity, coverage. Never critique Figures or What/Response mismatches.

Return **only** JSON `{score, notes}`.
"""


TEST_CASES: list[dict[str, Any]] = [
  # --- Batch 1: week grain, contradictions, tautology, partial spike ---
  {
    "name": "dining_hold_reconciles_headline_up_move",
    "batch": 1,
    "output": (
      '{"score": 5, "notes": "Reconciles the What’s up-at-$180 headline with posted vs-prior-week '
      'figures and the cleared hold so the reader sees why the move looks up."}'
    ),
    "input": """# Rationalize What

Explain: Dining Out is significantly up this week at $180. (2026-05-03 to 2026-05-09)

# Rationalize Response

## Figures

* **Dining Out posted (May 3–9, 2026):** $94.80 vs $139.55 (Apr 26–May 2, 2026).
* **Pending hold cleared this week (same category):** $85.20 authorization from May 2 that posted May 6.
* **Insight-style total (posted + cleared hold):** $180.00 vs $139.55 prior week → **up ~29%**.

## Drivers

Posted dining alone is **down** week-over-week, but the insight total treats the **$85.20** hold that cleared May 6 as part of this week’s dining story—**$94.80 + $85.20 ≈ $180**, which is higher than last week’s **$139.55**. The “up” move is driven by that timing artifact, not a broad restaurant surge.

## Next steps

1. Set a weekly cap on `meals_dining_out` if you want posted dining to stay under $120 regardless of holds.
""",
  },
  {
    "name": "groceries_up_circular_no_prior_week_story",
    "batch": 1,
    "output": (
      '{"score": 2, "notes": "Affirms the What’s up direction with a hollow because-clause and never '
      'explains why grocery spend rose vs the prior week shown in Figures."}'
    ),
    "input": """# Rationalize What

Explain: Groceries is significantly up this week at $412. (2026-05-10 to 2026-05-16)

# Rationalize Response

## Figures

* **Groceries (May 10–16, 2026):** $412.30 vs $286.15 (May 3–9, 2026).

## Drivers

Groceries are up this week at about $412 because you bought more groceries.

## Next steps

1. Review grocery receipts.
""",
  },
  {
    "name": "entertainment_denies_what_down_move",
    "batch": 1,
    "output": (
      '{"score": 2, "notes": "Drivers deny the What’s down move instead of explaining why '
      'entertainment fell vs last week using the provided figures."}'
    ),
    "input": """# Rationalize What

Explain: Entertainment is significantly down this week at $45. (2026-05-10 to 2026-05-16)

# Rationalize Response

## Figures

* **Entertainment (May 10–16, 2026):** $45.00 vs $198.40 (May 3–9, 2026).
* **Top May 3–9 charges:** AMC **$62.00**; Spotify **$11.99**; Steam **$124.41**.

## Drivers

Entertainment is **not really down**—you still had steady subscriptions, and the AMC ticket was just posted late. The category is effectively flat if you ignore window boundaries.

## Next steps

1. Tag AMC → `leisure_entertainment` consistently.
""",
  },
  {
    "name": "connectivity_spike_lists_charges_missing_prior_pattern",
    "batch": 1,
    "output": (
      '{"score": 3, "notes": "Compares weeks and itemizes focal charges but leaves unexplained '
      'why the overage pattern was absent in earlier weeks—partial movement insight."}'
    ),
    "input": """# Rationalize What

Explain: Connectivity is significantly up this week at $189. (2026-05-10 to 2026-05-16)

# Rationalize Response

## Figures

* **Connectivity (May 10–16, 2026):** $189.40 vs $78.20 (May 3–9, 2026) vs $81.05 (Apr 26–May 2, 2026).

## Drivers

Connectivity is higher this week at **$189.40** versus **$78.20** last week. The jump is from a **Verizon wireless overage** (**$95.00** on May 12) plus your usual **Xfinity internet** (**$74.99** on May 14).

## Next steps

1. Set an alert on `bills_connectivity` if weekly spend exceeds $100.
""",
  },
  # --- Batch 2: $0 restart, pure restatement, thin timing ---
  {
    "name": "delivered_food_restart_after_subscription_pause",
    "batch": 2,
    "output": (
      '{"score": 5, "notes": "Uses $0 prior months and mid-month ISO buckets to explain why '
      'delivered-food spend restarted and why the move looks sharply up."}'
    ),
    "input": """# Rationalize What

Explain: Delivered Food is significantly up this month at $214. (2026-06-01 to 2026-06-30)

# Rationalize Response

## Figures

* **Delivered Food (Jun 1–30, 2026):** $214.60.
* **Delivered Food (May 1–31, 2026):** $0.00.
* **Delivered Food (Apr 1–30, 2026):** $0.00.
* **ISO week buckets (Jun):** $0.00 (Jun 1–7); $18.40 (Jun 8–14); $96.20 (Jun 15–21); $100.00 (Jun 22–28).

## Drivers

This is a **restart**, not a steady climb: May and April are **$0** because Instacart+ and DoorDash DashPass were **paused** after a budget reset. Charges resume mid-June—**Instacart** (**$96.20** week of Jun 15) and **DoorDash** (**$100.00** week of Jun 22)—which is why June totals jump from a true **$0** baseline.

## Next steps

1. Tag Instacart and DoorDash → `meals_delivered_food` so pauses stay visible.
""",
  },
  {
    "name": "interest_income_restate_figures_only",
    "batch": 2,
    "output": (
      '{"score": 1, "notes": "Recites What and figure rows with no mechanism for why interest '
      'income fell vs prior months."}'
    ),
    "input": """# Rationalize What

Explain: Interest is significantly down this month at $12. (2026-06-01 to 2026-06-30)

# Rationalize Response

## Figures

* **Interest (Jun 1–30, 2026):** $12.18.
* **Interest (May 1–31, 2026):** $48.90.
* **Interest (Apr 1–30, 2026):** $52.10.

## Drivers

Interest is down to $12. May was $48.90 and April was $52.10.

## Next steps

1. Monitor interest income.
""",
  },
  {
    "name": "transportation_car_up_generic_timing",
    "batch": 2,
    "output": (
      '{"score": 2, "notes": "Cites vs-prior-month totals but only offers generic timing language '
      'without tying specific charges to why June rose vs May."}'
    ),
    "input": """# Rationalize What

Explain: Car and Transportation is significantly up this month at $680. (2026-06-01 to 2026-06-30)

# Rationalize Response

## Figures

* **Car and Transportation (Jun 1–30, 2026):** $680.40 vs $412.00 (May 1–31, 2026).
* **Largest Jun lines:** Shell **$186.00**; Jiffy Lube **$214.00**; City Parking **$88.00**.

## Drivers

Transportation is higher in June mostly because of **timing**—some car expenses hit this month. It is up compared to May.

## Next steps

1. Set a monthly cap on `transportation_car`.
""",
  },
  # --- Batch 3: refunds, multi-line What, payroll, partial shopping ---
  {
    "name": "shopping_gadgets_refunds_mechanism_clear",
    "batch": 3,
    "output": (
      '{"score": 5, "notes": "Links the What’s refund line to named merchants and explains how '
      'returns shape net gadget spend vs the prior month."}'
    ),
    "input": """# Rationalize What

Explain: Gadgets net spend is elevated this month at $920, including **$310 in refunds** for Gadgets that offset gross electronics charges. (2026-06-01 to 2026-06-30)

# Rationalize Response

## Figures

* **Gadgets gross (Jun 1–30, 2026):** $1,230.00.
* **Gadgets refunds/credits (Jun 1–30, 2026):** −$310.00 across 3 lines.
* **Gadgets net (Jun 1–30, 2026):** $920.00 vs $740.00 net (May 1–31, 2026).

## Drivers

The **$310 refunds** are concentrated: **Best Buy** return for an unopened monitor (**−$220.00** on Jun 8) and an **Amazon** duplicate-charge reversal (**−$90.00** on Jun 19). Gross looks like a spending spike, but returns explain why net is only **$180** above May despite the **$1,230** in posted buys.

## Next steps

1. Watch Amazon duplicate charges on `shopping_gadgets`.
""",
  },
  {
    "name": "dining_net_refunds_vague_no_march_story",
    "batch": 3,
    "output": (
      '{"score": 2, "notes": "Mentions refunds and activity but does not explain why net dining '
      'moved vs the prior month using the figures provided."}'
    ),
    "input": """# Rationalize What

Explain: Dining Out net spend is elevated this month at $510, including **$180 in refunds** for Dining Out that offset gross restaurant charges. (2026-06-01 to 2026-06-30)

# Rationalize Response

## Figures

* **Dining Out net (Jun 1–30, 2026):** $510.00 vs $455.00 (May 1–31, 2026).
* **Refunds (Jun):** −$180.00 total.

## Drivers

You dined out more in June and got some money back from refunds, so net landed at $510.

## Next steps

1. Set a monthly dining budget.
""",
  },
  {
    "name": "salary_down_biweekly_paycheck_count_explained",
    "batch": 3,
    "output": (
      '{"score": 5, "notes": "Explains the salary down move with paycheck timing and counts vs '
      'full prior months—clear period-over-period insight."}'
    ),
    "input": """# Rationalize What

Explain: Salary is significantly down this month at $2,140. (2026-06-01 to 2026-06-30)

# Rationalize Response

## Figures

* **Salary (Jun 1–30, 2026):** $2,140.00 vs $4,280.00 (May 1–31, 2026) vs $4,280.00 (Apr 1–30, 2026).
* **Paychecks posted:** Jun **1** deposit **$2,140.00** (May 30 pay date); May **2** deposits **$2,140.00** each (May 2 and May 16).

## Drivers

June salary is **half** of May/April because only **one bi-weekly paycheck** posted in June so far (**$2,140** on May 30), while May had **two** (**$4,280** total). The drop is **pay-cycle timing** at month-end, not a pay-rate cut.

## Next steps

1. Track expected pay dates against `income_salary` so mid-month dips are expected.
""",
  },
  {
    "name": "utilities_down_shelter_thin_multi_line_what",
    "batch": 3,
    "output": (
      '{"score": 3, "notes": "Utilities vs May is well explained but the shelter move is only '
      'named—overall insight is pulled down by the weakest What line."}'
    ),
    "input": """# Rationalize What

Explain: Utilities is significantly down this month at $210. Shelter is thus slightly down this month to $3,050. (2026-06-01 to 2026-06-30)

# Rationalize Response

## Figures

* **Utilities (Jun 1–30, 2026):** $210.40 vs $338.90 (May 1–31, 2026).
* **Shelter total (Jun 1–30, 2026):** $3,050.00 vs $3,088.90 (May 1–31, 2026).
* **Dominion Energy (utilities):** $118.00 Jun vs $226.40 May.

## Drivers

Utilities fell **$128.50** because **Dominion** summer billing posted lower usage (**$118** vs **$226** in May) after an estimated-read correction in May inflated that month. **Shelter** is slightly down this month.

## Next steps

1. Create a `shelter_utilities` budget at $230/month based on the last three months.
""",
  },
  # --- Batch 4: focal $0, focal skip, health partial ---
  {
    "name": "travel_zero_no_june_bookings_explained",
    "batch": 4,
    "output": (
      '{"score": 5, "notes": "Explains focal-month $0 travel by absence of new bookings against '
      'prior-month spend—full insight on the What move."}'
    ),
    "input": """# Rationalize What

Explain: Travel and Vacations is **$0** this month. (2026-06-01 to 2026-06-30)

# Rationalize Response

## Figures

* **Travel and Vacations (Jun 1–30, 2026):** $0.00.
* **Travel and Vacations (May 1–31, 2026):** $1,120.00 (Denver flights + hotel).
* **Travel and Vacations (Apr 1–30, 2026):** $640.00 (weekend rail + Airbnb).

## Drivers

June is **$0** because May already captured Denver airfare and lodging, and **no new flight, hotel, or Airbnb charges** posted in June—unlike April/May, which show booking clusters. The category is quiet by **absence of new trips**, not missing data.

## Next steps

1. Add a placeholder `leisure_travel` budget line if July travel is likely.
""",
  },
  {
    "name": "gym_wellness_zero_drivers_rehash_may_only",
    "batch": 4,
    "output": (
      '{"score": 2, "notes": "Narrates prior-month gym spend while leaving the focal $0 month '
      'unexplained despite June showing $0 in Figures."}'
    ),
    "input": """# Rationalize What

Explain: Gym and Wellness is **$0** this month. (2026-06-01 to 2026-06-30)

# Rationalize Response

## Figures

* **Gym and Wellness (Jun 1–30, 2026):** $0.00.
* **Gym and Wellness (May 1–31, 2026):** $156.00 (Equinox + yoga studio).

## Drivers

May gym spend was **Equinox $120.00** and **CorePower $36.00**—that is the recent wellness pattern in your account.

## Next steps

1. Review May gym receipts.
""",
  },
  {
    "name": "medical_pharmacy_up_lists_rx_without_prior_change",
    "batch": 4,
    "output": (
      '{"score": 3, "notes": "Itemizes June pharmacy charges and vs-May comparison but does not '
      'explain why the prescription pattern differed from May—partial movement insight."}'
    ),
    "input": """# Rationalize What

Explain: Medical and Pharmacy is significantly up this month at $385. (2026-06-01 to 2026-06-30)

# Rationalize Response

## Figures

* **Medical and Pharmacy (Jun 1–30, 2026):** $385.60 vs $92.40 (May 1–31, 2026).

## Drivers

Pharmacy is up this month at **$385.60** compared with **$92.40** in May. June includes **CVS** (**$142.00**), **Walgreens** (**$118.60**), and a **Mail Order Rx** (**$125.00**) on Jun 4.

## Next steps

1. Set a monthly cap on `health_medical_pharmacy`.
""",
  },
  {
    "name": "pets_up_new_vendor_no_prior_absence_story",
    "batch": 4,
    "output": (
      '{"score": 3, "notes": "Shows vs-prior-month increase and names focal pet charges but not '
      'why similar spend was missing in May—reader gets composition more than movement cause."}'
    ),
    "input": """# Rationalize What

Explain: Pets is significantly up this month at $240. (2026-06-01 to 2026-06-30)

# Rationalize Response

## Figures

* **Pets (Jun 1–30, 2026):** $240.80 vs $38.00 (May 1–31, 2026).

## Drivers

June pet spend totals **$240.80**, mainly **Banfield vet visit $165.00** (Jun 12), **Chewy $58.80** (Jun 18), and **Petco food $17.00**.

## Next steps

1. Budget `shopping_pets` at $120/month unless vet visits are planned.
""",
  },
  # --- Batch 5: multi-claim restatement, mixed complexity ---
  {
    "name": "income_interest_sidegig_multi_restate_only",
    "batch": 5,
    "output": (
      '{"score": 1, "notes": "Repeats all three What headlines and figure totals with no '
      'causal story for any move vs prior months."}'
    ),
    "input": """# Rationalize What

Explain: Salary is significantly down this month at $2,140. Interest is significantly down this month at $12. Side-Gig income is significantly up this month at $890. (2026-06-01 to 2026-06-30)

# Rationalize Response

## Figures

* **Salary (Jun):** $2,140.00 vs $4,280.00 (May).
* **Interest (Jun):** $12.18 vs $48.90 (May).
* **Side-Gig (Jun):** $890.00 vs $310.00 (May).

## Drivers

Salary is $2,140, interest is $12, and side-gig is $890 this month.

## Next steps

1. Review income categories.
""",
  },
  {
    "name": "groceries_up_warehouse_restart_clear",
    "batch": 5,
    "output": (
      '{"score": 5, "notes": "Explains the weekly up move with Costco vs prior week, Kroger '
      'contrast, and ISO timing—full movement insight."}'
    ),
    "input": """# Rationalize What

Explain: Groceries is significantly up this week at $512. (2026-05-17 to 2026-05-23)

# Rationalize Response

## Figures

* **Groceries (May 17–23, 2026):** $512.40 vs $298.10 (May 10–16, 2026).
* **ISO split:** $180.00 (May 17–19); $332.40 (May 20–23).
* **Largest lines:** Costco **$214.00** (May 20); Kroger **$118.40** (May 18); Trader Joe’s **$96.00** (May 22).

## Drivers

The week jumped to **$512.40** from **$298.10** because of a **Costco** restock (**$214**) in a week with **no warehouse trip last week**, plus a larger **Kroger** run (**$118** vs **$62** prior week). **Other stores** account for the remaining increase.

## Next steps

1. Set a weekly `meals_groceries` cap at $400 if you want to limit stock-up weeks.
""",
  },
  {
    "name": "entertainment_down_one_time_purchases_absent",
    "batch": 5,
    "output": (
      '{"score": 5, "notes": "Ties the down move to absent AMC/Steam one-timers vs last week while '
      'noting steady streaming—clear period-over-period insight."}'
    ),
    "input": """# Rationalize What

Explain: Entertainment is significantly down this week at $45. (2026-05-10 to 2026-05-16)

# Rationalize Response

## Figures

* **Entertainment (May 10–16, 2026):** $45.00 vs $198.40 (May 3–9, 2026).
* **May 3–9:** AMC **$62.00**; Steam **$124.41**; Spotify **$11.99**.
* **May 10–16:** Spotify **$11.99** only (no AMC/Steam).

## Drivers

Entertainment fell **$153.40** because last week included a **Steam** purchase (**$124.41**) and an **AMC** ticket (**$62**), while this week has **no similar one-time purchases**—only recurring streaming. Subscriptions were **about the same** as usual.

## Next steps

1. Tag AMC and Steam → `leisure_entertainment` for cleaner week-over-week reads.
""",
  },
  {
    "name": "uncategorized_slightly_up_trend_not_only_prior_month",
    "batch": 5,
    "output": (
      '{"score": 5, "notes": "Explains slightly up via recurring merchant, ATM/Amazon mix, '
      'and upward trend vs late-2025 baseline—valid without requiring vs prior month alone."}'
    ),
    "input": """# Rationalize What

Explain: Uncategorized is slightly up the first 25 days of this month at $3390. (partial month May 1-31, 2026)

Category Taxonomy: parent Uncategorized (uncategorized)

# Rationalize Response

## Figures

* **Uncategorized**
    * May 2026: from May 1-25, $3390 · entire month $3390
    * Apr 2026: from Apr 1-25, $3447 · entire month $3526
    * Mar 2026: from Mar 1-25, $3203 · entire month $3249
    * Feb 2026: from Feb 1-25, $2999 · entire month $2999
    * Jan 2026: from Jan 1-25, $3390 · entire month $3043
    * Dec 2025: from Dec 1-25, $2915 · entire month $2915
    * Nov 2025: from Nov 1-25, $2752 · entire month $2803
    * Oct 2025: from Oct 1-25, $2839 · entire month $3014
    * Sep 2025: from Sep 1-25, $3029 · entire month $3029

## Drivers

* **Uncategorized** is slightly up for the year despite being down slightly compared to the first 25 days of April ($3,447); the category is dominated by a recurring $2,850 payment to "Property Group LLC" on the 3rd of every month, while the remaining balance reflects a higher volume of small miscellaneous charges like "ATM Withdrawals" ($420 total in May) and Amazon purchases, which account for the upward trend in overall category spending compared to the baseline established in late 2025.

## Next steps

* **Recategorize:** Review the transaction for "Property Group LLC" ($2,850); if this is rent or a mortgage, recategorize it under `shelter_home`.
* **Budget:** Set a monthly budget of $300 for `uncategorized` to track miscellaneous spending while excluding identified recurring fixed costs.
* **Audit:** Review all ATM withdrawals and Amazon transactions currently sitting in `uncategorized` and move them to more appropriate categories like `leisure_entertainment` or `shopping_gadgets`.
""",
  },
]


class CheckerOptimizer:
  def __init__(
    self,
    model_name: str = "gemini-flash-lite-latest",
    *,
    max_output_tokens: int = 384,
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
  parser.add_argument("--max-output-tokens", type=int, default=384)
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

