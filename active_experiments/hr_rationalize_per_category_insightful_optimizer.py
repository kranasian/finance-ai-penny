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

**Rubric:** Grade **only** **`## Drivers`**. Use **`# Rationalize What`** for **which categories** to explain; use **`## Figures`** as **factual truth** (amounts, periods, charges). Every cited charge must name the **transaction** (ledger line label—not necessarily a business name). **Rounding:** Drivers amounts may differ slightly from What/Figures—acceptable. **`## Next steps`** out of scope. **18 fixtures** in batches **1–5**; **`--batch N --check`**.

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
        "One sentence on **## Drivers only**: why for each required category, named transaction(s) (ledger line labels), clarity, and coverage. Do not critique Figures-table formatting."
      ),
    ),
  },
)


SYSTEM_PROMPT = """Grade **only** **`## Drivers`** for **insightfulness**.

## Sources of truth
- **`## Figures`** = **factual ground truth** for amounts, date windows, period-over-period comparisons, and which charges/transactions exist. When Figures and **`# Rationalize What`** disagree, **trust Figures** for grading Drivers.
- **`# Rationalize What`** = **scope only**: which categories must be explained. **Do not** treat What’s up/down/$0 headline or dollar amount as truth when **Figures** disagree—use Figures for direction, totals, and comparisons.
- **`## Next steps`** — out of scope.

**Amounts:** Drivers may cite rounded totals or line amounts that differ slightly from **What** or **Figures**—**acceptable**; do not penalize or mention rounding-only gaps in `notes`. Use Figures for **direction**, **which charges exist**, and **material** amount differences—not penny-perfect parity with Drivers.

## Out of scope (never affects score)
- **`## Figures`** table formatting, labels, or grain.
- **Rounding-only amount gaps** between Drivers and What/Figures.
- Whether the What headline “should” match Figures—do not mention What/Figures doc mismatches in `notes` unless Drivers repeat a wrong What direction or total **instead of** Figures-backed facts.
- **What vs Figures direction/amount conflicts** — **trust Figures**. Grade whether Drivers explain the **Figures** move for each required category, not whether they parrot a wrong What headline.

## In scope
For **each category listed in What**, judge whether **`## Drivers`** explains **why that move happened** vs a relevant prior period, using **Figures** for the underlying facts.

## Period windows
- What date lines may be a **subset** of the period in Figures. Follow **Figures** for comparison windows when they are clearer.
- Drivers may compare to **full prior months** or **matching subsets** when that supports the movement story.
- Score from the **weakest** required category when several are listed in What.

## Transaction naming
When Drivers cite a charge, refund, hold, or deposit, name the **transaction** as it appears on the ledger line (label/description)—not only an amount or generic fee/category wording.

**Transaction name ≠ business name.** Fee labels, loan payee names, interest lines, P2P memos, and similar ledger text all satisfy naming.

**Unnamed or generic references** (amount-only, category-only, vague fee type with no specific line) → **partial (~3) or weaker**, even when amounts and direction are right. **Amount-only with correct direction and window → ~3, not ~2**—reserve **~2** for skipped categories, denied moves, or timing-only/vague activity when Figures list specific charges.

## Insight in Drivers
Insightful **Drivers** explain **why** each required category moved via:
- **Period-over-period change** — why spend/income rose, fell, restarted, or stayed $0 vs a relevant prior week/month; or
- **Driver identification** — which specific transaction(s) caused the increase or decrease vs the prior period (new, larger, absent, or resumed lines).

**Each What category** must be explained **per Figures** (direction, amounts, named charges where cited).

**What/Figures conflict:** Grade on whether Drivers explain the **Figures** move with named transactions. **Do not** penalize Drivers for diverging from a wrong What headline. **Do** penalize Drivers that parrot What’s wrong direction when Figures show the opposite. A **5** is possible when Drivers correctly explain the Figures move despite conflicting What.

- **Up/down moves:** identify transaction(s) that **drove** the change vs the prior period—not merely list all focal-period lines.
- **Composition without drivers → ~3, never 4–5:** naming every focal charge (with transaction labels) but **not** stating which line(s) drove the delta vs the comparison period caps at **~3** even when all lines are named. Restating focal vs prior totals while listing focal lines is **not** driver identification unless Drivers tie specific line(s) to the delta (new, absent last period, larger than last period).
- **Amount-only or generic references → ~3:** category + amount + window but no transaction name; direction correct → **~3**, not **~2**.
- **$0 focal period:** If prior periods had activity, stating **no transactions / no new charges / no activity** in the focal window is **complete**—do not demand extra mechanisms. **Prior-period charges only → ~2:** naming prior-window transactions without explaining why the **focal** window is $0 does **not** satisfy a $0 What line—never score **4–5**.
- **Refunds in What:** treat as net credit aggregates; name refund/charge lines in Drivers when explaining the move.
- **Timing-only or vague activity → ~2, not 1:** generic timing language or vague refund/activity wording with correct direction but **no transaction names** when Figures list specific charges → **~2**. **1** only when Drivers purely echo headlines and dollar totals with **no** mechanism, contrast, or activity claim.
- **Prior-period narrative without focal explanation → ~2:** discussing only prior-window charges without explaining why the **focal** window is $0/up/down; cannot score **5**.
- **Denying a Figures move → ~2.** Following Figures when What is wrong is **not** denial—score on Figures insight.
- **Hollow tautology** (restate direction with no transaction names and no vs-prior contrast) → **1**, same as pure restatement.
- **Several What lines:** score from the **weakest**; one strong category does not excuse another with only restated totals → overall **≤3**.

## Scoring process
1. From **What**, list each category that must be explained.
2. From **Figures**, note amounts, windows, and charges for each category.
3. Read **only** Drivers: does each required category get a **why vs prior period** story with **named transactions** where charges are cited?
4. One integer **1–5** from holistic impact (weakest category + severity).

## Scores (integer 1–5)
- **5** — Every required category: clear **why**, every cited charge **named** (transaction label), with driver(s) vs prior period (or absence/pause/restart for $0). **Never 5** when focal lines are listed without identifying which drove the move.
- **4** — Strong; one category thin or one charge referenced by amount only.
- **3** — **Partial:** transactions listed but not which drove the move; amount-only/generic references with correct direction; one category thin in multi-line What.
- **2** — **Weak:** category skipped; denies the move; timing-only or vague activity/refunds with **no transaction names** when Figures list specific charges; prior-period-only narrative with no focal-window story.
- **Multi-line What:** one category strong and another only restates direction with no named lines → overall **3**, not 2.
- **1** — Drivers **only** echo What/Figures headlines and recite dollar amounts—**no** mechanism, contrast, or activity claim of any kind.

**Weakest-category cap:** Score from the weakest required category. Any up/down category with only composition lists or unnamed charges → overall **≤3**. **Focal skip** (prior-period charges only with no focal-window $0/up/down story) → **~2**, never **4–5**.

## `notes`
One sentence on **Drivers-only** insight: coverage of What categories, named transactions, movement explanation. Do not critique Figures formatting.

Return **only** JSON `{score, notes}`.
"""


TEST_CASES: list[dict[str, Any]] = [
  # --- Batch 1: week grain, contradictions, tautology, partial spike ---
  {
    "name": "rideshare_hold_reconciles_headline_up_move",
    "batch": 1,
    "output": (
      '{"score": 5, "notes": "Reconciles the What’s up-at-$95 headline with posted vs-prior-week '
      'figures and the cleared authorization so the reader sees why the move looks up."}'
    ),
    "input": """# Rationalize What

Explain: Rideshare is significantly up this week at $95. (2026-07-06 to 2026-07-12)

# Rationalize Response

## Figures

* **Rideshare posted (Jul 6–12, 2026):** $48.30 vs $72.15 (Jun 29–Jul 5, 2026).
* **Pending authorization cleared this week (same category):** $46.70 from Jul 5 that posted Jul 9.
* **Insight-style total (posted + cleared auth):** $95.00 vs $72.15 prior week → **up ~32%**.

## Drivers

Posted rides alone are **down** week-over-week, but the insight total counts an **Uber $46.70** authorization from Jul 5 that cleared Jul 9 toward this week—**$48.30 + $46.70 ≈ $95**, above last week’s **$72.15**. The “up” move is driven by that posting delay, not a sudden surge in new trips.

## Next steps

1. Set a weekly cap on `transportation_rideshare` if you want posted rides to stay under $60 regardless of pending auths.
""",
  },
  {
    "name": "household_what_up_figures_down_drivers_follow_figures",
    "batch": 1,
    "output": (
      '{"score": 5, "notes": "What claims up but Figures show a decrease; Drivers explain the '
      'down move with merchant-named charges vs prior week—insight follows Figures, not What."}'
    ),
    "input": """# Rationalize What

Explain: Household Supplies is significantly up this week at $89. (2026-07-06 to 2026-07-12)

# Rationalize Response

## Figures

* **Household Supplies (Jul 6–12, 2026):** $89.20 vs $156.40 (Jun 29–Jul 5, 2026) → **down ~43%**.
* **Jun 29–Jul 5:** Target **$98.00**; Amazon Fresh **$58.40**.
* **Jul 6–12:** Target **$42.20**; Dollar Tree **$47.00** (no Amazon Fresh).

## Drivers

**Household Supplies is significantly up this week at $89.**: Household supplies actually **fell** to **$89.20** from **$156.40** last week because **Amazon Fresh $58.40** did not repeat and **Target** was smaller (**$42.20** vs **$98.00**); **Dollar Tree $47.00** was new but not enough to offset the missing grocery delivery run.

## Next steps

1. Review household receipts when What and posted totals disagree.
""",
  },
  {
    "name": "subscriptions_down_single_charge_no_merchant_name",
    "batch": 1,
    "output": (
      '{"score": 3, "notes": "Cites a lower single charge and time window but never names '
      'which subscription merchant posted—partial insight without establishment names."}'
    ),
    "input": """# Rationalize What

Explain: Subscriptions is significantly down this week at $28. (2026-07-06 to 2026-07-12)

# Rationalize Response

## Figures

* **Subscriptions (Jul 6–12, 2026):** $28.00 vs $142.50 (Jun 29–Jul 5, 2026).
* **Top Jun 29–Jul 5 charges:** Adobe Creative Cloud **$54.99**; NYTimes **$17.00**; Audible annual **$70.51**.
* **Jul 6–12:** Craftsy **$16.99**; Spotify **$11.01** (no Adobe, NYTimes, or Audible this week).

## Drivers

Subscriptions is down after a single lower **$28** charge appeared during **Jul 6–12, 2026**, compared with heavier billing last week.

## Next steps

1. Tag Adobe → `bills_subscriptions` consistently.
""",
  },
  {
    "name": "phone_spike_overage_drives_week_up",
    "batch": 1,
    "output": (
      '{"score": 5, "notes": "Names T-Mobile international overage and AT&T as the transactions '
      'driving the weekly up move—the overage is framed as the cause of the jump."}'
    ),
    "input": """# Rationalize What

Explain: Phone and Mobile is significantly up this week at $142. (2026-07-06 to 2026-07-12)

# Rationalize Response

## Figures

* **Phone and Mobile (Jul 6–12, 2026):** $142.30 vs $52.80 (Jun 29–Jul 5, 2026) vs $55.10 (Jun 22–28, 2026).

## Drivers

Phone spend is higher this week at **$142.30** versus **$52.80** last week. The jump is from a **T-Mobile international roaming overage** (**$67.31** on Jul 8) plus your regular **AT&T family plan** (**$74.99** on Jul 10).

## Next steps

1. Set an alert on `bills_phone_mobile` if weekly spend exceeds $80.
""",
  },
  # --- Batch 2: $0 restart, pure restatement, thin timing ---
  {
    "name": "coffee_delivery_restart_after_subscription_pause",
    "batch": 2,
    "output": (
      '{"score": 5, "notes": "Uses $0 prior months and mid-month ISO buckets to explain why '
      'coffee-delivery spend restarted and why the move looks sharply up."}'
    ),
    "input": """# Rationalize What

Explain: Coffee Delivery is significantly up this month at $118. (2026-08-01 to 2026-08-31)

# Rationalize Response

## Figures

* **Coffee Delivery (Aug 1–31, 2026):** $118.40.
* **Coffee Delivery (Jul 1–31, 2026):** $0.00.
* **Coffee Delivery (Jun 1–30, 2026):** $0.00.
* **ISO week buckets (Aug):** $0.00 (Aug 1–7); $12.60 (Aug 8–14); $48.80 (Aug 15–21); $57.00 (Aug 22–28).

## Drivers

This is a **restart**, not a steady climb: July and June are **$0** because **Blue Bottle** and **Philz** delivery subscriptions were **canceled** during a spending reset. Orders resume mid-August—**Blue Bottle** (**$48.80** week of Aug 15) and **Philz** (**$57.00** week of Aug 22)—which is why August totals jump from a true **$0** baseline.

## Next steps

1. Tag Blue Bottle and Philz → `meals_coffee_delivery` so pauses stay visible.
""",
  },
  {
    "name": "cashback_restate_figures_only",
    "batch": 2,
    "output": (
      '{"score": 1, "notes": "Recites What and figure rows with no mechanism for why cashback '
      'income fell vs prior months."}'
    ),
    "input": """# Rationalize What

Explain: Cashback is significantly down this month at $8. (2026-08-01 to 2026-08-31)

# Rationalize Response

## Figures

* **Cashback (Aug 1–31, 2026):** $8.42.
* **Cashback (Jul 1–31, 2026):** $34.60.
* **Cashback (Jun 1–30, 2026):** $38.15.

## Drivers

Cashback is down to $8. July was $34.60 and June was $38.15.

## Next steps

1. Monitor cashback rewards.
""",
  },
  {
    "name": "home_maintenance_up_generic_timing",
    "batch": 2,
    "output": (
      '{"score": 2, "notes": "Cites vs-prior-month totals but only offers generic timing language '
      'without tying specific charges to why August rose vs July."}'
    ),
    "input": """# Rationalize What

Explain: Home Maintenance is significantly up this month at $540. (2026-08-01 to 2026-08-31)

# Rationalize Response

## Figures

* **Home Maintenance (Aug 1–31, 2026):** $540.20 vs $285.00 (Jul 1–31, 2026).
* **Largest Aug lines:** Home Depot **$248.00**; Ace Hardware **$162.00**; Plumber Co **$130.20**.

## Drivers

Home maintenance is higher in August mostly because of **timing**—some repair and supply expenses hit this month. It is up compared to July.

## Next steps

1. Set a monthly cap on `housing_home_maintenance`.
""",
  },
  # --- Batch 3: refunds, multi-line What, payroll, partial shopping ---
  {
    "name": "clothing_refunds_mechanism_clear",
    "batch": 3,
    "output": (
      '{"score": 5, "notes": "Links the What’s refund line to named merchants and explains how '
      'returns shape net clothing spend vs the prior month."}'
    ),
    "input": """# Rationalize What

Explain: Clothing net spend is elevated this month at $640, including **$185 in refunds** for Clothing that offset gross apparel charges. (2026-08-01 to 2026-08-31)

# Rationalize Response

## Figures

* **Clothing gross (Aug 1–31, 2026):** $825.00.
* **Clothing refunds/credits (Aug 1–31, 2026):** −$185.00 across 2 lines.
* **Clothing net (Aug 1–31, 2026):** $640.00 vs $520.00 net (Jul 1–31, 2026).

## Drivers

The **$185 refunds** are concentrated: **Nordstrom** return for unworn boots (**−$120.00** on Aug 6) and a **Zara** wrong-size reversal (**−$65.00** on Aug 21). Gross buys include **Uniqlo $320.00** and **H&M $285.00** in August; those returns explain why net is only **$120** above July’s **$520** despite **$825** in posted apparel charges.

## Next steps

1. Watch Zara sizing returns on `shopping_clothing`.
""",
  },
  {
    "name": "takeout_net_refunds_vague_no_prior_story",
    "batch": 3,
    "output": (
      '{"score": 2, "notes": "Mentions refunds and activity but does not explain why net takeout '
      'moved vs the prior month using the figures provided."}'
    ),
    "input": """# Rationalize What

Explain: Takeout net spend is elevated this month at $380, including **$95 in refunds** for Takeout that offset gross delivery charges. (2026-08-01 to 2026-08-31)

# Rationalize Response

## Figures

* **Takeout net (Aug 1–31, 2026):** $380.00 vs $342.00 (Jul 1–31, 2026).
* **Refunds (Aug):** −$95.00 total.

## Drivers

Takeout net is **$380** versus **$342** in July—you ordered delivery more often in August and refunds offset some of the gross total, so net still landed above July.

## Next steps

1. Set a monthly takeout budget.
""",
  },
  {
    "name": "salary_down_semimonthly_paycheck_count_explained",
    "batch": 3,
    "output": (
      '{"score": 5, "notes": "Explains the salary down move with paycheck timing and counts vs '
      'full prior months—clear period-over-period insight."}'
    ),
    "input": """# Rationalize What

Explain: Salary is significantly down this month at $3,250. (2026-08-01 to 2026-08-31)

# Rationalize Response

## Figures

* **Salary (Aug 1–31, 2026):** $3,250.00 vs $6,500.00 (Jul 1–31, 2026) vs $6,500.00 (Jun 1–30, 2026).
* **Paychecks posted:** Aug **1** deposit **$3,250.00** (Jul 31 pay date); Jul **2** deposits **$3,250.00** each (Jul 1 and Jul 15).

## Drivers

August salary is **half** of July/June because only **one semi-monthly paycheck** posted in August so far (**$3,250** on Jul 31), while July had **two** (**$6,500** total). The drop is **pay-cycle timing** at month-end, not a pay-rate cut.

## Next steps

1. Track expected pay dates against `income_salary` so mid-month dips are expected.
""",
  },
  {
    "name": "insurance_down_rent_thin_multi_line_what",
    "batch": 3,
    "output": (
      '{"score": 3, "notes": "Insurance vs July is well explained but the rent move is only '
      'named—overall insight is pulled down by the weakest What line."}'
    ),
    "input": """# Rationalize What

Explain: Insurance is significantly down this month at $145. Rent is thus slightly down this month to $2,400. (2026-08-01 to 2026-08-31)

# Rationalize Response

## Figures

* **Insurance (Aug 1–31, 2026):** $145.80 vs $268.40 (Jul 1–31, 2026).
* **Rent total (Aug 1–31, 2026):** $2,400.00 vs $2,418.40 (Jul 1–31, 2026).
* **Geico auto (insurance):** $98.00 Aug vs $186.40 Jul.

## Drivers

Insurance fell **$122.60** because **Geico** posted a lower six-month renewal (**$98** vs **$186** in July) after a safe-driver discount applied. **Rent** is slightly down this month.

## Next steps

1. Create a `shelter_insurance` budget at $160/month based on the last three months.
""",
  },
  # --- Batch 4: focal $0, focal skip, health partial ---
  {
    "name": "hotels_zero_no_august_bookings_explained",
    "batch": 4,
    "output": (
      '{"score": 5, "notes": "Explains focal-month $0 hotels by absence of new bookings against '
      'prior-month spend—full insight on the What move."}'
    ),
    "input": """# Rationalize What

Explain: Hotels is **$0** this month. (2026-08-01 to 2026-08-31)

# Rationalize Response

## Figures

* **Hotels (Aug 1–31, 2026):** $0.00.
* **Hotels (Jul 1–31, 2026):** $890.00 (Chicago conference + lodging).
* **Hotels (Jun 1–30, 2026):** $420.00 (weekend Marriott stay).

## Drivers

August is **$0** because July already captured Chicago conference lodging, and **no new hotel or resort charges** posted in August—unlike June/July, which show booking clusters. The category is quiet by **absence of new stays**, not missing data.

## Next steps

1. Add a placeholder `leisure_hotels` budget line if September travel is likely.
""",
  },
  {
    "name": "fitness_zero_drivers_rehash_prior_only",
    "batch": 4,
    "output": (
      '{"score": 2, "notes": "Narrates prior-month fitness spend while leaving the focal $0 month '
      'unexplained despite August showing $0 in Figures."}'
    ),
    "input": """# Rationalize What

Explain: Fitness is **$0** this month. (2026-08-01 to 2026-08-31)

# Rationalize Response

## Figures

* **Fitness (Aug 1–31, 2026):** $0.00.
* **Fitness (Jul 1–31, 2026):** $142.00 (Peloton + yoga studio).

## Drivers

July fitness spend was **Peloton $39.00** and **YogaWorks $103.00**—that is the recent workout pattern in your account.

## Next steps

1. Review July fitness receipts.
""",
  },
  {
    "name": "dental_up_lists_visits_without_prior_change",
    "batch": 4,
    "output": (
      '{"score": 3, "notes": "Lists August dental charges but does not identify which '
      'transactions drove the increase vs July—partial, not full insight."}'
    ),
    "input": """# Rationalize What

Explain: Dental is significantly up this month at $520. (2026-08-01 to 2026-08-31)

# Rationalize Response

## Figures

* **Dental (Aug 1–31, 2026):** $520.40 vs $85.00 (Jul 1–31, 2026).

## Drivers

Dental is up this month at **$520.40** compared with **$85.00** in July. August includes **Bright Smiles cleaning** (**$185.00**), **Root Canal Specialists** (**$245.40**), and a **Delta Dental copay** (**$90.00**) on Aug 14.

## Next steps

1. Set a monthly cap on `health_dental`.
""",
  },
  {
    "name": "childcare_up_lists_vendors_no_prior_story",
    "batch": 4,
    "output": (
      '{"score": 3, "notes": "Itemizes August childcare merchants but does not say which charges '
      'drove the up move vs July—composition without drivers is partial."}'
    ),
    "input": """# Rationalize What

Explain: Childcare is significantly up this month at $1,840. (2026-08-01 to 2026-08-31)

# Rationalize Response

## Figures

* **Childcare (Aug 1–31, 2026):** $1,840.00 vs $620.00 (Jul 1–31, 2026).

## Drivers

August childcare totals **$1,840.00**, mainly **Bright Horizons tuition $1,200.00** (Aug 1), **KinderCare camp $480.00** (Aug 12), and **babysitter Venmo $160.00**.

## Next steps

1. Budget `family_childcare` at $1,400/month unless camp weeks are planned.
""",
  },
  # --- Batch 5: multi-claim restatement, mixed complexity ---
  {
    "name": "income_dividends_freelance_multi_restate_only",
    "batch": 5,
    "output": (
      '{"score": 1, "notes": "Repeats all three What headlines and figure totals with no '
      'causal story for any move vs prior months."}'
    ),
    "input": """# Rationalize What

Explain: Salary is significantly down this month at $3,250. Dividends is significantly down this month at $45. Freelance income is significantly up this month at $1,120. (2026-08-01 to 2026-08-31)

# Rationalize Response

## Figures

* **Salary (Aug):** $3,250.00 vs $6,500.00 (Jul).
* **Dividends (Aug):** $45.20 vs $128.00 (Jul).
* **Freelance (Aug):** $1,120.00 vs $380.00 (Jul).

## Drivers

Salary is $3,250, dividends are $45, and freelance is $1,120 this month.

## Next steps

1. Review income categories.
""",
  },
  {
    "name": "groceries_up_sams_club_restock_clear",
    "batch": 5,
    "output": (
      '{"score": 5, "notes": "Explains the weekly up move with Sam’s Club vs prior week, Safeway '
      'contrast, and ISO timing—full movement insight."}'
    ),
    "input": """# Rationalize What

Explain: Groceries is significantly up this week at $428. (2026-07-20 to 2026-07-26)

# Rationalize Response

## Figures

* **Groceries (Jul 20–26, 2026):** $428.60 vs $245.30 (Jul 13–19, 2026).
* **ISO split:** $142.00 (Jul 20–22); $286.60 (Jul 23–26).
* **Largest lines:** Sam’s Club **$198.00** (Jul 23); Safeway **$96.40** (Jul 21); Whole Foods **$84.20** (Jul 25).

## Drivers

The week jumped to **$428.60** from **$245.30** because of a **Sam’s Club** restock (**$198**) in a week with **no warehouse trip last week**, plus a larger **Safeway** run (**$96** vs **$48** prior week). **Other stores** account for the remaining increase.

## Next steps

1. Set a weekly `meals_groceries` cap at $350 if you want to limit stock-up weeks.
""",
  },
  {
    "name": "hobbies_down_one_time_purchases_absent",
    "batch": 5,
    "output": (
      '{"score": 5, "notes": "Ties the down move to absent Michaels/REI one-timers vs last week while '
      'noting steady craft subscriptions—clear period-over-period insight."}'
    ),
    "input": """# Rationalize What

Explain: Hobbies is significantly down this week at $22. (2026-07-06 to 2026-07-12)

# Rationalize Response

## Figures

* **Hobbies (Jul 6–12, 2026):** $22.00 vs $168.50 (Jun 29–Jul 5, 2026).
* **Jun 29–Jul 5:** REI **$89.00**; Michaels **$62.51**; Craftsy **$16.99**.
* **Jul 6–12:** Craftsy **$16.99** only (no REI/Michaels).

## Drivers

Hobbies fell **$146.50** because last week included a **REI** gear purchase (**$89.00**) and a **Michaels** supply run (**$62.51**), while this week has **no similar one-time purchases**—only recurring craft streaming. Subscriptions were **about the same** as usual.

## Next steps

1. Tag REI and Michaels → `leisure_hobbies` for cleaner week-over-week reads.
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

