"""
Rationalize rubric checker optimizer: **Actionable** — score **every** **`## Next steps`** line on **appropriateness**, **concreteness**, and **relevance** given **`## Figures`** and **`## Drivers`**.

Directional goals: **higher income**, **lower spending**, **fewer uncategorized transactions**. Budget trend/directive rules and categorization fit are part of these three checks—not a separate pass for “budget-only” lines. **Incomplete inputs are OK** (amounts, slugs may be omitted).

Use this to iterate on the system_prompt stored in `penny_templates` for `Chk:RationalizePerCategoryActionable`.

Run from `finance-ai-penny` repo root:

  python3 active_experiments/hr_rationalize_per_category_actionable_optimizer.py --test all
  python3 active_experiments/hr_rationalize_per_category_actionable_optimizer.py --batch 1 --check
  python3 active_experiments/hr_rationalize_per_category_actionable_optimizer.py --test all --model gemini-flash-lite-latest

**Recommended generation settings** (re-validate `--test all --check` after prompt edits):

- **model:** `gemini-flash-lite-latest`
- **temperature:** `0` · **top_p:** `0.95`
- **thinking_budget:** `0` (raise to `256` if scores drift; keep `include_thoughts=False`)
- **max_output_tokens:** `384`
- **response:** `application/json` + **response_schema** for `{score, notes}`

**Fixtures:** **4 batches** (`--batch 1|2|3|4 --check`). **Input:** markdown with `# Rationalize What` and `# Rationalize Response`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
      description="1-5; weakest Next-steps line × worst of appropriateness/concreteness/relevance.",
    ),
    "notes": types.Schema(
      type=types.Type.STRING,
      description="One sentence: decisive flaw (appropriateness, concreteness, relevance, or empty). Never cite missing budget $ or slug.",
    ),
  },
)


SYSTEM_PROMPT = """Grade **written `## Next steps` lines only** (never infer). **Figures** = truth; **Drivers** = why / miscoding; **Rationalize What** = focal scope. Do not grade Figures/Drivers quality.

**FIRST (hard):** If `## Next steps` contains **zero** bullet/numbered lines (section blank), return **exactly**:\n`{\"score\": 1, \"notes\": \"No next-step bullets were provided.\"}`\nThis overrides everything else.

**Hard rule — never score down for:** missing budget **$**, category slug, or horizon when step type, merchant, and focal category are right. **“Set a budget for X” without $ = concrete (5), not vague.**

**Hard rule (out of scope):** any next step that asks Penny/the user to set up **external autopay/automatic bank transfers** (e.g. bank→credit-card payment scheduling) is **2** for appropriateness (outside Penny levers), even if it sounds financially sensible.

If Figures/Drivers **do not necessitate** budget/goal/categorization work, general monitoring or human guidance can still score **5**.
**Budget phrasing (AI budget/goal steps only):**
1. Not budget/goal/categorization → **ignore** definitive vs optional.
2. **Continuous** multi-month up/down, **erratic** history, or baseline was **$0** → optional/hedged budget wording is **5**.
3. **Only** for **one-off** spend spike or income decline vs a **non-zero** stable band (flat/narrow band then a single jump—**not** a multi-month steady climb): **directive** “set budget” → **5**; **hedged** (“if necessary”, “check if you need”) → **3**.

**Weakest line × worst dimension** (appropriateness, concreteness, relevance):

- **Appropriateness:** improves situation as described? Recategorize **miscodings** only—not merchants that **belong** in the focal category (groceries↑ + recategorize Walmart out → 2). Bank/CC **autopay** outside Penny → 2.
- **Appropriateness (budget phrasing):** apply only to AI budget/goal lines. **3** when one-off spike/decline vs non-zero band uses hedged budget language; **5** for continuous trend, erratic, or $0-baseline cases with optional phrasing.
- **Concreteness:** budget, cap, rule, named recategorize, bounded scan = pass. “Spend less” / “stop buying X” with no lever → 3.
- **Relevance:** focal What/Figures category or named miscoding—not unrelated (tuition↑ → kids-activities budget → 2).

One “A or B” / “possibly X or Y” merchant categorization rule **without** memo split when Drivers separate patterns → **3** (not 5). **No bullets at all under `## Next steps` → 1**. Vague companion (“review finances”) **does not** cap overall below **5** if another line passes all three.

**Scores:** 5 all pass; 4 minor unrelated flaw; 3 one clear flaw; 2 wrong lever; **1 only when there are no next-step bullets**.

Return JSON `{score, notes}` only.
"""


TEST_CASES: list[dict[str, Any]] = [
  {
    "name": "recategorize_zelle_maria_matches_uncategorized_finding",
    "batch": 1,
    "output": (
      '{"score": 5, "notes": "Recategorizing the Maria Zelle pattern to transfers directly addresses '
      'the uncategorized spend Drivers and Figures describe."}'
    ),
    "input": """# Rationalize What

Explain: Several Zelle payments to Maria are sitting in uncategorized spend this month. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

* **Current Month (Apr 1–30, 2026) — uncategorized Zelle to Maria:** $840.00 across 6 posted transfers (still coded as generic spend / `miscellaneous` in the ledger).
* **Prior Month (Mar 1–31, 2026) — same payee pattern:** $600.00 across 4 transfers, also uncategorized at month-end.
* **Two Months Ago (Feb 1–28, 2026):** $200.00 across 2 transfers; one line was manually recategorized mid-month, the remainder stayed uncategorized.

## Drivers

The April activity is concentrated in repeatable peer-to-peer outflows where memos consistently include **“Zelle”** and **“Maria”** (for example **Zelle payment to Maria: $140.00** on April 6 and **Zelle to Maria — thank you: $200.00** on April 19). Nothing in the descriptions suggests merchant card spend; the pattern looks like personal transfers rather than dining or shopping.

March shows the same payee text with higher frequency but the same lack of a stable category assignment, which is why April’s uncategorized bucket still contains the full run of Maria Zelle lines despite the totals growing month over month.

## Next steps

1. Set all **Zelle to Maria** transactions to **`transfers`** (past and future).
""",
  },
  {
    "name": "dining_spike_with_trend_budget_appropriate",
    "batch": 1,
    "output": (
      '{"score": 5, "notes": "Sets a dining-out budget for the focal category; directive phrasing '
      'is appropriate and concrete even alongside a general review line."}'
    ),
    "input": """# Rationalize What

Explain: Dining out is elevated this month at $620. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

* **Current Month (Apr 1–30, 2026) — `meals_dining_out`:** $620.00 (restaurant / delivery outflows only; groceries excluded).
* **Prior Month (Mar 1–31, 2026) — `meals_dining_out`:** $410.00.
* **Two Months Ago (Feb 1–28, 2026) — `meals_dining_out`:** $385.00.
* **April share of total spend (all categories):** dining represents a materially larger slice vs March even though total household spend is only modestly higher.

## Drivers

The lift is not a single anomaly: April includes multiple elevated tickets (for example **Brasserie North: $118.42** on April 12 and **Sushi Yamato: $96.10** on April 26) plus a higher cadence of smaller coffee and lunch charges that still route to `meals_dining_out`.

Compared with March, you have more weekend restaurant clusters and fewer “groceries-only” weeks; that combination explains most of the +$210 month-over-month change without requiring a data correction.

## Next steps

1. Set a **dining out** spending budget tracked against **`meals_dining_out`** (amount TBD from recent months).
2. Review finances.
""",
  },
  {
    "name": "walmart_mixed_memos_ambiguous_categorization_step",
    "batch": 2,
    "output": (
      '{"score": 3, "notes": "Walmart memos split groceries vs upkeep in Drivers; a single undecided '
      'groceries-or-upkeep rule does not resolve the mixed pattern."}'
    ),
    "input": """# Rationalize What

Explain: Walmart spend is elevated and split across categories this month. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

* **Current Month (Apr 1–30, 2026) — all Walmart-tagged spend:** $482.63 across 11 posted lines.
* **Within April, current ledger coding (pre-review):** $215.40 remains in `meals_groceries`, $198.10 in `shelter_upkeep`, and $69.13 is still split/flagged as “needs category confirmation” on import.
* **Prior Month (Mar 1–31, 2026) — Walmart-tagged spend:** $305.20 with a cleaner memo profile (mostly grocery-like descriptions).

## Drivers

April’s Walmart charges include mixed signals in the memos: several lines read like pantry and household consumables (**Walmart Grocery pickup: $84.22** on April 4), while others look like hardware and small home repairs (**Walmart Store #1441 — hardware: $63.77** on April 17). A few ambiguous “Walmart.com” charges lack item detail, which is why the importer left a residual bucket uncategorized.

That pattern matches your Rationalize prompt: spend is elevated versus March, and the category split is genuinely unclear from text alone—so the narrative risk is misclassification if we force everything into a single bucket too early.

## Next steps

1. Match **Walmart** transactions to the appropriate category—possibly **`meals_groceries`** or **`shelter_upkeep`**—based on context.
""",
  },
  {
    "name": "erratic_dining_history_budget_not_required",
    "batch": 2,
    "output": (
      '{"score": 5, "notes": "Dining Figures are erratic with no stable baseline, so omitting a budget '
      'while recategorizing miscoded lines is appropriate."}'
    ),
    "input": """# Rationalize What

Explain: Dining out is elevated this month at $250. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

* **Current Month (Apr 1–30, 2026) — `meals_dining_out`:** $250.00.
* **Prior months — `meals_dining_out` (newest to oldest):** $0, $50, $500, $0, $0, $300, $1 — no stable monthly band.
* **April miscoded restaurant lines still in `miscellaneous`:** $88.00 across 3 charges Drivers list below.

## Drivers

April’s $250 is not a clean “spike over a $150 norm”: history bounces from zero to large one-off months. The actionable issue in the feed is three restaurant charges (**Taco Cantina $32**, **Olive Garden $41**, **DoorDash — Sushi Co $15**) sitting in `miscellaneous` instead of `meals_dining_out`, which inflates uncategorized-style totals more than a new dining habit.

## Next steps

1. Recategorize the **miscellaneous** restaurant charges listed in Drivers to **`meals_dining_out`**.
""",
  },
  {
    "name": "coinbase_miscoded_exchange_fits_categorization",
    "batch": 2,
    "output": (
      '{"score": 5, "notes": "Moving Coinbase ACH debits out of miscellaneous matches Drivers’ '
      'exchange-funding pattern even without a stated slug."}'
    ),
    "input": """# Rationalize What

Explain: Coinbase purchase debits are still landing in miscellaneous this month. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

* **Current Month (Apr 1–30, 2026) — Coinbase-labeled debits posted to `miscellaneous`:** $750.00 across 3 ACH pulls (no merchant category override applied).
* **Prior Month (Mar 1–31, 2026) — same pattern:** $500.00 across 2 pulls, also in `miscellaneous` at month close.
* **Two Months Ago (Feb 1–28, 2026):** $0.00 with Coinbase in the memo field (no exchange-linked activity that month).

## Drivers

The April lines are classic exchange funding descriptors (for example **COINBASE INC. ACH DEBIT: $250.00** on April 2 and **COINBASE.COM WEB PURCHASE: $300.00** on April 18). They do not resemble normal card shopping at a retailer, and there is no stable mapping yet from the bank feed to a non-`miscellaneous` Penny category.

Because the amounts are recurring in shape (multiple pulls in one month) but the ledger keeps coding them as miscellaneous, the “still landing in miscellaneous” framing in your Rationalize prompt matches what we see in the categorized totals.

## Next steps

1. Set **Coinbase** transactions to a dedicated non-shopping category for past and future (slug TBD).
""",
  },
  {
    "name": "dining_trend_optional_budget_phrasing_not_directive",
    "batch": 3,
    "output": (
      '{"score": 5, "notes": "Three-month continuous dining increase is trend-consistent, so '
      'optional budget phrasing is acceptable for this AI budget step."}'
    ),
    "input": """# Rationalize What

Explain: Dining out is elevated this month at $620. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

* **Current Month (Apr 1–30, 2026) — `meals_dining_out`:** $620.00.
* **Prior Month (Mar 1–31, 2026) — `meals_dining_out`:** $410.00.
* **Two Months Ago (Feb 1–28, 2026) — `meals_dining_out`:** $385.00.

## Drivers

April restaurant tickets (**Brasserie North $118.42**, **Sushi Yamato $96.10**) plus more weekday lunch/coffee charges explain the +$210 vs March; the category has risen three months in a row.

## Next steps

1. Check if you need a **dining out** budget for **`meals_dining_out`** given the recent increase.
""",
  },
  {
    "name": "dining_trend_missing_budget_only_generic_advice",
    "batch": 3,
    "output": (
      '{"score": 3, "notes": "Figures show a steady dining climb ($385→$410→$620) but next steps only '
      'offer generic spend-less advice with no category budget or cap."}'
    ),
    "input": """# Rationalize What

Explain: Dining out is elevated this month at $620. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

* **Current Month (Apr 1–30, 2026) — `meals_dining_out`:** $620.00.
* **Prior Month (Mar 1–31, 2026) — `meals_dining_out`:** $410.00.
* **Two Months Ago (Feb 1–28, 2026) — `meals_dining_out`:** $385.00.

## Drivers

April restaurant tickets (**Brasserie North $118.42**, **Sushi Yamato $96.10**) plus more weekday lunch/coffee charges explain the +$210 vs March; the category has risen three months in a row.

## Next steps

1. Spend less on restaurants this month.
""",
  },
  {
    "name": "dining_trend_monitor_then_budget_if_necessary",
    "batch": 3,
    "output": (
      '{"score": 3, "notes": "Stable dining trend plus spike warrants a directive budget; '
      'monitor and budget if necessary leaves the budget optional."}'
    ),
    "input": """# Rationalize What

Explain: Dining out is elevated this month at $250. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

* **Current Month (Apr 1–30, 2026) — `meals_dining_out`:** $250.00.
* **Prior Month (Mar 1–31, 2026) — `meals_dining_out`:** $160.00.
* **Two Months Ago (Feb 1–28, 2026) — `meals_dining_out`:** $145.00.

## Drivers

Dining out has sat in a narrow $140–$160 band for the past two months and then stepped up to **$250** this month, driven by a few larger weekend meals plus more frequent smaller tickets, not a one-off refund reversal.

## Next steps

1. Monitor your restaurant spending and set a **dining out** budget for **`meals_dining_out`** if necessary.
""",
  },
  {
    "name": "dining_one_off_from_zero_baseline_optional_budget_ok",
    "batch": 3,
    "output": (
      '{"score": 5, "notes": "One-off spike from a $0 baseline does not require definitive budget '
      'phrasing; optional budget language is acceptable."}'
    ),
    "input": """# Rationalize What

Explain: Dining out jumped this month to $180. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

* **Current Month (Apr 1–30, 2026) — `meals_dining_out`:** $180.00.
* **Prior two months — `meals_dining_out`:** $0.00, $0.00 (no restaurant spend posted).

## Drivers

April is the first month with material dining activity after two $0 months—a one-off restart of restaurant spend, not a non-zero baseline band that broke.

## Next steps

1. Consider setting a **dining out** budget for **`meals_dining_out`** if you want to cap this new activity.
""",
  },
  {
    "name": "only_generic_spend_less_discretionary_up",
    "batch": 3,
    "output": (
      '{"score": 3, "notes": "Spend less does not target the discretionary categories or merchants '
      'Drivers tied to the April lift."}'
    ),
    "input": """# Rationalize What

Explain: Discretionary spend is up this month. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

* **Current Month (Apr 1–30, 2026) — combined discretionary (`leisure_entertainment` + `leisure_travel` + `shopping_clothing` + `shopping_gadgets`):** $1,095.40.
* **Prior Month (Mar 1–31, 2026) — same bucket definition:** $820.75.
* **Two Months Ago (Feb 1–28, 2026):** $790.20.
* **April lift vs March (absolute):** +$274.65, driven more by frequency than by one or two huge charges.

## Drivers

The increase is broad-based: entertainment subscriptions ticked up slightly, but the bigger change is a higher count of mid-sized discretionary purchases (new headphones, weekend tickets, and a few apparel orders) rather than a single outlier transaction.

March was comparatively quiet in apparel and gadgets, so April’s month-over-month change is visible in both the category totals and the transaction list even though essentials like rent and utilities look stable.

## Next steps

1. Spend less.
""",
  },
  {
    "name": "automatic_bank_transfer_cc_payment_out_of_scope",
    "batch": 4,
    "output": (
      '{"score": 2, "notes": "Automatic bank-to-card payment scheduling is outside Penny product levers '
      'and does not address the categorized spend Drivers explain."}'
    ),
    "input": """# Rationalize What

Explain: Credit card balance is up and minimum payments are larger this month. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

* **Current Month (Apr 1–30, 2026) — statement balance (primary card, ending 4412):** $4,860.00 outstanding as of April 28 close (up from March’s statement snapshot).
* **Prior Month (Mar 1–31, 2026) — statement balance (same card):** $3,220.00.
* **April minimum due / payment lines posted:** minimum due **$122.00** vs March minimum due **$78.00**; scheduled payment amount in the ledger also increased accordingly.
* **April net new purchases on the card (approx., from feed):** +$1,540.00 after payments and credits.

## Drivers

The larger minimum is mechanically tied to the higher statement balance: you carried more into April and added net new spend after the prior cycle’s payment. The feed shows fewer large paydowns in early April compared with March, so the required minimum moves up even without assuming any penalty APR change.

Your Rationalize prompt is directionally right: both the balance level and the minimum payment line items are elevated versus last month, and the payment cadence in the transaction history looks tighter around the due date.

## Next steps

1. Set up **automatic transfers** from the user’s checking account to **pay the credit card** each month before the due date.
""",
  },
  {
    "name": "missing_recurring_payments_historical_scan_appropriate",
    "batch": 1,
    "output": (
      '{"score": 5, "notes": "A bounded April vs prior-6-month recurring scan matches Drivers’ missing '
      'CityPower and BroadbandCo anchors."}'
    ),
    "input": """# Rationalize What

Explain: Total spend looks lower than usual this month; some recurring bills may not have cleared. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

* **Current Month (Apr 1–30, 2026) — all-category outflows (cash + card + ACH debits):** $6,420.00.
* **Prior Month (Mar 1–31, 2026) — same definition:** $8,910.00.
* **Two Months Ago (Feb 1–28, 2026):** $8,540.00.
* **April “full month run rate” context:** if you annualize only the first 21 days of April vs the first 21 days of March, totals still trail—so the softness is not solely “April isn’t finished yet.”

## Drivers

April is missing several recurring anchors that normally appear every month in the ledger between the 1st and the 25th (for example your typical **CityPower Electric** ACH and **BroadbandCo** autopay). In the prior six months, those payees posted like clockwork, but April’s feed shows no matching debit lines yet.

That gap is consistent with “lower than usual” spend: it may be timing/posting delay, a skipped bill, or a changed payment method—but the pattern is visible as absent transactions rather than as a category miscode.

## Next steps

1. For **2026-04-01 through 2026-04-30**, scan **historical transactions from the prior 6 full calendar months** and list **recurring monthly payees** that **did not post any payment** in that April window (include amount pattern used as the recurrence signal).
""",
  },
  {
    "name": "zelle_finding_wrong_grocery_categorization",
    "batch": 2,
    "output": (
      '{"score": 2, "notes": "Drivers describe P2P Zelle transfers, but the step routes Maria payments to '
      'groceries, which contradicts the finding."}'
    ),
    "input": """# Rationalize What

Explain: Several Zelle payments to Maria are sitting in uncategorized spend this month. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

* **Current Month (Apr 1–30, 2026) — uncategorized Zelle to Maria:** $840.00 across 6 posted transfers.
* **Prior Month (Mar 1–31, 2026) — same payee pattern:** $600.00 across 4 transfers, also uncategorized.

## Drivers

Memos consistently show **Zelle** payments to **Maria** with no merchant spend cues—personal transfers, not grocery runs.

## Next steps

1. Set all **Zelle to Maria** transactions to **`meals_groceries`**.
""",
  },
  {
    "name": "utilities_trend_up_budget_step_appropriate",
    "batch": 1,
    "output": (
      '{"score": 5, "notes": "Shelter utilities rose on a stable band in Figures; a utilities budget step '
      'fits the spike Drivers explain."}'
    ),
    "input": """# Rationalize What

Explain: Utilities are higher than usual this month. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

* **Current Month (Apr 1–30, 2026) — `shelter_utilities`:** $248.00.
* **Prior three months — `shelter_utilities`:** $162, $158, $165 (CityPower + gas in a narrow band).
* **April drivers in category:** **CityPower Electric $142.00** (vs ~$95–$98 prior months) plus **Metro Gas $106.00** (winter rate step).

## Drivers

The jump is not random noise: **CityPower** posted ~$45 higher than its usual bill and **Metro Gas** added a seasonal surcharge visible in the memo line—together they explain almost all of the month-over-month change.

## Next steps

1. Set a monthly **utilities** budget for **`shelter_utilities`**.
""",
  },
  {
    "name": "empty_next_steps_when_recategorization_clearly_needed",
    "batch": 4,
    "output": (
      '{"score": 1, "notes": "No next-step bullets were provided."}'
    ),
    "input": """# Rationalize What

Explain: Several Zelle payments to Maria are sitting in uncategorized spend this month. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

* **Current Month (Apr 1–30, 2026) — uncategorized Zelle to Maria:** $840.00 across 6 transfers in `miscellaneous`.
* **Prior Month (Mar 1–31, 2026):** $600.00 across 4 transfers, same coding issue.

## Drivers

Repeat **Zelle to Maria** memos with no retail merchant text; pattern unchanged from March.

## Next steps

""",
  },
  {
    "name": "no_ai_levers_needed_general_steps_ok",
    "batch": 4,
    "output": (
      '{"score": 5, "notes": "No AI levers are necessitated by Figures/Drivers, so general monitoring steps are fine."}'
    ),
    "input": """# Rationalize What

Explain: Dining out looks normal this month. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

* **Current Month (Apr 1–30, 2026) — `meals_dining_out`:** $152.00.
* **Prior Month (Mar 1–31, 2026) — `meals_dining_out`:** $148.00.
* **Two Months Ago (Feb 1–28, 2026) — `meals_dining_out`:** $155.00.

## Drivers

Dining out is steady in a narrow band with no sign of a spike, miscoding, or missing recurring items; the mix of merchants and ticket sizes looks consistent with prior months.

## Next steps

1. Keep an eye on dining out to maintain this steady level.
2. Review your weekly spend summary.
""",
  },
  {
    "name": "groceries_up_walmart_recategorize_inappropriate",
    "batch": 2,
    "output": (
      '{"score": 2, "notes": "Walmart charges belong in groceries per Drivers; recategorizing them '
      'out would not appropriately address groceries being up."}'
    ),
    "input": """# Rationalize What

Explain: Groceries is up this week at $210. (2026-07-06 to 2026-07-12)

# Rationalize Response

## Figures

* **Groceries (`meals_groceries`) Jul 6–12, 2026:** $210.00 vs $145.00 (Jun 29–Jul 5, 2026).
* **Within Jul 6–12 groceries:** **Walmart Grocery pickup** and in-store Walmart food lines total **$118.00** (already coded `meals_groceries`).

## Drivers

The week-over-week lift is mostly larger **Walmart** grocery runs (**Walmart Grocery pickup $64.22**, **Walmart Supercenter $53.78**)—descriptions read as pantry/food, not miscoded gadgets or transfers.

## Next steps

1. Recategorize **Walmart** transactions out of **`meals_groceries`** to lower grocery spend.
""",
  },
  {
    "name": "groceries_up_apple_miscode_recategorize_appropriate",
    "batch": 1,
    "output": (
      '{"score": 5, "notes": "Recategorizing miscoded Apple.com charges to gadgets appropriately '
      'addresses groceries inflated by non-grocery spend."}'
    ),
    "input": """# Rationalize What

Explain: Groceries is up this week at $210. (2026-07-06 to 2026-07-12)

# Rationalize Response

## Figures

* **Groceries (`meals_groceries`) Jul 6–12, 2026:** $210.00 vs $145.00 prior week.
* **Included in groceries total:** **Apple.com $129.00** (posted under `meals_groceries`).

## Drivers

**Apple.com $129.00** is a one-time **AirPods** order (electronics), miscoded into groceries at import—true grocery merchants (**Trader Joe’s**, **Walmart Grocery**) explain only part of the lift; fixing Apple removes the miscoding distortion.

## Next steps

1. Recategorize **Apple.com** transactions to **`shopping_gadgets`**.
""",
  },
  {
    "name": "clothing_up_stop_buying_not_concrete",
    "batch": 3,
    "output": (
      '{"score": 3, "notes": "Stop buying clothing is not a concrete Penny action—no budget, rule, '
      'or merchant target for how spend would improve."}'
    ),
    "input": """# Rationalize What

Explain: Clothing is up this week at $180. (2026-07-06 to 2026-07-12)

# Rationalize Response

## Figures

* **`shopping_clothing` Jul 6–12, 2026:** $180.00 vs $95.00 prior week.

## Drivers

Two apparel orders (**Nordstrom $110**, **Old Navy $70**) drive the increase vs a quiet prior week.

## Next steps

1. Stop buying clothing.
""",
  },
  {
    "name": "clothing_up_set_budget_concrete",
    "batch": 1,
    "output": (
      '{"score": 5, "notes": "A directive clothing budget is concrete and relevant to the '
      'shopping_clothing spike in Figures."}'
    ),
    "input": """# Rationalize What

Explain: Clothing is up this week at $180. (2026-07-06 to 2026-07-12)

# Rationalize Response

## Figures

* **`shopping_clothing` Jul 6–12, 2026:** $180.00 vs $95.00 prior week.

## Drivers

Two apparel orders (**Nordstrom $110**, **Old Navy $70**) drive the increase vs a quiet prior week.

## Next steps

1. Set a monthly budget for **`shopping_clothing`**.
""",
  },
  {
    "name": "tuition_up_kids_activities_budget_irrelevant",
    "batch": 4,
    "output": (
      '{"score": 2, "notes": "A kids activities budget does not target tuition—the focal category '
      'named in What and Figures."}'
    ),
    "input": """# Rationalize What

Explain: Tuition is up this month at $2,400. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

* **`education_tuition` Apr 2026:** $2,400.00 vs $1,800.00 Mar 2026 (spring installment posting).

## Drivers

**State University spring tuition $2,400** posted Apr 3; no change in **`education_kids_activities`** ($120, flat vs prior months).

## Next steps

1. Set a monthly budget for **`education_kids_activities`**.
""",
  },
  {
    "name": "tuition_up_tuition_budget_relevant",
    "batch": 1,
    "output": (
      '{"score": 5, "notes": "A tuition budget is relevant and concrete for the education_tuition '
      'increase Figures and Drivers describe."}'
    ),
    "input": """# Rationalize What

Explain: Tuition is up this month at $2,400. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

* **`education_tuition` Apr 2026:** $2,400.00 vs $1,800.00 Mar 2026 (spring installment posting).

## Drivers

**State University spring tuition $2,400** posted Apr 3; installment is larger than the fall monthly plan amounts.

## Next steps

1. Set a monthly budget for **`education_tuition`**.
""",
  },
]


class CheckerOptimizer:
  def __init__(
    self,
    model_name: str = "gemini-flash-lite-latest",
    *,
    max_output_tokens: int = 384,
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

    # Hard, deterministic override: if there are no bullet/numbered lines under
    # "## Next steps", the score must be 1 (per product spec).
    if "## Next steps" in user_msg:
      after = user_msg.split("## Next steps", 1)[1]
      # Stop at next markdown H2 heading if present.
      after = re.split(r"(?m)^##\s+", after, maxsplit=1)[0]
      has_bullets = any(
        re.match(r"^\s*(?:\d+\.|[-*])\s+\S", line)
        for line in after.splitlines()
      )
      if not has_bullets:
        return {"score": 1, "notes": "No next-step bullets were provided."}

    contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_msg)])]
    cfg = types.GenerateContentConfig(
      temperature=self.temperature,
      top_p=self.top_p,
      max_output_tokens=self.max_output_tokens,
      safety_settings=self.safety_settings,
      system_instruction=[types.Part.from_text(text=self.system_prompt)],
      thinking_config=types.ThinkingConfig(
        thinking_budget=self.thinking_budget,
        include_thoughts=False,
      ),
      response_mime_type="application/json",
      response_schema=OUTPUT_SCHEMA,
    )
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
  parser.add_argument("--max-output-tokens", type=int, default=384)
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
