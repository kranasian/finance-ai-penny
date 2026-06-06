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
      description=(
        "Integer 1-5 rubric score."
      ),
    ),
    "notes": types.Schema(
      type=types.Type.STRING,
      description=(
        "One short sentence on the decisive appropriateness issue for ## Next steps."
      ),
    ),
  },
)


SYSTEM_PROMPT = """Grade **AI-directed (ie. budget/goal/categorization) `## Next steps` lines only**, if any. Do not grade Figures/Drivers quality.

## Input
- **Figures** = truth
- **Drivers** = why / miscoding
- **Rationalize What** = focal scope. 

## Rules

If `## Next steps` is blank, return **exactly**:\n`{\"score\": 1, \"notes\": \"No next-step bullets were provided.\"}`. If Figures/Drivers **do not necessitate** budget/goal/categorization work, general monitoring or human guidance can still score **5**.

Should improve the situation as described (ie. minimize outflows and/or maximize inflows).

**Re-categorization and Categorization Rules:**
- **Inferred Categories**: Appropriate when transaction name is at least 70% likely to be a certain category, and/or Drivers support the category. Inappropriate otherwise, especially for P2P or bank transfers without a specified purpose.
- **Multiple Suggested Categories**: "categorize X as A or B" / "X is possibly X or Y" merchant rule → **3**.
- **Valid Categories** (parent → subcategories): Synonyms allowed.
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

**Budget:**
1. Multi-month steady climb or decline (e.g. $385→$410→$620), erratic history, or baseline was **$0**: optional/hedged budget wording ("check if you need", "if necessary") is **5**—do **not** require directive "set budget" for trend-consistent rises.
2. Narrow stable band then one step-up (e.g. $145→$160 in a tight band, then $250), one-off spike with non-zero stable band: directive "set budget" → **5**; **hedged** "monitor and set budget if necessary" → **3**.

**Weakest line × worst dimension** (appropriateness, concreteness, relevance)—grade **each** line, then take the **minimum** score; never average or upgrade because another line passes:

**Scores:** 5 all pass; 4 minor unrelated flaw; 3 one clear flaw (vague advice, hedged budget, ambiguous A-or-B rule); **2 wrong lever or contradicts Drivers**; **1 only when there are no next-step bullets**.

Return JSON `{score, notes}` only.
"""


TEST_CASES: list[dict[str, Any]] = [
  {
    "name": "zelle_maria_blanket_dining_out_rule_inappropriate",
    "batch": 1,
    "output": (
      '{"score": 2, "notes": "Drivers describe P2P Zelle transfers to Maria, but routing all payee '
      'lines to dining out contradicts the transfer pattern—Zelle memos can be for anything."}'
    ),
    "input": """# Rationalize What

Explain: Several Zelle payments to Maria are sitting in uncategorized spend this month. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

* **Current Month (Apr 1–30, 2026) — uncategorized Zelle to Maria:** $840.00 across 6 posted transfers (still coded as generic spend / `miscellaneous` in the ledger).
* **Prior Month (Mar 1–31, 2026) — same payee pattern:** $600.00 across 4 transfers, also uncategorized at month-end.
* **Two Months Ago (Feb 1–28, 2026):** $200.00 across 2 transfers; one line was manually recategorized mid-month, the remainder stayed uncategorized.

## Drivers

The April activity is concentrated in repeatable peer-to-peer outflows where memos consistently include **"Zelle"** and **"Maria"** (for example **Zelle payment to Maria: $140.00** on April 6 and **Zelle to Maria — thank you: $200.00** on April 19). Nothing in the descriptions suggests merchant card spend; the pattern looks like personal transfers rather than dining or shopping.

March shows the same payee text with higher frequency but the same lack of a stable category assignment, which is why April's uncategorized bucket still contains the full run of Maria Zelle lines despite the totals growing month over month.

## Next steps

1. Set all **Zelle to Maria** transactions to **`meals_dining_out`** (past and future).
""",
  },
  {
    "name": "zelle_maria_dinner_line_ok_blanket_dining_rule_not",
    "batch": 1,
    "output": (
      '{"score": 2, "notes": "Recategorizing the Dinner memo line fits, but a blanket Zelle-to-Maria '
      'dining-out rule is inappropriate because P2P transfers can be for anything."}'
    ),
    "input": """# Rationalize What

Explain: Several Zelle payments to Maria are sitting in uncategorized spend this month. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

* **Current Month (Apr 1–30, 2026) — uncategorized Zelle to Maria:** $840.00 across 6 posted transfers in `miscellaneous`.
* **Prior Month (Mar 1–31, 2026) — same payee pattern:** $600.00 across 4 transfers, also uncategorized.

## Drivers

April memos mix purposes: **Zelle to Maria: Dinner $85.00** on April 8 reads like a shared-meal reimbursement, while **Zelle payment to Maria: rent split $400.00** on April 14 and **Zelle to Maria — thank you: $120.00** on April 22 look like general P2P transfers with no dining cue. The payee pattern is repeatable, but memo text does not justify one dining category for every line.

## Next steps

1. Recategorize **Zelle to Maria: Dinner** transactions to **`meals_dining_out`**.
2. Set all **Zelle to Maria** transactions to **`meals_dining_out`** (past and future).
""",
  },
  {
    "name": "mcdonalds_dining_out_rule_appropriate",
    "batch": 1,
    "output": (
      '{"score": 5, "notes": "McDonald\'s charges are restaurant spend miscoded in miscellaneous; '
      'a dining-out categorization rule is appropriate for this merchant pattern."}'
    ),
    "input": """# Rationalize What

Explain: Dining out looks understated because fast-food charges are still in miscellaneous this month. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

* **Current Month (Apr 1–30, 2026) — `meals_dining_out`:** $412.00.
* **Current Month — `miscellaneous` McDonald's-tagged card spend:** $68.40 across 4 posted lines not yet routed to dining out.
* **Prior Month (Mar 1–31, 2026) — same McDonald's pattern in `miscellaneous`:** $41.20 across 3 lines.

## Drivers

The feed shows classic quick-service restaurant descriptors (**McDonald's #2841: $12.85** on April 3, **MCDONALDS MOBILE ORDER: $18.22** on April 17, **McDonald's Drive Thru: $9.41** on April 24). These are merchant card charges, not peer-to-peer transfers, and the merchant is overwhelmingly dining-out in nature (~90% likelihood from historical coding).

## Next steps

1. Set **McDonald's** transactions to **`meals_dining_out`** (past and future).
""",
  },
  {
    "name": "walmart_groceries_rule_acceptable_default",
    "batch": 1,
    "output": (
      '{"score": 5, "notes": "Walmart is a mixed retailer but groceries is an acceptable default rule '
      'given the grocery-heavy memo profile Drivers describe."}'
    ),
    "input": """# Rationalize What

Explain: Groceries looks low because Walmart runs are still uncategorized this month. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

* **Current Month (Apr 1–30, 2026) — `meals_groceries`:** $318.00.
* **Current Month — uncategorized Walmart-tagged spend:** $186.50 across 7 posted lines flagged on import.
* **Prior Month (Mar 1–31, 2026) — Walmart eventually coded to groceries after review:** $142.00 across 5 lines.

## Drivers

April's uncategorized Walmart charges mostly read like food and pantry purchases (**Walmart Grocery pickup: $64.22** on April 4, **Walmart Supercenter — groceries: $38.15** on April 11, **Walmart.com grocery delivery: $29.88** on April 19). A minority of lines lack item detail, but the dominant signal is grocery-like; Walmart is ~70% likely to be groceries in this household's history.

## Next steps

1. Set **Walmart** transactions to **`meals_groceries`** (past and future).
""",
  },
  {
    "name": "user_directed_payment_and_cancel_steps_acceptable",
    "batch": 1,
    "output": (
      '{"score": 5, "notes": "Canceling unused subscriptions and sending the landlord payment are '
      'reasonable user-directed steps even though Penny cannot execute them directly."}'
    ),
    "input": """# Rationalize What

Explain: Connectivity and shelter bills look higher than usual this month. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

* **Current Month (Apr 1–30, 2026) — `bills_connectivity`:** $186.00 vs $142.00 prior month (duplicate streaming services visible in the feed).
* **Current Month (Apr 1–30, 2026) — `shelter_home`:** $1,650.00 rent due Apr 1; ledger shows no outbound rent payment posted by Apr 28.
* **Prior Month (Mar 1–31, 2026) — `shelter_home`:** $1,650.00 with rent paid Mar 2.

## Drivers

Connectivity rose because both **StreamFlix $15.99** and **CineMax+ $14.99** posted again in April even though usage logs suggest one service is idle. Rent is the larger issue: the April **Oak Street Apartments** charge is still outstanding in the transaction list with a due date in the memo, unlike March where payment cleared in the first week.

## Next steps

1. Cancel any **unused streaming subscriptions** you no longer watch.
2. Send the **April rent payment** to **Oak Street Apartments** through your bank or landlord portal before the due date.
""",
  },
  {
    "name": "recategorize_zelle_maria_to_transfers_appropriate",
    "batch": 2,
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

## Drivers

The April activity is concentrated in repeatable peer-to-peer outflows where memos consistently include **"Zelle"** and **"Maria"** (for example **Zelle payment to Maria: $140.00** on April 6 and **Zelle to Maria — thank you: $200.00** on April 19). Nothing in the descriptions suggests merchant card spend; the pattern looks like personal transfers rather than dining or shopping.

## Next steps

1. Set all **Zelle to Maria** transactions to **`transfers`** (past and future).
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
* **Within April, current ledger coding (pre-review):** $215.40 remains in `meals_groceries`, $198.10 in `shelter_upkeep`, and $69.13 is still split/flagged as "needs category confirmation" on import.
* **Prior Month (Mar 1–31, 2026) — Walmart-tagged spend:** $305.20 with a cleaner memo profile (mostly grocery-like descriptions).

## Drivers

April's Walmart charges include mixed signals in the memos: several lines read like pantry and household consumables (**Walmart Grocery pickup: $84.22** on April 4), while others look like hardware and small home repairs (**Walmart Store #1441 — hardware: $63.77** on April 17). A few ambiguous "Walmart.com" charges lack item detail, which is why the importer left a residual bucket uncategorized.

That pattern matches your Rationalize prompt: spend is elevated versus March, and the category split is genuinely unclear from text alone—so the narrative risk is misclassification if we force everything into a single bucket too early.

## Next steps

1. Match **Walmart** transactions to the appropriate category—possibly **`meals_groceries`** or **`shelter_upkeep`**—based on context.
""",
  },
  {
    "name": "groceries_up_walmart_recategorize_out_inappropriate",
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
    "name": "dining_spike_with_trend_budget_appropriate",
    "batch": 2,
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

## Drivers

The lift is not a single anomaly: April includes multiple elevated tickets (for example **Brasserie North: $118.42** on April 12 and **Sushi Yamato: $96.10** on April 26) plus a higher cadence of smaller coffee and lunch charges that still route to `meals_dining_out`.

Compared with March, you have more weekend restaurant clusters and fewer "groceries-only" weeks; that combination explains most of the +$210 month-over-month change without requiring a data correction.

## Next steps

1. Set a **dining out** spending budget tracked against **`meals_dining_out`** (amount TBD from recent months).
2. Review finances.
""",
  },
  {
    "name": "dining_trend_optional_budget_phrasing_acceptable",
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
    "name": "dining_trend_generic_spend_less_not_concrete",
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
    "name": "dining_spike_monitor_then_budget_if_necessary",
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

* **Current Month (Apr 1–30, 2026) — statement balance (primary card, ending 4412):** $4,860.00 outstanding as of April 28 close (up from March's statement snapshot).
* **Prior Month (Mar 1–31, 2026) — statement balance (same card):** $3,220.00.
* **April minimum due / payment lines posted:** minimum due **$122.00** vs March minimum due **$78.00**; scheduled payment amount in the ledger also increased accordingly.

## Drivers

The larger minimum is mechanically tied to the higher statement balance: you carried more into April and added net new spend after the prior cycle's payment. The feed shows fewer large paydowns in early April compared with March, so the required minimum moves up even without assuming any penalty APR change.

## Next steps

1. Set up **automatic transfers** from the user's checking account to **pay the credit card** each month before the due date.
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
