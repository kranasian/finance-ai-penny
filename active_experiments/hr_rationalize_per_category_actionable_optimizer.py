"""
Rationalize rubric checker optimizer: **Actionable** — score **only** AI-facing **## Next steps** (instruction-level specificity; categorization targets must use valid category slugs). If there are **no** AI-facing steps, score **defaults to 5**.

Use this to iterate on the system_prompt that will be stored in `penny_templates`
for the checker template `Chk:RationalizePerCategoryActionable`.

Run from `finance-ai-penny` repo root:

  python3 active_experiments/hr_rationalize_per_category_actionable_optimizer.py --test all
  python3 active_experiments/hr_rationalize_per_category_actionable_optimizer.py --batch 1 --check
  python3 active_experiments/hr_rationalize_per_category_actionable_optimizer.py --test all --model gemini-flash-lite-latest

Batches **1–6** partition fixtures (two in batch 1; one each in batches 2–6). Use **`--check`** to assert each case’s integer **score** matches expected JSON.

**Recommended minimal generation settings** (validated `python3 active_experiments/hr_rationalize_per_category_actionable_optimizer.py --test all`; spot-check scores vs `ideal_response`):

- **model:** `gemini-flash-lite-latest`
- **temperature:** `0` · **top_p:** `0.95`
- **thinking_budget:** `0`
- **max_output_tokens:** `128` (raise to `256` if truncated)
- **response:** `application/json` + **response_schema** for `{score, notes}`

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
      description="One short sentence.",
    ),
  },
)


SYSTEM_PROMPT = """Rubric grader. Input: markdown with `# Rationalize Response`. Judge **only** `## Next steps` that are **for the AI** (product-executable instructions). Read Figures/Drivers **only** to interpret references in those steps—**do not** score Figures or Drivers for quality, truth, or completeness.

**AI-facing line** = tells product/AI to **recategorize**, **merchant→category rule**, **budget/goal** with numbers and horizon, or **historical analysis** with explicit scope (merchants/categories/dates, **or** detect **missing expected recurring payments** for a **named month** by contrasting with **prior months** in history)—not generic “review finances.”

**Human-only lines** (lifestyle, “spend less,” vague self-review): **not** AI-facing. They **do not** affect the numeric score.

**Default when there are no AI-facing lines** (empty `## Next steps`, section missing, or **only** human-only / non-product bullets): **5**. There is nothing for the AI to execute, so there is nothing to mark down.

When **one or more** AI-facing lines exist, the score follows the **weakest** AI-facing line only (ignore human-only lines beside them).

**Executable** = another model could run it **without inventing** missing inputs: match text or merchant set, **numeric** budget/goal + period, and for any assign/recategorize/rule (and category-scoped budgets): **exactly one** target slug from the Category List (character-for-character). Multiple allowed slugs named for **one** operation (“A or B”) = **not** one target → treat as **2**.

**Invalid slug** = categorization target string **not** in the list (e.g. `investments`). If that is the **only** substantive defect and the merchant/action is otherwise clear → **4**, not **3** or **5**.

**Category List** (exact tokens; categorization/recategorization/rule/budget-by-category targets must match one line):

- `meals`
- `meals_dining_out`
- `meals_delivered_food`
- `meals_groceries`
- `leisure`
- `leisure_entertainment`
- `leisure_travel`
- `shopping_pets`
- `bills`
- `bills_connectivity`
- `bills_insurance`
- `bills_tax`
- `bills_service_fees`
- `shelter`
- `shelter_home`
- `shelter_utilities`
- `shelter_upkeep`
- `education`
- `education_kids_activities`
- `education_tuition`
- `shopping`
- `shopping_clothing`
- `shopping_gadgets`
- `shopping_kids`
- `transportation`
- `transportation_car`
- `transportation_public`
- `health`
- `health_medical_pharmacy`
- `health_gym_wellness`
- `health_personal_care`
- `donations_gifts`
- `miscellaneous`
- `income`
- `income_salary`
- `income_sidegig`
- `income_business`
- `income_interest`
- `transfers`

**Scores (1–5)** — if **no** AI-facing lines → **5** (default). If **≥1** AI-facing line → worst AI line wins:
- **5** — Every AI line executable; category targets are valid list tokens; no unresolved “pick one of several categories” for a single action.
- **4** — **Single** defect: one invalid slug **or** one mildly underspecified AI line; otherwise strong.
- **3** — **Two or more** stacked defects (e.g. multiple invalid slugs, or invalid slug **and** vagueness).
- **2** — Main AI categorization leaves **which one** category unresolved among alternatives, or core parameters missing, **or** the step(s) sit **outside in-scope levers** (e.g. automatic bank→card payment transfers / external autopay scheduling).
- **1** — Only when ≥1 AI-facing line: worst line is below the bar for **2** (e.g. mutually contradictory AI instructions with no resolution path).

**`notes`**: One sentence on the decisive issue (invalid slug, ambiguous multi-target categorization, out-of-scope payment automation, all AI lines clear, **or** no AI-facing steps so default **5**). If **5** with a mix of solid AI steps and human-only bullets, say human lines were **ignored for the score**.

Return only JSON `{score, notes}` per schema.
"""


TEST_CASES: list[dict[str, Any]] = [
  {
    "name": "ai_next_steps_specific_enough_recategorize_zelle",
    "batch": 1,
    "output": '{"score": 5, "notes": "Recategorization names a clear payee pattern and a valid target slug (`transfers`)."}',
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
    "name": "ai_specific_human_vague_human_out_of_scope",
    "batch": 1,
    "output": '{"score": 5, "notes": "AI budget step is concrete with amount and valid category; vague human line is out of scope."}',
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

1. Set **dining out** spending budget to **$500/month** tracked against **`meals_dining_out`**.
2. Review finances.
""",
  },
  {
    "name": "ai_vague_walmart_ambiguous_target_category",
    "batch": 2,
    "output": '{"score": 2, "notes": "Walmart step leaves the target category undecided between groceries and upkeep, so the AI cannot execute it as written."}',
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
    "name": "ai_specific_coinbase_invalid_investments_slug",
    "batch": 3,
    "output": '{"score": 4, "notes": "Merchant and intent are clear, but `investments` is not on the Category List, so categorization is not actionable."}',
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

1. Set **Coinbase** transactions to **`investments`** for past and future.
""",
  },
  {
    "name": "only_human_next_step_not_ai_actionable",
    "batch": 4,
    "output": '{"score": 5, "notes": "No AI-facing next steps; score defaults to 5."}',
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
    "batch": 5,
    "output": '{"score": 2, "notes": "Automatic bank-to-card payment transfers are outside Penny in-scope AI levers, so the step is not executable here."}',
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
    "name": "ai_identify_missing_monthly_payments_via_history",
    "batch": 6,
    "output": '{"score": 5, "notes": "Bounded historical analysis: named month plus comparison window to find recurring payees with no payment."}',
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
]


class CheckerOptimizer:
  def __init__(
    self,
    model_name: str = "gemini-flash-lite-latest",
    *,
    max_output_tokens: int = 128,
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
    cfg = types.GenerateContentConfig(
      temperature=self.temperature,
      top_p=self.top_p,
      max_output_tokens=self.max_output_tokens,
      safety_settings=self.safety_settings,
      system_instruction=[types.Part.from_text(text=self.system_prompt)],
      thinking_config=types.ThinkingConfig(
        thinking_budget=self.thinking_budget,
        include_thoughts=True,
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
  parser.add_argument("--max-output-tokens", type=int, default=128)
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

