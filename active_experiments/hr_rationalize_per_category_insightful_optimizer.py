"""
Rationalize rubric checker optimizer: Insightful.

Use this to iterate on the system_prompt that will be stored in `penny_templates`
for the checker template `Chk:RationalizePerCategoryInsightful`.

Run from `finance-ai-penny` repo root:

  python3 active_experiments/hr_rationalize_per_category_insightful_optimizer.py --test all
  python3 active_experiments/hr_rationalize_per_category_insightful_optimizer.py --test direction_mismatch_corrected_with_figure_chain
  python3 active_experiments/hr_rationalize_per_category_insightful_optimizer.py --test all --model gemini-flash-lite-latest

**Recommended minimal generation settings** (re-validate `--test all`; scores match `ideal_response` **5 / 1**):

- **model:** `gemini-flash-lite-latest`
- **temperature:** `0` · **top_p:** `0.95`
- **thinking_budget:** `0`
- **max_output_tokens:** `128` (`256` if truncation)
- **response:** `application/json` + **response_schema** for `{score, notes}`

**Rubric:** **`# Rationalize What` is ground truth**; grade **only** **`# Rationalize Response`** for insightful **period-over-period movement** (vs prior week/month or a **$0** baseline)—not composition-only totals. **$0 in Figures = real tracked zero.** **Do not** reward disputing the What. **Ignore** **`## Next steps`**.

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
      description="One short sentence on cause coverage, data-to-cause fit, or clarity of links to the What.",
    ),
  },
)


SYSTEM_PROMPT = """Grade **only** **`# Rationalize Response`** for **insightfulness** against **`# Rationalize What`**.

**Ground truth:** **`# Rationalize What` is authoritative.** Every amount, direction (“up,” “down,” “flat,” “$0”), period, and labeled move in the What is **correct as stated**. Grade whether the Response **explains those moves**—not whether the What “should” differ. **Never reward** disputing the What. **Figure/What tension:** score from **`## Drivers` only** for contradiction. A **Figures** table that differs from the What’s headline is **not** a contradiction—**only** Driver sentences that **deny** the What’s stated direction/amount count.
**Focal-period rule:** each What move pins a **focal period** (e.g. “this month,” “this week”). Drivers must explain **that focal period’s** move. **Prior-period** detail **alone** (last month’s trip, last week’s total) **without** explaining the **focal** move → **≤2** for that What line.

**Scope:** Evaluate **only** **`## Figures`** and **`## Drivers`**. Ignore **`## Next steps`**. Assume Response data is **sufficient**; judge **use** and **explanation**, not missing pulls.

**Core bar — movement, not composition:** The What names **increases/decreases** vs a prior week or month. Insightful Responses must explain **that period-over-period movement** (why focal spend/income **rose or fell** vs the immediately relevant prior bucket in Figures—prior week, prior month, or a **real $0** baseline). **Insufficient:** only describing **how the focal total was built** (gross vs net mix, line items, refunds netting) **without** tying causes to **change vs the prior period**. **Composition ≠ movement cause:** naming **which charges** sum to this week’s total and noting “higher than last week” is **not enough**—insight requires **why this week’s pattern differs from prior weeks** (e.g. new/unusual subscriptions, first-time fees, absence of similar lines in prior buckets). **Wise use** means prior-period figures drive **why the move happened**, not wallpaper.

**Insightful (two pillars):**
1. **Wise use of Response data** — figures/drivers deployed **on purpose** for **What** moves (not wallpaper). Tight **prior period → mechanism → focal movement** chains.
2. **Clear causal linkage** — reader sees **how** each cause produces the **specific** What line’s **direction of change** (which line, up/down, focal period).

**Interpretation rules (apply when scoring):**
- **Any $0 in Figures (prior or focal):** **real tracked $0** spending/income for that period—**not** missing tracking, not “data gap.” Use $0 history to explain **movement** (restart from zero, drop to zero, stayed at zero vs prior spend).
- **“Refunds” in the What:** a **net negative aggregate** (credits/returns/reversals summed), **not** necessarily one refund transaction. Insightful Responses explain **that net refund line** and its role in net/gross moves the What names.
- **$0 total with no transactions implied:** if the What’s move is **$0** and Response figures show **$0** with **no new charges**, stating **no transactions / no activity in period** is a **complete** cause—do **not** demand extra mechanisms. **Cap at 2** if Drivers **only** describe **prior-period trips/spend** and **never** state why the **focal $0 period** has no new charges/activity.
- **Restatement vs partial:** **1** = Drivers only **repeat** What amounts/directions and/or **recite** figure lines (including “Mar/Feb were $0”) with **no why/mechanism**. **3** = names a **why** but thin (generic refunds, generic “more activity”) without txn/merchant/pause detail.
- **Tautology vs movement:** circular “up because you spent more” **without** citing **change vs prior week/month** (when Figures include a prior bucket) → **2**, not **3**. **3** requires at least a **stated** prior-period contrast (even thin), not just affirming the What.

**Scores (integer 1–5)**
- **5** — **Every** What move gets a **specific, traceable** cause for **movement vs prior week/month** (or vs a **$0** baseline), using Response figures/drivers **in service of the What** (including “no new charges” explaining a **drop to / stay at $0** vs prior spend).
- **4** — Strong; **one** move slightly thin on data use or link clarity.
- **3** — **Accepts the What** and cites **vs prior week/month** plus **focal charge names**, but **only composition** (which txns built the total) **without** why those charges/pattern **did not appear** in prior periods; **or** thin/generic movement why; **or** partial multi-line coverage.
- **2** — **Contradicts** the What in Drivers; **ignores** focal movement while narrating **other periods** or **composition only**; **or** restates direction with **no prior-period comparison**.
- **1** — Bare restatement of What/figures with **no** causal story, or causes **wholly disconnected** from What moves.

**Hard caps (apply before returning):** (1) Drivers **deny** a What direction/amount in prose → **≤2**. (2) What says **$0 this period** and Drivers lack **why focal $0 vs prior spend** (no activity / no new charges) → **≤2**. (3) Drivers **only** restate What + figure lines → **1**. (4) Drivers give **composition only** (txn list, gross/net mix) **with no vs-prior-period contrast at all** → **≤2**. **Composition + vs-prior contrast but no why pattern differs from prior weeks** → **3** (do not score **4/5**). **Exception:** if Drivers **only recite** What + figure lines (e.g. “up to $118; Mar/Feb were $0”) with **zero mechanism** → **1**, not **3**. **Do not** cap for Drivers **repeating** the What headline while a **Figures** row differs—that is **3** if a thin movement why is present, else **2**.

**Calibration examples:** What “up at **$180**” + Figures prior week **$139.55** + Drivers “up because you ate out more” **with no vs-prior-week movement** → **2**. What “net **$430**” + Figures **$430 vs $395** March + Drivers “more activity and some refunds” **without vs-March movement** → **2**. Service Fees **$626** vs **$53** prior week + Drivers name **Dave Subscription** charges **without why those did not occur prior weeks** → **3** (composition + comparison, not spike cause). What “up at **$180**” + Drivers “posted dining is **actually down**” → **≤2** (denies What). What “**$0** April travel” + Drivers “**no April bookings** vs **$890** March” → **5** (movement to zero explained).

**`notes`:** One auditable sentence on **What-move coverage**, **movement vs composition**, **data-to-cause fit**, or **link clarity** (note **contradiction**, **focal skip**, **composition-only**, or **restatement**).

**Before JSON:** Re-read **`## Drivers`** and re-apply **all hard caps**; confirm score matches the **movement-not-composition** bar.

Return **only** JSON `{score, notes}` per schema.
"""


TEST_CASES: list[dict[str, Any]] = [
  {
    "name": "direction_mismatch_corrected_with_figure_chain",
    "batch": 1,
    "output": '{"score": 2, "notes": "Contradicts the What’s up-at-$180 move instead of explaining it; figures are used to dispute truth, not reconcile causes."}',
    "input": """# Rationalize What

Explain: Dining Out is significantly up this week at $180. (2026-05-03 to 2026-05-09)

# Rationalize Response

## Figures

* **Dining Out (May 3–9, 2026):** $94.80 posted vs $139.55 (Apr 26–May 2, 2026) vs $128.40 (Apr 19–25, 2026).
* **Pending hold cleared this week (same category):** $85.20 authorization from May 2 that posted May 6 and is no longer open.

## Drivers

Your summary says dining is “significantly up,” but posted **Dining Out** is actually **down** week-over-week ($94.80 vs $139.55). The $180 headline likely mixed in the **$85.20** hold that cleared—without that artifact, spend is softer, not higher.

## Next steps

1. Set a weekly cap on `meals_dining_out` if you want posted dining to stay under $120.
""",
  },
  {
    "name": "direction_mismatch_unaddressed_low_insight",
    "batch": 1,
    "output": '{"score": 2, "notes": "Affirms the What’s up direction but gives no movement vs prior week—only a tautology, not period-over-period insight."}',
    "input": """# Rationalize What

Explain: Dining Out is significantly up this week at $180. (2026-05-03 to 2026-05-09)

# Rationalize Response

## Figures

* **Dining Out (May 3–9, 2026):** $94.80 vs $139.55 (Apr 26–May 2, 2026).

## Drivers

Dining out is up this week at about $180 because you ate out more.

## Next steps

1. Watch restaurant spend.
""",
  },
  {
    "name": "prior_weeks_zero_totals_explains_restart",
    "batch": 2,
    "output": '{"score": 5, "notes": "Uses $0 prior weeks on-purpose to explain why April delivered-food spend looks like a restart after a pause."}',
    "input": """# Rationalize What

Explain: Delivered Food is significantly up this month at $118. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

* **Delivered Food (Apr 1–30, 2026):** $118.40.
* **Delivered Food (Mar 1–31, 2026):** $0.00.
* **Delivered Food (Feb 1–28, 2026):** $0.00.
* **ISO week buckets (Apr):** $0.00 (Apr 1–7); $0.00 (Apr 8–14); $41.20 (Apr 15–21); $77.20 (Apr 22–28).

## Drivers

The “up” story is really a **restart**: March and February show **$0** because delivery subscriptions were paused—there is no prior-week baseline to compare against until mid-April, when DoorDash and Uber Eats charges resume and concentrate in the last two ISO weeks.

## Next steps

1. Tag DoorDash and Uber Eats → `meals_delivered_food` so the restart stays visible next month.
""",
  },
  {
    "name": "prior_months_zero_restated_without_cause",
    "batch": 2,
    "output": '{"score": 1, "notes": "Lists $0 prior months but only restates the What move without explaining why spend restarted."}',
    "input": """# Rationalize What

Explain: Delivered Food is significantly up this month at $118. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

* **Delivered Food (Apr 1–30, 2026):** $118.40.
* **Delivered Food (Mar 1–31, 2026):** $0.00.
* **Delivered Food (Feb 1–28, 2026):** $0.00.

## Drivers

Delivered food is up to $118. March and February were $0.

## Next steps

1. Keep monitoring delivery spend.
""",
  },
  {
    "name": "dining_out_refunds_cause_clear",
    "batch": 3,
    "output": '{"score": 5, "notes": "Ties the What’s $250 refund line to gross vs net figures and named return/chargeback mechanisms."}',
    "input": """# Rationalize What

Explain: Dining Out net spend is elevated this month at $430, including **$250 in refunds** for Dining Out that offset gross restaurant charges. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

* **Dining Out gross (Apr 1–30, 2026):** $680.00 posted charges.
* **Dining Out refunds/credits (Apr 1–30, 2026):** −$250.00 across 4 lines.
* **Dining Out net (Apr 1–30, 2026):** $430.00 vs $395.00 net (Mar 1–31, 2026).

## Drivers

The **$250 in refunds** is not generic noise—it clusters on two mechanisms: **Chipotle** duplicate-tap reversals (**−$142.00** total across Apr 9 and Apr 21) and an **OpenTable** deposit chargeback (**−$108.00** on Apr 14). Gross dining looks high, but refunds explain why net is only modestly above March despite several large tickets.

## Next steps

1. Review OpenTable holds so future deposits do not post as full dining spend before they clear.
""",
  },
  {
    "name": "dining_out_refunds_vague_partial",
    "batch": 3,
    "output": '{"score": 2, "notes": "Names refunds and activity but does not explain movement vs March—composition-only, not period-over-period causes."}',
    "input": """# Rationalize What

Explain: Dining Out net spend is elevated this month at $430, including **$250 in refunds** for Dining Out that offset gross restaurant charges. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

* **Dining Out net (Apr 1–30, 2026):** $430.00 vs $395.00 (Mar 1–31, 2026).
* **Refunds (Apr):** −$250.00 total.

## Drivers

You had more dining activity in April and some refunds came back, which is why net is $430.

## Next steps

1. Set a monthly dining budget.
""",
  },
  {
    "name": "travel_zero_explained_by_absence",
    "batch": 4,
    "output": '{"score": 5, "notes": "Explains the What’s $0 Travel move with purposeful use of prior-month totals and no new April bookings."}',
    "input": """# Rationalize What

Explain: Travel and Vacations is **$0** this month. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

* **Travel and Vacations (Apr 1–30, 2026):** $0.00.
* **Travel and Vacations (Mar 1–31, 2026):** $890.00 (airfare + hotel for spring break).
* **Travel and Vacations (Feb 1–28, 2026):** $1,205.00 (ski trip deposits and lift tickets).

## Drivers

April is **$0** because the March trip already captured airfare and lodging, and there are **no new flight or hotel charges** posted in April—unlike February/March, which show concentrated booking clusters. The category is flat by absence of new trips, not a data gap.

## Next steps

1. If you expect May travel, add a placeholder budget line for `leisure_travel` so $0 months are easier to spot.
""",
  },
  {
    "name": "travel_zero_ignored_in_drivers",
    "batch": 4,
    "output": '{"score": 2, "notes": "Shows $0 April travel in figures but drivers pivot to March trip detail without explaining why April stayed at zero."}',
    "input": """# Rationalize What

Explain: Travel and Vacations is **$0** this month. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

* **Travel and Vacations (Apr 1–30, 2026):** $0.00.
* **Travel and Vacations (Mar 1–31, 2026):** $890.00.

## Drivers

March travel was driven by spring-break airfare and a hotel stay in Orlando; that is the main travel story in your recent history.

## Next steps

1. Review March trip receipts.
""",
  },
  {
    "name": "low_insight_restates_without_interpretation",
    "batch": 5,
    "output": '{"score": 1, "notes": "Restates What and headline figures without real causes linking each named move to drivers."}',
    "input": """# Rationalize What

Explain: Utilities is significantly down this month at $324. Shelter is thus slightly down this month to $3174. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

* Utilities is $324.
* Shelter is $3,174.

## Drivers

Utilities is down and shelter changed.

## Next steps

1. Keep an eye on it.
""",
  },
  {
    "name": "service_fees_spike_composition_without_prior_week_why",
    "batch": 1,
    "review_outcome": "",
    "output": '{"score": 3, "notes": "States vs-prior-week movement and names focal charges, but does not explain why similar fees were absent in prior weeks—composition of the total, not the spike cause."}',
    "input": """# Rationalize What

Explain: Service Fees is significantly up this week at $626. (2026-05-10 to 2026-05-13)

# Rationalize Response

## Figures

* **$626.50**: Service Fees, current week (May 10–13)
* **$53.13**: Service Fees, prior week (May 3–9)
* **$193.92**: Service Fees, two weeks ago (April 26–May 2)

## Drivers

Service Fees are significantly higher this week, totaling $626.50 compared to $53.13 in the previous week. This spike is primarily driven by two large payments labeled "Dave Subscription" on May 11th.

**Concrete charges for the current period:**
* **Dave Subscription**: $405.00 (2026-05-11)
* **Dave Subscription**: $219.00 (2026-05-11)
* **7-Eleven FCTI Withdrawal Fee**: $2.50 (2026-05-11)

## Next steps

1. **Review Subscriptions**: Audit the two "Dave Subscription" charges ($405.00 and $219.00) to confirm if these are authorized and expected, as they represent the vast majority of this week's service fees.
2. **Budgeting**: Set a weekly limit for `bills_service_fees` to avoid future spikes.
3. **Categorization Rule**: If these subscriptions are recurring, ensure they are categorized correctly in your budget plan to prevent them from inflating your "Service Fees" category unexpectedly.
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
      thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget, include_thoughts=True),
      response_mime_type="application/json",
      response_schema=OUTPUT_SCHEMA,
    )
    out = self.client.models.generate_content(model=self.model_name, contents=contents, config=cfg)
    text = (out.text or "").strip()
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

  print(f"\n{_TEST_SEPARATOR}\n")
  print(f"# Total tests: {len(cases)}\n")


if __name__ == "__main__":
  main()

