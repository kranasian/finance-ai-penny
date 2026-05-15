"""
Rationalize rubric checker optimizer: Comprehensive.

Use this to iterate on the system_prompt that will be stored in `penny_templates`
for the checker template `Chk:RationalizePerCategoryComprehensive`.

Run from `finance-ai-penny` repo root:

  python3 active_experiments/hr_rationalize_per_category_comprehensive_optimizer.py --test all --check
  python3 active_experiments/hr_rationalize_per_category_comprehensive_optimizer.py --test groceries_spike_off_topic_travel_data_ideal_one
  python3 active_experiments/hr_rationalize_per_category_comprehensive_optimizer.py --batch 1 --check
  python3 active_experiments/hr_rationalize_per_category_comprehensive_optimizer.py --test all --model gemini-flash-lite-latest

**Fixtures:** four calibrated cases (batches **1–4**); each supplies **`name`**, **`batch`**, optional **`review_outcome`**, markdown **`input`**, and expected **`output`** JSON.

**Recommended minimal generation settings** (re-validate with `python3 active_experiments/hr_rationalize_per_category_comprehensive_optimizer.py --test all --check`; rubric: **unnecessary data**, **usable scope**, **timeframe**, **`notes` completeness**, parent/leaf rules):

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
        "Semicolon-separated clauses; list every rubric issue observed (unnecessary rows, wrong grain, "
        "scope bleed, missing causes, etc.), not only the severest."
      ),
    ),
  },
)


SYSTEM_PROMPT = """Grade **`## Figures`** + **`## Drivers`** against `# Rationalize What`: **no unnecessary data**, **usable causal fit**, **category scope** (leaf vs parent, siblings, full child set), **timeframe fit**, and **sufficiency**—ignore prose polish and **`## Next steps`** unless they repeat numbers.

**Judge order:** **Unnecessary / wrong-zoom data** → **Usable / off-topic** → **What window match** → **scope & aggregation** → **per-sub cause bar** (when the What names multiple children) → **MoM/WoW** → **parallel** legs.

**Unnecessary data (strict):** Drop or penalize figures/drivers that **do not help** the What’s **stated category + timeframe**:
- **Finer What** (e.g. one **week**, one **leaf**): do **not** add **coarser** series (full **month**, **MTD month**, unrelated month rollups) for that **same** leaf **unless** the What explicitly widens the lens.
- **Leaf-only What:** do **not** bulk-present **parent** roll-ups or **sibling** sub-lines as explanatory content “for context” unless the What asks for the parent or those siblings.
Violations cap quality (**≤4**; paired with weak proof or many extras → **≤3**).

**Usable data (hard 1):** If nothing on-topic could explain the What’s category and window, **score 1**.

**What window match:** **Figures** ranges must match the **What’s inclusive dates**. If any **focal** bucket **ends after** the What’s **end** or **starts before** its **start**, **score 1**. If the What uses **this week**/**week** language and **no** **Figures** row uses **week-based** labeling (e.g. **Week**, **ISO week**), **score 1**.

**MoM / WoW span:** **Three inclusive** periods for that track are enough unless the What asks for more.

**Grain purity (weekly-only What):** **Monthly/MTD** for the **same** weekly-only claim → **≤3** even if weekly rows look strong.

**Category scope (taxonomy):** **Leaf-only What:** stay on the leaf; **material sibling** lines in **Figures** or **Drivers** → **≤3**—**never** **4/5** with that bleed. **Parent What:** only that parent’s children—Meals → Dining Out, Delivered Food, Groceries; Leisure → Entertainment, Travel; Shopping → Pets, Clothing, Gadgets, Kids; Bills → Connectivity, Insurance, Tax, Service Fees; Shelter → Home, Utilities, Upkeep; Education → Kids Activities, Tuition; Transportation → Car, Public; Health → Medical and Pharmacy, Gym and Wellness, Personal Care; Income → Salary, Sidegig, Business, Interest; Donations and Gifts, Miscellaneous, Transfers.

**Parent aggregation:** Parent-as-actual must show **every** composing child **or** reconcile omissions; subset-only parent picture → **≤3**.

**Per-sub causes:** If the What names **several** children, **Drivers** need cause-grade support **per named child**; parent-only narrative with a child table only → **≤3**.

**Comparisons optional:** Isolating focal rows can suffice when they explain the move.

**Parallel movements:** Each named leg needs support unless the What relaxes one side.

**Narrow-ask tangents:** Centered stream coherent but **Drivers** pivot elsewhere → cap **5** at **4** unless timeframe/scope fails.

**Driver certainty:** Named dated txn pins the driver while hedged → **cap at 4**.

**Hard outputs (apply last, literal caps):** (1) If **Usable data** mandates **1** → **1**. (2) If **What window match** mandates **1** (focal **end date** after the What’s **end**, **start** before the What’s **start**, or **this week**/**week** language without **week-based** figure labels) → **1**. (3) Else keep the drafted score from the ladder—**do not** soften a **1** to **2** or **3** for partial narrative.

**Scores (1–5):** **5** clean; **4** one narrow miss; **3** unnecessary data, scope, aggregation, or per-sub gaps; **2** thin but on-topic; **1** unusable/off-topic or hard timeframe fail.

**`notes` (completeness, mandatory style):** Use **semicolon-separated** short clauses and list **every** issue you flag in **Figures** or **Drivers**—**wrong zoom**, **unnecessary parent/sibling/month rows**, **window mismatch**, **missing per-sub causes**, **tangents**, **hedge**, etc.—**including small ones**; do **not** summarize down to only the single “main” gap.

**Before JSON:** Re-check **Hard outputs**, **Unnecessary data**, and **`notes` completeness**.

Return only JSON `{score, notes}` per schema.
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
    "output": "{\"score\": 3, \"notes\": \"Sibling meals_delivered_food and meals_groceries figure rows are unnecessary for a dining-out-only What; Drivers pivot to overall Meals narrative—extra parent-scope content vs the leaf ask.\"}",
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
