"""
Rationalize rubric checker optimizer: **Actionable** — AI-executable **## Next steps** (instruction-level specificity; categorization targets must use valid category slugs).

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


SYSTEM_PROMPT = """Rubric grader. Input: markdown with `# Rationalize Response`. Judge **only** `## Next steps`. Read Figures/Drivers **only** to interpret references—**do not** score those sections for quality, truth, or completeness.

**AI-facing line** = tells product/AI to **recategorize**, **merchant→category rule**, **budget/goal** with numbers and horizon, or **historical analysis** with explicit scope (merchants/categories/dates, **or** detect **missing expected recurring payments** for a **named month** by contrasting with **prior months** in history)—not generic “review finances.”

**Human-only lines** (lifestyle, “spend less,” vague self-review): **ignore** for the numeric score. The score follows the **weakest AI-facing line** only. If there are **no** AI-facing lines → **1**.

**Executable** = another model could run it **without inventing** missing inputs: match text or merchant set, **numeric** budget/goal + period, and for any assign/recategorize/rule (and category-scoped budgets): **exactly one** target slug from the Category List (character-for-character). Multiple allowed slugs named for **one** operation (“A or B”) = **not** one target → treat as **2**.

**Invalid slug** = categorization target string **not** in the list (e.g. `investments`). If that is the **only** substantive defect and the merchant/action is otherwise clear → **4**, not **3** or **5**.

**Category List** (exact tokens; categorization/recategorization/rule/budget-by-category targets must match one line):

- `meals_dining_out`
- `meals_delivered_food`
- `meals_groceries`
- `leisure_entertainment`
- `leisure_travel`
- `shopping_pets`
- `bills_connectivity`
- `bills_insurance`
- `bills_tax`
- `bills_service_fees`
- `shelter_home`
- `shelter_utilities`
- `shelter_upkeep`
- `education_kids_activities`
- `education_tuition`
- `shopping_clothing`
- `shopping_gadgets`
- `shopping_kids`
- `transportation_car`
- `transportation_public`
- `health_medical_pharmacy`
- `health_gym_wellness`
- `health_personal_care`
- `donations_gifts`
- `miscellaneous`
- `income_salary`
- `income_sidegig`
- `income_business`
- `income_interest`
- `transfers`

**Scores (1–5)** — worst AI line wins; **1** if no AI lines.
- **5** — Every AI line executable; category targets are valid list tokens; no unresolved “pick one of several categories” for a single action.
- **4** — **Single** defect: one invalid slug **or** one mildly underspecified AI line; otherwise strong.
- **3** — **Two or more** stacked defects (e.g. multiple invalid slugs, or invalid slug **and** vagueness).
- **2** — Main AI categorization leaves **which one** category unresolved among alternatives, or core parameters missing, **or** the step(s) sit **outside in-scope levers** (e.g. automatic bank→card payment transfers / external autopay scheduling).
- **1** — Missing section, or **only** non-AI steps.

**`notes`**: One sentence on the decisive issue (invalid slug, ambiguous multi-target categorization, no AI steps, out-of-scope payment automation, or all clear). If **5** and a **human-only** bullet sits next to solid AI steps, say human lines were **ignored for the score**.

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

- Multiple “Zelle to Maria” lines remain uncategorized.

## Drivers

Peer-to-peer pattern is stable; memos consistently include Zelle and Maria.

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

- Dining out: $620 this month.

## Drivers

Restaurant spend is above the user’s typical range.

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

- Walmart-tagged lines total about $480 across mixed categories.

## Drivers

Receipt mix suggests both groceries and home upkeep, but the split is unclear from memos alone.

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

- Coinbase-labeled debits remain in `miscellaneous`.

## Drivers

Descriptions read like crypto exchange funding rather than general shopping.

## Next steps

1. Set **Coinbase** transactions to **`investments`** for past and future.
""",
  },
  {
    "name": "only_human_next_step_not_ai_actionable",
    "batch": 4,
    "output": '{"score": 1, "notes": "Only a human behavior tip; no AI-executable product instruction."}',
    "input": """# Rationalize What

Explain: Discretionary spend is up this month. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

- Discretionary categories are up vs last month.

## Drivers

General lift across small purchases.

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

- Card balance and payment lines are elevated vs last month.

## Drivers

Higher statement balance drives larger required payments.

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

- April 2026 spend is down vs typical full-month run rates.

## Drivers

Several merchants that usually post monthly have not appeared yet on the ledger.

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

