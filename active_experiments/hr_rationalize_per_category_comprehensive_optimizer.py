"""
Rationalize rubric checker optimizer: Comprehensive.

Use this to iterate on the system_prompt that will be stored in `penny_templates`
for the checker template `Chk:RationalizePerCategoryComprehensive`.

Run from `finance-ai-penny` repo root:

  python3 active_experiments/hr_rationalize_per_category_comprehensive_optimizer.py --test all
  python3 active_experiments/hr_rationalize_per_category_comprehensive_optimizer.py --test good_comprehensive_shelter
  python3 active_experiments/hr_rationalize_per_category_comprehensive_optimizer.py --test good_comprehensive_shopping
  python3 active_experiments/hr_rationalize_per_category_comprehensive_optimizer.py --test all --model gemini-flash-lite-latest

**Recommended minimal generation settings** (validated with `python3 active_experiments/hr_rationalize_per_category_comprehensive_optimizer.py --test all`; scores **5 / 5 / 2** vs `ideal_response`; prompt stresses evidence, investigation depth, auditable `notes`):

- **model:** `gemini-flash-lite-latest` — smallest model that held calibration on all fixtures; larger Flash variants not re-checked here (API limits).
- **temperature:** `0` · **top_p:** `0.95` — deterministic judge (`template_run_configs.py` uses `temp=0.2` for the same template; use `0` here for optimizer reproducibility).
- **thinking_budget:** `0` — JSON schema output needs no chain-of-thought budget.
- **max_output_tokens:** `128` suffices for `{score, notes}`; production config often uses `256` with margin.
- **response:** `response_mime_type=application/json` + `response_schema` for `score` / `notes`.

**Input:** a single markdown **`str`**—only the sections **`# Rationalize What`**, **`# Rationalize Response`** (typically **`## Figures`**, **`## Drivers`**, **`## Next steps`**). No JSON wrapper; no **`rubric`** / **`case_id`**.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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
      description="One short sentence.",
    ),
  },
)


SYSTEM_PROMPT = """**Axis: comprehensiveness** for **rationalize_change**. Does **Rationalize Response** give **evidence-backed** explanations—amounts, periods, named categories/merchants—and cover the **investigations implied by Rationalize What** (comparisons across months, subcategories, merchant mix, etc.)? Judge **only** the markdown provided.

**Structure:** `# Rationalize What` then `# Rationalize Response` (typically `## Figures`, `## Drivers`, `## Next steps`).

**Grade on**
1. **Figures** — Enough concrete numbers/dates/splits to anchor the story.
2. **Drivers** — Causes tied to those figures or named merchants/categories (not generic language).
3. **Investigation depth** — Missing angles the insight needs (e.g. multi-line insights need each line addressed in writing; “why up/down” needs prior-period or mix context when the strong answers do).

**Scores (integer 1–5):** **5** = figures + drivers + needed investigations, all supported by the text. **4** = one secondary gap. **3** = misses an important angle. **2** = thin (several gaps) but **some** substance—use **2** not **1** when at least one real figure + weak driver exists. **1** = no usable rationalization.

If unsure between **1** and **2**, choose **2** when any concrete total appears with a loosely relevant driver.

**`notes`:** One sentence; name concrete dimensions (subcategories, which months compared, merchants) that justify the score.
"""


TEST_CASES: list[dict[str, Any]] = [
  {
    "name": "good_comprehensive_shelter",
    "batch": 1,
    "output": '{"score": 5, "notes": "Figures by subcategory + total; My data shows tone; explains April slightly-up-to-$170 vs quiet months and vs March."}',
    "input": """# Rationalize What

Explain: Home is slightly up this month at $120. Utilities is significantly down this month at $38. Shelter is thus slightly up this month to $170. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

*   **Shelter Home:** $120.00 (Apr 1–30, 2026), $120.00 (Mar 1–31, 2026), $120.00 (Feb 1–28, 2026).
*   **Shelter Utilities:** $38.00 (Apr 1–30, 2026), $95.00 (Mar 1–31, 2026), $25.00 (Feb 1–28, 2026).
*   **Shelter Upkeep:** $12.00 (Apr 1–30, 2026), $35.00 (Mar 1–31, 2026), $5.00 (Feb 1–28, 2026).
*   **Total Identified Shelter Costs:** $170.00 (Apr 1–30, 2026), $250.00 (Mar 1–31, 2026), $150.00 (Feb 1–28, 2026).
*   **Total Identified Shelter Costs (earlier months):** $150.00 (Nov 1–30, 2025), $155.00 (Dec 1–31, 2025), $150.00 (Jan 1–31, 2026).

## Drivers

My data shows that your `shelter_home` spending has stayed flat at **$120.00** for the last three months (Apr–Feb), rather than increasing—so the “home is slightly up” line does not match actual `shelter_home` in this window.

Versus **March**, your total shelter is **down**: that step-down is mostly from lower `shelter_utilities` (**$38.00** this month vs **$95.00** in March) and lower `shelter_upkeep` (**$12.00** vs **$35.00** in March). March transaction detail shows the utilities/upkeep side concentrated in a few larger lines; April does not repeat March’s biggest lumps—more recurring-sized charges.

The insight still reads **“Shelter … slightly up … to $170” for April** because **$170** is being judged against your **quieter months**, not against March’s spike. November–January totals sit around **~$150–155**, and February is **$150**; March at $250 is the clear outlier once you include that history. **April at $170** is only **modestly above** that **~$150–155** band (about **+$15–20**), which is what “slightly up” is picking up—while March’s extra spend **does not fully disappear in one month**, so April **has not fully dropped back** to the fall/winter floor even though it is **down from March**.

## Next steps

1. Review your April utility bills to confirm whether the lower `shelter_utilities` amount reflects usage, seasonal timing, or a one-time billing adjustment.
2. Monitor `shelter_upkeep` in May to see whether April’s lower upkeep holds or March-level charges return.
3. After May closes, check whether **Total Identified Shelter Costs** move back toward the **~$150–155** range or settle nearer **~$170**.
""",
  },
  {
    "name": "good_comprehensive_shopping",
    "batch": 1,
    "output": '{"score": 5, "notes": "Totals plus merchant mix: Nordstrom/REI/Apple vs Feb–Mar Target/Walmart (discount big-box) and Old Navy (value apparel)."}',
    "input": """# Rationalize What

Explain: Shopping is up significantly this month to $485. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

*   **Shopping (`shopping` and tagged children):** $485.00 (Apr 1–30, 2026), $195.00 (Mar 1–31, 2026), $210.00 (Feb 1–28, 2026).
*   **April average transaction size (category):** ~$97 across 5 posted purchases vs ~$33 across 6 in March and ~$35 across 6 in February (from transaction export).

## Drivers

My data shows the increase is **not** from shopping more often in April (similar visit counts vs Feb–Mar)—it is from **where** the spend landed. Transactions only include **merchant** (`name`), **amount**, and **category**—there is no product or receipt line. In April, the top shopping rows are merchants **Nordstrom**, **REI**, and **Apple**; **Apple** is posted to `shopping_gadgets` with a much larger **amount** than any `shopping_gadgets` row in Feb–Mar. By contrast, **February and March** are dominated by **discount / big-box** (**Target**, **Walmart**) and **value apparel** (**Old Navy**)—those are your **cheaper-store** merchants—with **smaller amounts** per transaction on average.

So **shopping is up** mainly because the **merchant mix shifted** from that discount-and-value pattern toward **Nordstrom**, **REI**, and **Apple**, not because the category picked up lots of tiny impulse buys.

## Next steps

1. Review April’s **Nordstrom** / **REI** / **Apple** merchant totals (data is merchant + amount only—triage using category and size of the charge).
2. If you want to bring the total down, route routine basics back to the **Target / Walmart / Old Navy** pattern you used in Feb–Mar for comparable items.
3. Watch May’s first half: if large-ticket store spend repeats, treat it as a habit shift; if not, April likely stays a one-off mix change.
""",
  },
  {
    "name": "missing_investigation",
    "batch": 1,
    "output": '{"score": 2, "notes": "Missing key investigations and merchant-level drivers for the delta."}',
    "input": """# Rationalize What

Explain: Home is slightly up this month at $120. Utilities is significantly down this month at $38. Shelter is thus slightly up this month to $170. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

* Shelter is up this month to $170.

## Drivers

It looks like utilities went down, which helped offset other shelter changes.

## Next steps

1. Watch shelter spending next month.
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
    user_msg = (
      "Grade **comprehensive** only. Input is markdown (plain text):\n\n"
      + (agent_outcome or "")
      + "\n\nRespond with the JSON object only."
    )
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
      # best-effort extract
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

