"""
Top-takeaways rubric checker optimizer — **accuracy only**.

Grades only the **accuracy** axis for `type=top_takeaways` outcomes.

Per `finance-ai-docs/Backend/DB/AiAgentOutcomes.md`, the `agent_outcome` for `top_takeaways`
is a single markdown `str` with:
- `# Top Takeaways`
- `## Highlights`
- `## Lowlights`

Run from `finance-ai-penny` repo root:

  python3 active_experiments/hr_top_takeaways_accuracy_optimizer.py
  python3 active_experiments/hr_top_takeaways_accuracy_optimizer.py --test all
  python3 active_experiments/hr_top_takeaways_accuracy_optimizer.py --test 0
  python3 active_experiments/hr_top_takeaways_accuracy_optimizer.py --batch 1
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

try:
  from google import genai  # type: ignore[import-not-found]
  from google.genai import types  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
  genai = None
  types = None

if load_dotenv is not None:
  load_dotenv()

_TEST_SEPARATOR = "=" * 72
_SECTION_RULE = "-" * 72


def _print_section_banner(title: str) -> None:
  print(f"\n{_SECTION_RULE}\n{title}\n{_SECTION_RULE}\n")


SYSTEM_PROMPT = """Grade **accuracy** only.

Input includes:
1) The **rationalize contexts bundle**: a `<CONTEXTS>` block with **one or more** `<CONTEXT>` entries (even for a single context, it is still wrapped in `<CONTEXTS>`), each containing `<RATIONALIZE>` markdown.
2) The candidate **top_takeaways agent_outcome** markdown (`<TOP_TAKEAWAYS>...</TOP_TAKEAWAYS>`), which must be exactly one `# Top Takeaways` section with `## Highlights` and `## Lowlights`.

**Accuracy rubric** (score 1–5):
- **5** — The `<TOP_TAKEAWAYS>` bullets are grounded in the `<CONTEXTS>` rationalize contexts (ignore each rationalize’s `## Next steps`), avoid hallucinated figures/merchants/categories/periods, and keep coherent directionality (highlight vs lowlight).
- **3** — Some vagueness or minor unsupported specificity, but no clear contradictions vs `<CONTEXTS>`.
- **1** — Clear hallucinations or contradictions vs `<CONTEXTS>` (e.g., invents precise dollar amounts/merchants not present, swaps highlight vs lowlight direction, or adds extra sections / wrong headings).

Return **only** the JSON object matching the schema (`score`, `notes`).
"""


TEST_CASES: list[dict[str, Any]] = [
  {
    "name": "accurate_real_two_contexts_food_and_uncategorized",
    "batch": 1,
    "input": """<CONTEXTS>

<CONTEXT index="1">

<RATIONALIZE>

# Rationalize What

Explain: Dining Out is significantly down last month at $789.  Delivered Food is significantly down last month at $0.  Food is thus significantly down last month to $1385.

# Rationalize Response

## Figures

*   **Total Food (Dining + Delivered + Groceries):** $1,385.47 (Apr 1–30, 2026) vs $1,534.25 (Mar 1–31, 2026).
*   **Dining Out:** $865.07 (Apr 1–30, 2026) vs $1,059.39 (Mar 1–31, 2026).
*   **Delivered Food:** $0.00 (Apr 1–30, 2026) vs $157.77 (Mar 1–31, 2026).

## Drivers

The decrease in food spending is primarily driven by the total elimination of **Delivered Food** in April, compared to $157.77 spent on services like Uber Eats and Bite Squad in March. Additionally, your **Dining Out** spending decreased by approximately $194.32 month-over-month. While you still frequented restaurants like IHOP and Chipotle in April, you avoided larger recurring charges at establishments like TGI Fridays and school cafeterias, which were prominent in your March spending.

## Next steps

1.  **Create a Budget:** Establish a monthly dining budget of $900 based on your April spending to maintain this downward trend.
2.  **Categorization Rule:** If you use a food delivery service in the future, ensure it is correctly categorized by using a rule: `Uber Eats → meals_delivered_food`.
3.  **Review Transactions:** If any transactions in `meals_dining_out` appear to be food delivery rather than dining out, please identify them so we can reclassify them to `meals_delivered_food` to keep your category tracking accurate.

</RATIONALIZE>

</CONTEXT>

<CONTEXT index="2">

<RATIONALIZE>

# Rationalize What

Explain: Uncategorized is slightly up last week at $1116.

# Rationalize Response

## Figures

*   **Uncategorized spending (Apr 26–May 2, 2026):** $1,115.66
*   **Uncategorized spending (Apr 19–25, 2026):** $446.84
*   **Weekly change:** +$668.82

## Drivers

The spike in uncategorized spending is primarily driven by recurring charges for **Summer Camp** and **Costco**, which accounted for all $1,115.66 in this category last week. Specifically, "Summer Camp" contributed $611.26 (two transactions of $305.63) and "Costco" contributed $504.40 (two transactions of $188.86 and two transactions of $63.34). These appear to be standard merchant charges that simply lack a permanent categorization rule.

## Next steps

1.  **Rule:** Costco → meals_groceries
2.  **Rule:** Summer Camp → kids_activities
3.  **Action:** Please review your "kids_activities" and "meals_groceries" budgets, as moving these charges will increase the totals in those categories accordingly.

</RATIONALIZE>

</CONTEXT>

</CONTEXTS>

<TOP_TAKEAWAYS>

# Top Takeaways

## Highlights

- **Reduced Food Spending:** Total monthly food expenditure dropped from $1,534.25 to $1,385.47, driven by a reduction in dining out and the elimination of delivery services.
- **Improved Dining Habits:** Dining out decreased by approximately $194 month-over-month, helped by avoiding recurring charges at higher-cost establishments like TGI Fridays.

## Lowlights

- **Uncategorized Spending Spike:** Uncategorized expenses rose significantly last week to $1,115.66, up $668.82 from the previous week.
- **Categorization Gaps:** High-frequency recurring charges for Summer Camp ($611.26) and Costco ($504.40) are currently lacking permanent classification, obscuring true spending in "kids_activities" and "meals_groceries."

</TOP_TAKEAWAYS>
""",
    "output": '{"score": 5, "notes": "Highlights/lowlights are grounded in the task contexts with consistent figures and no hallucinated specifics."}',
  },
  {
    "name": "inaccurate_hallucinates_merchants_and_amounts",
    "batch": 1,
    "input": """<CONTEXTS>

<CONTEXT index="1">

<RATIONALIZE>

# Rationalize What

Explain: Dining Out is significantly down last month at $789.  Delivered Food is significantly down last month at $0.  Food is thus significantly down last month to $1385.

# Rationalize Response

## Figures

*   **Total Food (Dining + Delivered + Groceries):** $1,385.47 (Apr 1–30, 2026) vs $1,534.25 (Mar 1–31, 2026).
*   **Dining Out:** $865.07 (Apr 1–30, 2026) vs $1,059.39 (Mar 1–31, 2026).
*   **Delivered Food:** $0.00 (Apr 1–30, 2026) vs $157.77 (Mar 1–31, 2026).

## Drivers

The decrease in food spending is primarily driven by the total elimination of **Delivered Food** in April, compared to $157.77 spent on services like Uber Eats and Bite Squad in March. Additionally, your **Dining Out** spending decreased by approximately $194.32 month-over-month. While you still frequented restaurants like IHOP and Chipotle in April, you avoided larger recurring charges at establishments like TGI Fridays and school cafeterias, which were prominent in your March spending.

## Next steps

1.  **Create a Budget:** Establish a monthly dining budget of $900 based on your April spending to maintain this downward trend.
2.  **Categorization Rule:** If you use a food delivery service in the future, ensure it is correctly categorized by using a rule: `Uber Eats → meals_delivered_food`.
3.  **Review Transactions:** If any transactions in `meals_dining_out` appear to be food delivery rather than dining out, please identify them so we can reclassify them to `meals_delivered_food` to keep your category tracking accurate.

</RATIONALIZE>

</CONTEXT>

<CONTEXT index="2">

<RATIONALIZE>

# Rationalize What

Explain: Uncategorized is slightly up last week at $1116.

# Rationalize Response

## Figures

*   **Uncategorized spending (Apr 26–May 2, 2026):** $1,115.66
*   **Uncategorized spending (Apr 19–25, 2026):** $446.84
*   **Weekly change:** +$668.82

## Drivers

The spike in uncategorized spending is primarily driven by recurring charges for **Summer Camp** and **Costco**, which accounted for all $1,115.66 in this category last week. Specifically, "Summer Camp" contributed $611.26 (two transactions of $305.63) and "Costco" contributed $504.40 (two transactions of $188.86 and two transactions of $63.34). These appear to be standard merchant charges that simply lack a permanent categorization rule.

## Next steps

1.  **Rule:** Costco → meals_groceries
2.  **Rule:** Summer Camp → kids_activities
3.  **Action:** Please review your "kids_activities" and "meals_groceries" budgets, as moving these charges will increase the totals in those categories accordingly.

</RATIONALIZE>

</CONTEXT>

</CONTEXTS>

<TOP_TAKEAWAYS>

# Top Takeaways

## Highlights

- **Food Delivery Savings:** You saved $842 by cutting DoorDash and Postmates usage to $0 in April.
- **Dining Out Improvement:** Dining out dropped by $500 because you stopped going to Starbucks and Sweetgreen.

## Lowlights

- **Big Ticket Purchase:** A $2,199 Apple Store charge drove gadgets above plan.
- **New Subscription Creep:** A new $79.99/month Peloton subscription appeared and increased recurring spend.

</TOP_TAKEAWAYS>
""",
    "output": '{"score": 1, "notes": "Invents merchants/amounts and unrelated claims not supported by the task contexts."}',
  },
]


def _build_output_schema(_types: Any) -> Any:
  return _types.Schema(
    type=_types.Type.OBJECT,
    required=["score", "notes"],
    properties={
      "score": _types.Schema(type=_types.Type.INTEGER, description="Integer 1–5."),
      "notes": _types.Schema(type=_types.Type.STRING, description="One short sentence."),
    },
  )


class TopTakeawaysAccuracyCheckerOptimizer:
  def __init__(
    self,
    model_name: str = "gemini-flash-lite-latest",
    *,
    max_output_tokens: int = 128,
    thinking_budget: int = 0,
  ):
    if genai is None or types is None:
      raise ImportError(
        "Missing dependency for Gemini client. Install `google-genai` (and ensure your venv is active) "
        "to run this script."
      )
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set.")
    self.client = genai.Client(api_key=api_key)
    self._types = types
    self.model_name = model_name
    self.temperature = 0.0
    self.top_p = 0.95
    self.max_output_tokens = max_output_tokens
    self.thinking_budget = thinking_budget
    self.system_prompt = SYSTEM_PROMPT
    self.safety_settings = [
      self._types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      self._types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      self._types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      self._types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    ]
    self.output_schema = _build_output_schema(self._types)

  def grade(self, agent_outcome: str) -> Dict[str, Any]:
    user_msg = (agent_outcome or "").strip()
    t = self._types
    contents = [t.Content(role="user", parts=[t.Part.from_text(text=user_msg)])]
    cfg = t.GenerateContentConfig(
      temperature=self.temperature,
      top_p=self.top_p,
      max_output_tokens=self.max_output_tokens,
      safety_settings=self.safety_settings,
      system_instruction=[t.Part.from_text(text=self.system_prompt)],
      thinking_config=t.ThinkingConfig(thinking_budget=self.thinking_budget, include_thoughts=True),
      response_mime_type="application/json",
      response_schema=self.output_schema,
    )
    out = self.client.models.generate_content(model=self.model_name, contents=contents, config=cfg)
    text = (out.text or "").strip()
    try:
      return json.loads(text)
    except Exception:
      s = text[text.find("{") : text.rfind("}") + 1] if ("{" in text and "}" in text) else "{}"
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

  opt = TopTakeawaysAccuracyCheckerOptimizer(
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

