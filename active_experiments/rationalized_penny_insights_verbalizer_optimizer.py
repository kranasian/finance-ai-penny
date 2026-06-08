"""
Optimizer runner for `P:RationalizedPennyInsightsVerbalizer`.

Run from `finance-ai-penny` repo root:

  python3 active_experiments/rationalized_penny_insights_verbalizer_optimizer.py --test 0
  python3 active_experiments/rationalized_penny_insights_verbalizer_optimizer.py --test all --no-thinking
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import Any, Dict

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

try:
    from google import genai
    from google.genai import types
    from google.genai.errors import ClientError
except Exception:  # pragma: no cover
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]
    ClientError = Exception  # type: ignore[misc,assignment]

if load_dotenv is not None:
    load_dotenv()

GEMINI_2_5_FLASH_MODEL = "gemini-flash-lite-latest"


def _extract_stream_chunk_text(chunk: Any) -> str:
    """Build text from a streaming chunk without reading ``chunk.text`` first.

    ``chunk.text`` triggers noisy SDK warnings when responses mix ``thought_signature``
    or other non-text parts with JSON text (google-genai streaming).
    """
    pieces: list[str] = []
    for cand in getattr(chunk, "candidates", None) or []:
        content = getattr(cand, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "thought", False):
                continue
            t = getattr(part, "text", None)
            if isinstance(t, str) and t:
                pieces.append(t)
    if pieces:
        return "".join(pieces)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*non-text parts in the response.*")
        agg = getattr(chunk, "text", None)
        return agg if isinstance(agg, str) else ""


def _build_output_schema() -> "types.Schema":
    if types is None:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency for Gemini optimizers. Install `google-genai` in this environment."
        )
    return types.Schema(
        type=types.Type.OBJECT,
        required=["insight", "insight_correct"],
        properties={
            "insight": types.Schema(
                type=types.Type.STRING,
                description=(
                    "Single-line rationalized message: g{}/r{} colored link and amount, optional driver detail, "
                    "one trailing emoji; no newlines."
                ),
            ),
            "insight_correct": types.Schema(
                type=types.Type.BOOLEAN,
                description=(
                    "True if # Insight matches # Drivers. False only when drivers contradict facts in the insight "
                    "(e.g. spend exists but insight says $0)."
                ),
            ),
        },
    )


def format_insight_input_for_llm(insight: Dict[str, Any]) -> str:
    """Build Markdown input for the model from structured fields (type, insight, drivers).

    Uses top-level Markdown headings `# Type`, `# Insight`, `# Drivers` (exact spellings).
    """
    for k in ("type", "insight", "drivers"):
        if k not in insight:
            raise ValueError(f"insight missing required key: {k!r}")
    return (
        f"# Type\n\n{insight['type']}\n\n"
        f"# Insight\n\n{insight['insight']}\n\n"
        f"# Drivers\n\n{insight['drivers']}"
    )


# Canonical system prompt for ``P:RationalizedPennyInsightsVerbalizer`` — paste into DB ``penny_templates`` when promoting changes.
SYSTEM_PROMPT = """## Quality gates

- JSON only.
- **Timeframe**: Reuse **# Insight** wording ("this week", "last week", "this month", "last month"). The link uses `/weekly` or `/monthly` matching that period—never swap week ↔ month.
- **Direction**: Keep up/down/higher/lower/exceeded from **# Insight**; do not replace with neutral **"currently $X"** / **"is currently"** as the main framing when **# Insight** already states direction.
- **Banned phrasing:** never use the words **currently** or **is currently** anywhere in `insight`—use **"down to"**, **"so far"**, **"as of early …"**, or explicit direction verbs instead.
- **Emoji**: Exactly one trailing emoji after a space. **Exceeded budget/limit** (goal types, red) → prefer **😟**; false-zero corrections → **⚠️**; uncategorized spikes → **😟**; avoid **💸** for breaches.
- **`*spend*_vs_forecast*` / `*spent*_vs_forecast*` in **# Type** → **forecast** copy (divergence: "down at", "up at", "down to", "lower at")—not budget/limit wording.
- **`*vs_goal*` in **# Type** → **budget/limit** copy ("exceeded your limit", "over your budget")—not standalone forecast phrasing.

## Persona

You are Penny, a friendly personal finance advisor.

## Objective

Compress **# Insight** + **# Drivers** into one rationalized, single-line message in JSON field `insight`.

## Input

Markdown with exactly three top-level headings in order: `# Type`, `# Insight`, `# Drivers`.

## Output

- `insight_correct`: **false** only if **# Drivers** contradicts **# Insight** on facts (e.g. claims $0 but drivers show spend). Early-month context that explains the same figures is **not** a contradiction → **true**.
- `insight`: one line (no newlines). No `Drivers:` prefix.

## Rules

1. No greeting. Facts only from **# Insight** and **# Drivers**; merchant names verbatim when cited.
2. Target ≤320 characters; prefer whole dollars unless amount < $10.
3. **Category link (required):** include **one** primary drill-down as `g{[Display](/slug/weekly)}` or `g{[Display](/slug/monthly)}` (or `r{…}`). **Never** color a bare category name without the markdown link inside the same wrapper (invalid: `r{Groceries}` alone; valid: `r{[Groceries](/meals_groceries/weekly)}`).
4. Color the **main dollar amount** you feature with the **same** `g`/`r` as that link.
5. **Food umbrella** in **# Insight** (e.g. "Food is thus…"): prefer **`[Meals](/meals/weekly)`** or **`[Meals](/meals/monthly)`**. Use a narrower slug (e.g. `meals_dining_out`) only when the sentence is **only** about that subcategory.
6. **`insight_correct` false:** open with drivers' facts (correct totals). For Food umbrella wrong-$0 cases, use **`g{[Meals](/meals/…)}`** and an explicit correction such as **"— not $0"** when appropriate.
7. **Forecast phrasing:** prefer **"down at / up at / down to / lower at / higher at"** plus timeframe words from **# Insight**; include merchants or drivers when space allows.
8. **Goal phrasing:** budget/limit language; map Display from category names in **# Insight** to slugs (e.g. Dining Out → `meals_dining_out`). When **# Insight** contains **"exceeded your limit"** / **"over your budget"**, prefer opening **"You exceeded your budget for …"** with the linked category when it fits.
9. If **# Drivers** lists multiple notable merchants or charges, prefer weaving **at least two** concrete items into `insight` when ≤320 chars allows (names/amounts verbatim).
10. Never tautology on zero: do not write both "$0 spend" and "down to $0"; use **one** **down to g{$0}** (or `r{}`) clause with colored amount.

## Coloring

- `g{…}` / `r{…}`. For typical **outflows**: lower spend vs forecast or good news → **g**; higher / exceeded limit → **r**. Invert for true inflow categories when applicable.

## Slugs (Display→slug, pipe-separated)

Outflows: Meals→meals | Dining Out→meals_dining_out | Delivered Food→meals_delivered_food | Groceries→meals_groceries | Leisure→leisure | Entertainment→leisure_entertainment | Travel and Vacations→leisure_travel | Education→education | Kids Activities→education_kids_activities | Tuition→education_tuition | Transport→transportation | Public Transit→transportation_public | Car and Fuel→transportation_car | Health→health | Medical and Pharmacy→health_medical_pharmacy | Gym and Wellness→health_gym_wellness | Personal Care→health_personal_care | Donations and Gifts→donations_gifts | Uncategorized→uncategorized | Miscellaneous→miscellaneous | Bills→bills | Connectivity→bills_connectivity | Insurance→bills_insurance | Taxes→bills_taxes | Service Fees→bills_service_fees | Shelter→shelter | Home→shelter_home | Utilities→shelter_utilities | Upkeep→shelter_upkeep | Shopping→shopping | Clothing→shopping_clothing | Gadgets→shopping_gadgets | Kids→shopping_kids | Pets→shopping_pets | Transfers→transfers

Inflows: Income→income | Salary→income_salary | Sidegig→income_sidegig | Business→income_business | Interest→income_interest
"""


class RationalizedPennyInsightsVerbalizerOptimizer:
    """Gemini runner to iterate on the P:RationalizedPennyInsightsVerbalizer prompt shape."""

    def __init__(
        self,
        model_name: str = GEMINI_2_5_FLASH_MODEL,
        *,
        thinking_budget: int = 1024,
    ):
        if genai is None or types is None:  # pragma: no cover
            raise RuntimeError(
                "Gemini client dependencies not available. Install `google-genai` (and optionally `python-dotenv`)."
            )
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set. Set it in .env or environment.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.thinking_budget = thinking_budget

        self.temperature = 0.5
        self.top_p = 0.95
        self.top_k = 40
        # Enough for JSON + ≤320-char insight; keeps generation cheap vs 4k ceiling.
        self.max_output_tokens = 1152
        self.safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
        ]

        self.system_prompt = SYSTEM_PROMPT
        self.output_schema = _build_output_schema()

    def generate_response(self, insight_input: str) -> Dict[str, Any]:
        if not isinstance(insight_input, str) or not insight_input.strip():
            raise ValueError(
                "insight_input must be a non-empty string (Markdown with # Type / # Insight / # Drivers)."
            )

        request_text = types.Part.from_text(text=f"input:\n{insight_input.strip()}\n\noutput:")
        contents = [types.Content(role="user", parts=[request_text])]

        cfg = types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_output_tokens=self.max_output_tokens,
            safety_settings=self.safety_settings,
            system_instruction=[types.Part.from_text(text=self.system_prompt)],
            thinking_config=types.ThinkingConfig(
                thinking_budget=self.thinking_budget,
                include_thoughts=True,
            ),
            response_mime_type="application/json",
            response_schema=self.output_schema,
        )

        output_text = ""
        thought_summary = ""
        try:
            for chunk in self.client.models.generate_content_stream(
                model=self.model_name,
                contents=contents,
                config=cfg,
            ):
                piece = _extract_stream_chunk_text(chunk)
                if piece:
                    output_text += piece
                if hasattr(chunk, "candidates") and chunk.candidates:
                    for candidate in chunk.candidates:
                        if hasattr(candidate, "content") and candidate.content:
                            if hasattr(candidate.content, "parts") and candidate.content.parts:
                                for part in candidate.content.parts:
                                    if getattr(part, "thought", False) and getattr(part, "text", None):
                                        t = part.text
                                        thought_summary = (thought_summary + t) if thought_summary else t
        except ClientError as e:
            if self.thinking_budget == 0 and "only works in thinking mode" in (str(e) or ""):
                print(
                    "\n[NOTE] This model requires thinking mode; use default (no --no-thinking) or a different model.",
                    flush=True,
                )
                sys.exit(1)
            raise

        if thought_summary:
            print("\n" + "-" * 80)
            print("THOUGHT SUMMARY:")
            print("-" * 80)
            print(thought_summary.strip())
            print("-" * 80 + "\n")

        if not output_text or not output_text.strip():
            raise ValueError("Empty response from model. Check API key and model availability.")

        out_clean = output_text.strip()
        if out_clean.startswith("```"):
            lines = out_clean.split("\n")
            out_clean = "\n".join(lines[1:-1]) if len(lines) > 2 else out_clean
        if out_clean.startswith("```json"):
            lines = out_clean.split("\n")
            out_clean = "\n".join(lines[1:-1]) if len(lines) > 2 else out_clean

        result = json.loads(out_clean)
        if not isinstance(result, dict):
            raise ValueError(f"Expected JSON object, got {type(result)}")
        return result  # type: ignore[return-value]


TEST_CASES: list[dict[str, Any]] = [
    {
        "name": "spend_vs_forecast_leisure_down",
        "input": {
            "type": "month_spend_vs_forecast",
            "insight": "Leisure is significantly down this month at $11.",
            "drivers": (
                "The significant decrease in leisure spending is due to a reduction in the number and type of "
                "entertainment transactions compared to previous months. In April, you had four leisure transactions "
                "($76.15 total) including streaming subscriptions and a cinema visit (AMC Theatres: $41.68). So far in "
                "May, the only leisure transaction is your monthly Spotify subscription ($10.99 on May 3). March saw "
                "notably higher spending ($584.50) due to a higher volume of transactions."
            ),
        },
        "output": """{
  "insight_correct": true,
  "insight": "g{Leisure is way down to g{$11} at g{[Leisure](/leisure/monthly)}}. April had streaming plus AMC ($41.68); May so far is just Spotify ($11), after a much higher March. 😊"
}""",
    },
    {
        "name": "spend_vs_forecast_leisure_entertainment_zero_early_month",
        "input": {
            "type": "month_spend_vs_forecast",
            "insight": "Entertainment is significantly down this month at $0.",
            "drivers": (
                "The $0 spent on entertainment so far in May is not necessarily a permanent change; it reflects the "
                "fact that no entertainment transactions have posted to your account in the first six days of the month. "
                "In April, you had multiple recurring and one-off charges, including Netflix ($116.40 and $112.80) and "
                "StubHub ($93.90). The drop simply indicates that these regular subscription cycles or discretionary "
                "purchases have not yet occurred or hit your account during the early part of this month."
            ),
        },
        "output": """{
  "insight_correct": true,
  "insight": "g{Entertainment is down at g{$0} this month at g{[Entertainment](/leisure_entertainment/monthly)}}, as usual subscriptions and discretionary purchases have not yet posted early in May. 📆"
}""",
    },
    {
        "name": "spend_vs_forecast_food_weekly_mix",
        "input": {
            "type": "week_spend_vs_forecast",
            "insight": (
                "Dining Out is significantly down this week at $84. Delivered Food is significantly up this week at $91. "
                "Food is thus significantly down this week to $231."
            ),
            "drivers": (
                "Your food spending this week is characterized by a shift toward convenience, despite an overall decline "
                "in total food expenditure compared to last week ($275). While **Dining Out** spending totaled $84 "
                "(e.g., Five Guys: $39, Wendy's: $23, McDonald's: $21), **Delivered Food** (DoorDash, Uber Eats, Grubhub) "
                "reached $91, suggesting that delivery services have overtaken dining out as your primary method for "
                "prepared meals this week."
            ),
        },
        "output": """{
  "insight_correct": true,
  "insight": "Your total g{[Meals](/meals/weekly)} spend is down to g{$231} this week, as Dining Out decreased to $84 (e.g., Five Guys, Wendy's) while Delivered Food rose to $91 via services like DoorDash and Uber Eats. 📉"
}""",
    },
    {
        "name": "spend_vs_forecast_food_week_dining_down_volume",
        "input": {
            "type": "week_spend_vs_forecast",
            "insight": (
                "Dining Out is significantly down this week at $105.  Food is thus significantly down this week to $188."
            ),
            "drivers": (
                "The reduction in Dining Out is primarily due to a lower volume of transactions compared to the previous "
                "week. In the prior week (Apr 26–May 2), you had 6 Dining Out transactions totaling $299, whereas this week "
                "(May 3–9) you had only 2 transactions (Five Guys: $17, Chipotle: $88). The overall Food total is also "
                "lower because you had no Grocery spending this week, compared to $189 at Walmart in the prior week."
            ),
        },
        "output": """{
  "insight_correct": true,
  "insight": "Your g{[Meals](/meals/weekly)} spend is down to g{$188} this week, with dining out at Five Guys ($17) and Chipotle ($88) significantly lower and no grocery spending recorded."
}""",
    },
    {
        "name": "spend_vs_forecast_food_month_zero_vs_actual",
        "input": {
            "type": "month_spend_vs_forecast",
            "insight": (
                "Delivered Food is significantly down this month at $0. Groceries is significantly down this month at $0. "
                "Food is thus significantly down this month to $0."
            ),
            "drivers": (
                "The insight indicating $0 spend is inaccurate. While spending on meals_delivered_food and "
                "meals_groceries has not been recorded yet in May, you have spent $24.00 on food so far this month, "
                "which is categorized as meals_dining_out (Merchant: AM PM Convenience, $24.00). The significant drop "
                "compared to April ($286.07) and March ($345.51) is due to the fact that we are only six days into May, "
                "and you have not yet made your typical recurring grocery or food delivery purchases for the month."
            ),
        },
        "output": """{
  "insight_correct": false,
  "insight": "Your g{[Meals](/meals/monthly)} spend is down at g{$24} so far this month — not $0 — driven by a $24.00 charge at AM PM Convenience. This reflects a significant decrease compared to previous months as you have not yet made your typical grocery or delivery purchases. ⚠️"
}""",
    },
    {
        "name": "spend_vs_forecast_food_zero_accurate_early_may_keeps_direction",
        "input": {
            "type": "month_spend_vs_forecast",
            "insight": (
                "Dining Out is significantly down this month at $0. Groceries is significantly down this month at $0. "
                "Food is thus significantly down this month to $0."
            ),
            "drivers": (
                "The reported $0 spend for May 2026 accurately reflects that there have been no recorded dining out or "
                "grocery transactions processed for the month of May as of May 6th. In contrast, you spent $1,201.43 on "
                "food in April and $2,400.93 in March. The current \"significant drop\" is simply due to this being very "
                "early in the month."
            ),
        },
        "output": """{
  "insight_correct": true,
  "insight": "As of early this month, your g{[Meals](/meals/monthly)} spend is down to g{$0} — not a significant drop, but simply because the month has just begun. ⚠️"
}""",
    },
    {
        "name": "spend_vs_forecast_uncategorized_weekly_not_monthly",
        "input": {
            "type": "week_spend_vs_forecast",
            "insight": "Uncategorized is slightly up this week at $2988.",
            "drivers": (
                "The spike in uncategorized spending this week is driven by a single large transaction: "
                "**Property Group LLC for $2,850.00**. This makes up the vast majority of the $2,988.42 currently listed "
                "as uncategorized. Smaller items contributing to the total include an ATM withdrawal ($100.00) and a "
                "charge at the School Cafeteria ($38.42), both from May 3rd."
            ),
        },
        "output": """{
  "insight_correct": true,
  "insight": "r{[Uncategorized](/uncategorized/weekly)} is up this week at r{$2988}, primarily driven by a $2,850 property payment alongside smaller ATM and school cafeteria charges. 😟"
}""",
    },
    {
        "name": "spend_vs_goal_dining_out_over_budget_penny_style",
        "input": {
            "type": "week_spend_vs_goal",
            "insight": "Dining Out significantly exceeded your limit this week at $79.",
            "drivers": (
                "Your spending in **Dining Out** reached $78.89 for the current week, an increase of $35.19 compared to "
                "last week, with purchases including Olive Garden and Dunkin' Donuts."
            ),
        },
        "output": """{
  "insight_correct": true,
  "insight": "You exceeded your budget for r{[Dining Out](/meals_dining_out/weekly)} this week at r{$79} due to spending at Olive Garden and Dunkin' Donuts. 😟"
}""",
    },
    {
        "name": "spent_vs_goal_groceries_over_limit_costco_publix_applebees",
        "input": {
            "type": "week_spent_vs_goal",
            "insight": "Groceries significantly exceeded your limit last week at $414.",
            "drivers": (
                "The grocery spend was driven primarily by two large trips: Costco: $190.12 and Publix: $163.92. "
                "Additionally, there was a $59.56 transaction at Applebee's that was categorized as meals_groceries. "
                "This appears to be a miscategorization, as dining out typically falls under meals_dining_out. Excluding "
                "this restaurant charge, your grocery spending would have been $354.04, which still exceeds your typical "
                "weekly spend of ~$180–$190 but is significantly closer to your average."
            ),
        },
        "output": """{
  "insight_correct": true,
  "insight": "You blew past your grocery budget at r{[Groceries](/meals_groceries/weekly)} last week at r{$414}: Costco ~$190 and Publix ~$164 carried most of it; Applebee's ~$60 is tagged groceries but reads like dining out — without it you're still ~$354 vs ~$180–190 typical. ⚠️"
}""",
    },
]


def _run_test_with_logging(
    insight_input: Dict[str, Any] | str,
    optimizer: RationalizedPennyInsightsVerbalizerOptimizer | None = None,
):
    if optimizer is None:
        optimizer = RationalizedPennyInsightsVerbalizerOptimizer()

    print("=" * 80)
    if isinstance(insight_input, str):
        input_text = insight_input.strip()
        if not input_text:
            raise ValueError("insight_input string is empty.")
    else:
        input_text = format_insight_input_for_llm(insight_input)
    print("LLM INPUT (markdown):")
    print("=" * 80)
    print(input_text)
    print("=" * 80 + "\n")

    result = optimizer.generate_response(input_text)

    print("=" * 80)
    print("LLM OUTPUT (rationalized verbalized):")
    print("=" * 80)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=" * 80 + "\n")
    return result


def get_test_case(test_name_or_index):
    if isinstance(test_name_or_index, int):
        if 0 <= test_name_or_index < len(TEST_CASES):
            return TEST_CASES[test_name_or_index]
        return None
    if isinstance(test_name_or_index, str):
        for tc in TEST_CASES:
            if tc["name"] == test_name_or_index:
                return tc
        return None
    return None


def run_test(test_name_or_index_or_dict, optimizer: RationalizedPennyInsightsVerbalizerOptimizer | None = None):
    if optimizer is None:
        optimizer = RationalizedPennyInsightsVerbalizerOptimizer()

    if isinstance(test_name_or_index_or_dict, dict):
        di = test_name_or_index_or_dict
        if "input" in di:
            payload = di["input"]
        elif "insight_input" in di:
            payload = di["insight_input"]
        else:
            print("Invalid test dict: must contain 'input' or 'insight_input' key.")
            return None
        name = di.get("name", "custom_test")
        print(f"\n{'='*80}\nRunning test: {name}\n{'='*80}\n")
        result = _run_test_with_logging(payload, optimizer)
        ideal = di.get("output") or di.get("ideal_response")
        if ideal:
            print("\n" + "=" * 80 + "\nIDEAL RESPONSE:\n" + "=" * 80 + "\n" + ideal + "\n" + "=" * 80 + "\n")
        return result

    tc = get_test_case(test_name_or_index_or_dict)
    if tc is None:
        print(f"Test case '{test_name_or_index_or_dict}' not found.")
        return None
    print(f"\n{'='*80}\nRunning test: {tc['name']}\n{'='*80}\n")
    result = _run_test_with_logging(tc["input"], optimizer)
    ideal = tc.get("output") or tc.get("ideal_response")
    if ideal:
        print("\n" + "=" * 80 + "\nIDEAL RESPONSE:\n" + "=" * 80 + "\n" + ideal + "\n" + "=" * 80 + "\n")
    return result


def main(test: str | None = None, *, no_thinking: bool = False):
    thinking_budget = 0 if no_thinking else 1024
    optimizer = RationalizedPennyInsightsVerbalizerOptimizer(thinking_budget=thinking_budget)

    if test is not None:
        if test.strip().lower() == "all":
            print(f"\n{'='*80}\nRunning ALL test cases\n{'='*80}\n")
            for i in range(len(TEST_CASES)):
                run_test(i, optimizer)
                if i < len(TEST_CASES) - 1:
                    print("\n" + "-" * 80 + "\n")
            return
        test_val = int(test) if test.isdigit() else test
        run_test(test_val, optimizer)
        return

    _print_usage()


def _print_usage():
    print("Usage:")
    print("  Run a single test: --test <name_or_index>")
    print("  Run all tests: --test all")
    print("  Disable thinking: --no-thinking (thinking_budget=0)")
    print("\nAvailable test cases:")
    for i, tc in enumerate(TEST_CASES):
        print(f"  {i}: {tc['name']}")
    print("  all: run all test cases")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run P:RationalizedPennyInsightsVerbalizer optimizer tests")
    parser.add_argument("--test", type=str, help='Test name or index (e.g. "spend_vs_forecast_leisure_down" or "0")')
    parser.add_argument("--no-thinking", action="store_true", help="Set thinking_budget=0 (Thinking OFF)")
    args = parser.parse_args()
    main(test=args.test, no_thinking=args.no_thinking)

