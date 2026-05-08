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


def _build_output_schema() -> "types.Schema":
    if types is None:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency for Gemini optimizers. Install `google-genai` in this environment."
        )
    return types.Schema(
        type=types.Type.OBJECT,
        required=["key", "insight", "insight_correct"],
        properties={
            "key": types.Schema(type=types.Type.STRING),
            "insight": types.Schema(
                type=types.Type.STRING,
                description="User-facing message (may be multi-line).",
            ),
            "insight_correct": types.Schema(
                type=types.Type.BOOLEAN,
                description=(
                    "True if input insight agrees with drivers; false if drivers contradict or correct the insight."
                ),
            ),
        },
    )


def format_insight_input_for_llm(insight: Dict[str, Any]) -> str:
    """Build Markdown input for the model from structured fields (key, insight, drivers).

    Uses top-level Markdown headings `# Key`, `# Insight`, `# Drivers` (exact spellings).
    """
    for k in ("key", "insight", "drivers"):
        if k not in insight:
            raise ValueError(f"insight missing required key: {k!r}")
    return (
        f"# Key\n\n{insight['key']}\n\n"
        f"# Insight\n\n{insight['insight']}\n\n"
        f"# Drivers\n\n{insight['drivers']}"
    )


SYSTEM_PROMPT = """## Persona

You are Penny, a friendly and positive personal finance advisor.

## Objective

Turn a single structured insight into a *rationalized* user message: a short headline plus a concrete driver summary.

## Input

Markdown with **exactly three top-level headings** (`#`), in this order (spellings must match):

# Key

…insight identifier (preserve exactly in JSON output `key`).

# Insight

…one-sentence summary (may be improved).

# Drivers

…paragraph explaining what drove the change (may include merchant examples).

## Output

Return a JSON object with:

- `key`: exactly as given under **# Key**
- `insight_correct`: boolean. **true** if the **# Insight** body is consistent with what the **# Drivers** section states; **false** if drivers contradict, correct, or invalidate the insight (e.g. inaccurate totals, missing categories, partial month not reflected in the headline). Judge using **only** those two sections, no outside facts.
- `insight`: one continuous **single-line** rationalized message (no newline characters). Combine (a) a punchy opening that includes the linked+colored category and main amount per Coloring rules, then (b) 1–2 short clauses grounded in **# Drivers**—join with a space or light punctuation, not a line break. If `insight_correct` is false, the opening may reflect the correction implied by **# Drivers** only. **Do not** prefix with "Drivers:".

## Rules

1. No greeting.
2. Do not invent facts beyond the **# Insight** and **# Drivers** bodies.
3. Keep merchant names exactly as provided when you mention them.
4. Prefer dollars without decimals (e.g. $11 not $10.99) unless the input figure is < $10.
5. Be concise: target ≤ 320 characters total per output `insight`.
6. Use only the provided Markdown (no outside context).
7. Rely on the linked category for drill-down to detailed totals.

## Coloring (required)

Use these wrappers exactly:

- Green: `g{...}`
- Red: `r{...}`

**Always color:**
- The linked category (wrap the full link token): `g{[Leisure](/leisure/monthly)}` or `r{[Leisure](/leisure/monthly)}`
- The main dollar amount you cite in the headline clause: `g{$11}` or `r{$11}`

**Color consistency:** the category link and the main amount must use the **same** color.

**Default direction rules:**
- For **spending / outflows**: down/lower is **green**, up/higher is **red**.
- For **income / inflows**: up/higher is **green**, down/lower is **red**.

If the **# Insight** body uses words like "down", "lower", "up", "higher", "exceeded", "within limit", follow that direction.
If direction is unclear, infer from the insight type under **# Key** (`spend_vs_*` / `spent_vs_*` are outflows unless the category is an Inflow category from the list).

## Linking + category slug mapping (required)

Your output must include a linked category for the category referenced under **# Key**.

- **Category link format**: `[Display Name](/slug/weekly)` or `[Display Name](/slug/monthly)`.
- **Timeframe**:
  - If **# Key** includes a `YYYY-MM` segment (e.g. `spend_vs_forecast:2026-05:Leisure`), use `/monthly`.
  - If **# Key** includes a `YYYY-MM-DD` segment (weekly keys), use `/weekly`.
  - Otherwise, infer timeframe from **# Insight** ("this week" → weekly, "this month" → monthly).
- **Slug mapping**: use the slug in parentheses from the Official Category List below. If **# Key** uses a general label (e.g. "Food"), pick the closest official slug (e.g. `meals`).

### Official Category List (Display Name → slug)

#### Outflows
*   Meals (meals)
*   Dining Out (meals_dining_out)
*   Delivered Food (meals_delivered_food)
*   Groceries (meals_groceries)
*   Leisure (leisure)
*   Entertainment (leisure_entertainment)
*   Travel and Vacations (leisure_travel)
*   Education (education)
*   Kids Activities (education_kids_activities)
*   Tuition (education_tuition)
*   Transport (transportation)
*   Public Transit (transportation_public)
*   Car and Fuel (transportation_car)
*   Health (health)
*   Medical and Pharmacy (health_medical_pharmacy)
*   Gym and Wellness (health_gym_wellness)
*   Personal Care (health_personal_care)
*   Donations and Gifts (donations_gifts)
*   Uncategorized (uncategorized)
*   Miscellaneous (miscellaneous)
*   Bills (bills)
*   Connectivity (bills_connectivity)
*   Insurance (bills_insurance)
*   Taxes (bills_taxes)
*   Service Fees (bills_service_fees)
*   Shelter (shelter)
*   Home (shelter_home)
*   Utilities (shelter_utilities)
*   Upkeep (shelter_upkeep)
*   Shopping (shopping)
*   Clothing (shopping_clothing)
*   Gadgets (shopping_gadgets)
*   Kids (shopping_kids)
*   Pets (shopping_pets)
*   Transfers (transfers)

#### Inflows
*   Income (income)
*   Salary (income_salary)
*   Sidegig (income_sidegig)
*   Business (income_business)
*   Interest (income_interest)
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
        self.max_output_tokens = 4096
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
                "insight_input must be a non-empty string (Markdown with # Key / # Insight / # Drivers)."
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
                if chunk.text is not None:
                    output_text += chunk.text
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
        "input": """# Key

spend_vs_forecast:2026-05:Leisure

# Insight

Leisure is significantly down this month at $11.

# Drivers

The significant decrease in leisure spending is due to a reduction in the number and type of entertainment transactions compared to previous months. In April, you had four leisure transactions ($76.15 total) including streaming subscriptions and a cinema visit (AMC Theatres: $41.68). So far in May, the only leisure transaction is your monthly Spotify subscription ($10.99 on May 3). March saw notably higher spending ($584.50) due to a higher volume of transactions.""",
        "output": """{
  "key": "spend_vs_forecast:2026-05:Leisure",
  "insight_correct": true,
  "insight": "g{Leisure is way down to g{$11} at g{[Leisure](/leisure/monthly)}}. April had streaming plus AMC ($41.68); May so far is just Spotify ($11), after a much higher March."
}""",
    },
    {
        "name": "spend_vs_forecast_food_weekly_mix",
        "input": """# Key

spend_vs_forecast:2026-05:Food

# Insight

Dining Out is significantly down this week at $84. Delivered Food is significantly up this week at $91. Food is thus significantly down this week to $231.

# Drivers

Your food spending this week is characterized by a shift toward convenience, despite an overall decline in total food expenditure compared to last week ($275). While **Dining Out** spending totaled $84 (e.g., Five Guys: $39, Wendy's: $23, McDonald's: $21), **Delivered Food** (DoorDash, Uber Eats, Grubhub) reached $91, suggesting that delivery services have overtaken dining out as your primary method for prepared meals this week.""",
        "output": """{
  "key": "spend_vs_forecast:2026-05:Food",
  "insight_correct": true,
  "insight": "g{Food is down to g{$231} at g{[Meals](/meals/weekly)}} with g{delivery at g{$91}} edging out g{dining out at g{$84}}. DoorDash, Uber Eats, and Grubhub made up most prepared meals versus Five Guys, Wendy's, and McDonald's."
}""",
    },
    {
        "name": "spend_vs_forecast_food_week_dining_down_volume",
        "input": """# Key

spend_vs_forecast:2026-05:Food

# Insight

Dining Out is significantly down this week at $105.  Food is thus significantly down this week to $188.

# Drivers

The reduction in Dining Out is primarily due to a lower volume of transactions compared to the previous week. In the prior week (Apr 26–May 2), you had 6 Dining Out transactions totaling $299, whereas this week (May 3–9) you had only 2 transactions (Five Guys: $17, Chipotle: $88). The overall Food total is also lower because you had no Grocery spending this week, compared to $189 at Walmart in the prior week.""",
        "output": """{
  "key": "spend_vs_forecast:2026-05:Food",
  "insight_correct": true,
  "insight": "g{Dining out is down to g{$105} and food to g{$188} at g{[Meals](/meals/weekly)}}. You had two dining trips this week vs six last week ($299); no Walmart groceries vs $189 before."
}""",
    },
    {
        "name": "spend_vs_forecast_food_month_zero_vs_actual",
        "input": """# Key

spend_vs_forecast:2026-05:Food

# Insight

Delivered Food is significantly down this month at $0. Groceries is significantly down this month at $0. Food is thus significantly down this month to $0.

# Drivers

The insight indicating $0 spend is inaccurate. While spending on meals_delivered_food and meals_groceries has not been recorded yet in May, you have spent $24.00 on food so far this month, which is categorized as meals_dining_out (Merchant: AM PM Convenience, $24.00). The significant drop compared to April ($286.07) and March ($345.51) is due to the fact that we are only six days into May, and you have not yet made your typical recurring grocery or food delivery purchases for the month.""",
        "output": """{
  "key": "spend_vs_forecast:2026-05:Food",
  "insight_correct": false,
  "insight": "g{Food spend is g{$24} so far at g{[Meals](/meals/monthly)}} — not $0. Drivers show AM PM Convenience $24; delivered/grocery buckets are quiet early in May vs heavier March and April."
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

