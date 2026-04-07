from google import genai
from google.genai import types
from google.genai.errors import ClientError
import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()

GEMINI_2_5_FLASH_MODEL = "gemini-flash-lite-latest"

OUTPUT_SCHEMA = types.Schema(
    type=types.Type.ARRAY,
    items=types.Schema(
        type=types.Type.OBJECT,
        required=["key", "insight"],
        properties={
            "key": types.Schema(type=types.Type.STRING),
            "insight": types.Schema(type=types.Type.STRING),
        },
    ),
)


SYSTEM_PROMPT = """## Persona

You are Penny, a friendly and positive personal finance advisor for a user.

## Objective

Draft a message to notify the user about important insights on their finances.

## Input
List of insights (and their matching keys) that are each independent of each other.

- `key`: type of insight to be shared to the user
- `insight`: notable activity in the user's finances

## Output

- `key`: type of insight to be shared to the user, which should be exactly as it is in the input
- `insight`: insight rewritten as an actionable message from Penny to the user

### General Guidelines
1. Exclude greeting.
2. Use emojis conversationally.
3. Use the appropriate tone depending on the insight. If the insight is positive (e.g., spending less than forecast, higher income), use an encouraging and happy tone. If negative (e.g., overspending, lower income), use a concerned but helpful tone.
4. Strictly keep to 100 characters only, including spaces and emojis. Remove pieces of information if necessary to follow the character limit.
5. Be mindful when using directional prepositions. Inflows are from an establishment, while outflows are to an establishment.
6. For monetary amounts, use commas, include currency, and exclude decimals.
7. Follow the proper format, depending on the context of the insight.
   - Text can be made green by enclosing it in g{} (ex: g{$1,234})
   - Text can be made green and linked by specifying the category and timeframe (ex: g{[food spending](/food/monthly)})
   - Text can be made red by enclosing it in r{} (ex: r{$1,234})
   - Text can be made red and linked by specifying the category and timeframe (ex: r{[food spending](/food/monthly)})
8. **Categories must always be linked and colored**, except for `uncat_txn` insights where they should be plain text.
9. **Amounts must always be colored** (red for negative/warning, green for positive/good).
10. **Timeframes (weekly/monthly)** must be included in links (e.g., `/food/weekly` or `/food/monthly`). **Link paths must strictly use the words "weekly" or "monthly"**; do not use "this_week" or "this_month" in the URL path.
11. **Tone**: Word messages casually but professionally. **Always spell out words in full** (e.g., use "your" instead of "ur", "transaction" instead of "transaction", "and" instead of "&"). Avoid overly casual interjections like "Whoa" or "Yay"; prefer "Fantastic news" or "Heads up". **Never use symbols like "&" or "+" in the message body.**
12. **Contextual Timeframes**: Insights are snapshots of current performance. **Always use specific time references** like "this week", "last week", "this month", or "last month" in the message body based on the input text. **Do not use generic terms like "weekly" or "monthly" in the message body.** Match the specific timeframe mentioned in the input insight exactly.
13. **Independence of Insights**: Treat each insight in the input list as belonging to a different person. Do not carry over context, facts, or assumptions from one insight to the next. Each output must be self-contained.
14. **Fact-Checking**: Strictly stick to the facts provided in the input. Do not invent details, establishment names, or reasons for spending/income if they are not explicitly mentioned. **Use the establishment name exactly as provided in the input** (e.g., "CITY GENERAL HOSPITAL").
15. **Color Consistency**: The category link and the amount must have the SAME color (both g{} or both r{}). If spending is lower than forecast (good), both are green. If income is lower than forecast (bad), both are red.
16. **No Averages**: Do not describe amounts as "averages" or "usually". Describe them as the actual amount for the specific timeframe mentioned in the input. If the input says "larger than usual", you can mention it is "larger than usual" for this specific instance, but do not imply the current amount is an average.
17. **Inflows in Outflow Categories**: If an insight explicitly mentions an "inflow" for a category that is typically an outflow (e.g., "Dining Out inflow"), treat it as an inflow category (e.g., increasing is good/green, decreasing is bad/red). By default, if no direction is specified, treat outflow categories as outflows.

### `...large_txn` Guidelines
- **Messaging**: Include transaction name and monetary amount, and that it is smaller/larger than usual.
- **Format**
   - Inflow is smaller than usual: monetary amount in red r{$1,234}
   - Outflow is larger than usual: monetary amount in red r{$1,234}
   - Outflow is smaller than usual: monetary amount in green g{$1,234}
   - Inflow is larger than usual: monetary amount in green g{$1,234}
- **Categories in `large_txn`**: Do NOT link or color categories here, focus only on the amount color.

### `...spend_vs_forecast` and `...spent_vs_forecast` Guidelines
- **Messaging**: Include monetary amounts, timeframe (ie. weekly or monthly), and severity of divergence (or synonyms).
   - When referring to category totals, avoid using "higher by X", "up X", "X higher", and anything similar. Instead, use "higher at X", "up at X", "increased to X", and similar phrases. Note that "increased this week to $264" means that the total for the week is $264.
   - Specify if the insight is on an inflow/income or outflow/spending.
   - **Focus on the main category** mentioned in the key (e.g., if key is `...:Food`, focus on "Food" or "Meals") rather than listing every sub-item, to keep it concise.
- **Format**
   - Outflow category is higher than forecasted: category and monetary amount in red, with category linked (ex: r{[Meals](/meals/weekly)} at r{$1,234})
   - Inflow category is lower than forecasted: category and monetary amount in red, with category linked (ex: r{[Income](/income/monthly)} at r{$1,234})
   - Inflow category is higher than forecasted: category and monetary amount in green, with category linked (ex: g{[Income](/income/monthly)} at g{$1,234})
   - Outflow category is lower than forecasted: category and monetary amount in green, with category linked (ex: g{[Meals](/meals/weekly)} at g{$1,234})
- **Linking**: Use the slug in parentheses from the Official Category List for the link path (e.g., `meals_dining_out` becomes `/meals_dining_out/weekly`). Use the display name for the link text. If the category in the key is a general one like "Food", use the closest official slug (e.g., `meals`).
- **Timeframe**: Always include the timeframe in the link (e.g., `/monthly` or `/weekly`) based on the insight text.
- **Color Consistency**: The category link and the amount must have the SAME color (both g{} or both r{}). If spending is lower than forecast (good), both are green. If income is lower than forecast (bad), both are red.

### `...uncat_txn` Guidelines
- **Messaging**: Include transaction name and monetary amount, then ask for confirmation on the suggested category, if any. Ask for how it should be categorized as if there is no suggested category.
- **Format**
   - Transaction is an outflow: monetary amount in red r{$1,234}, category in plain text without color
   - Transaction is an inflow: monetary amount in green g{$1,234}, category in plain text without color
   - **Category names in `uncat_txn` must NOT be linked or colored.**
   - **Avoid using generic phrases** like "that outflow" or "that inflow" if the establishment name is provided. Use the establishment name directly.

### Official Category List

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


class PennyInsightsVerbalizerOptimizer:
    """Handles Gemini API calls for P:PennyInsightsVerbalizerWithLinksJsonKey-style verbalization."""

    def __init__(
        self,
        model_name: str = GEMINI_2_5_FLASH_MODEL,
        thinking_budget: int = 1024,
    ):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set. Set it in .env or environment."
            )
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.thinking_budget = thinking_budget
        # From user_insights_combiner_lib: temp=0.5, top_p=0.95, top_k=40, max_output=4096, json=True
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
        self.output_schema = OUTPUT_SCHEMA

    def generate_response(self, insight_input: list) -> list:
        """
        Verbalize insights using the P:PennyInsightsVerbalizerWithLinksJsonKey prompt.

        Args:
            insight_input: List of dicts with "key" and "insight" (raw insight text).

        Returns:
            List of dicts with "key" and "insight" (verbalized message from Penny).
        """
        if not isinstance(insight_input, list):
            raise ValueError("insight_input must be a list of {key, insight} objects.")
        input_str = json.dumps(insight_input, indent=2)
        request_text = types.Part.from_text(text=f"input:\n{input_str}\n\noutput:")

        contents = [types.Content(role="user", parts=[request_text])]
        generate_content_config = types.GenerateContentConfig(
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
                config=generate_content_config,
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
        print(f"DEBUG: Raw output text: '{output_text}'")
        print(f"DEBUG: Cleaned output text: '{out_clean}'")
        if out_clean.startswith("```"):
            lines = out_clean.split("\n")
            out_clean = "\n".join(lines[1:-1]) if len(lines) > 2 else out_clean
        if out_clean.startswith("```json"):
            lines = out_clean.split("\n")
            out_clean = "\n".join(lines[1:-1]) if len(lines) > 2 else out_clean

        result = json.loads(out_clean)
        if not isinstance(result, list):
            raise ValueError(f"Expected JSON array, got {type(result)}")
        return result


def _run_test_with_logging(
    insight_input: list,
    optimizer: PennyInsightsVerbalizerOptimizer = None,
):
    if optimizer is None:
        optimizer = PennyInsightsVerbalizerOptimizer()

    print("=" * 80)
    print("LLM INPUT (insight_input):")
    print("=" * 80)
    print(json.dumps(insight_input, indent=2))
    print("=" * 80 + "\n")

    result = optimizer.generate_response(insight_input)

    print("=" * 80)
    print("LLM OUTPUT (verbalized):")
    print("=" * 80)
    print(json.dumps(result, indent=2))
    print("=" * 80 + "\n")
    return result


# Batch types for --batch: large_txn, uncat_txn, spend_vs_forecast, mixed
BATCH_TYPES = ("large_txn", "uncat_txn", "spend_vs_forecast", "mixed")

# Test cases covering all different scopes: large_txn, spend_vs_forecast, uncat_txn; grouped by batch type
TEST_CASES = [
    {
        "name": "large_txn_outflow_larger_than_usual_medical",
        "batch": "large_txn",
        "insight_input": [
            {
                "key": "200:large_txn",
                "insight": "Outflow to CITY GENERAL HOSPITAL of $1,250 is larger than usual.",
            }
        ],
        "ideal_response": "One item. key unchanged. insight: actionable message; amount in red r{$1,250}; mention larger than usual; ≤100 chars.",
    },
    {
        "name": "large_txn_outflow_smaller_than_usual_utilities",
        "batch": "large_txn",
        "insight_input": [
            {
                "key": "200:large_txn",
                "insight": "Outflow to PACIFIC GAS & ELECTRIC of $45 is smaller than usual.",
            }
        ],
        "ideal_response": "One item. key unchanged. insight: amount in green g{$45}; smaller than usual; ≤100 chars.",
    },
    {
        "name": "large_txn_inflow_larger_than_usual_bonus",
        "batch": "large_txn",
        "insight_input": [
            {
                "key": "200:large_txn",
                "insight": "Inflow from ANNUAL PERFORMANCE BONUS of $5,000 is larger than usual.",
            }
        ],
        "ideal_response": "One item. key unchanged. insight: amount in green g{$5,000}; larger than usual; ≤100 chars.",
    },
    {
        "name": "large_txn_inflow_smaller_than_usual_dividend",
        "batch": "large_txn",
        "insight_input": [
            {
                "key": "200:large_txn",
                "insight": "Inflow from STOCK DIVIDEND of $12 is smaller than usual.",
            }
        ],
        "ideal_response": "One item. key unchanged. insight: amount in red r{$12}; smaller than usual; ≤100 chars.",
    },
    {
        "name": "spend_vs_forecast_2026_02_transport",
        "batch": "spend_vs_forecast",
        "insight_input": [
            {
                "key": "14:spend_vs_forecast",
                "insight": "Public Transit is significantly up this month at $120. Car and Fuel is slightly up this month at $350. Transport is thus significantly up this month to $470.",
            }
        ],
        "ideal_response": "One item. key unchanged. insight: outflow higher → red category + amount, linked (e.g. r{[Transport](/transportation/monthly)}); up this month to $470; ≤100 chars.",
    },
    {
        "name": "spend_vs_forecast_2026_02_health",
        "batch": "spend_vs_forecast",
        "insight_input": [
            {
                "key": "14:spend_vs_forecast",
                "insight": "Gym and Wellness is significantly down this week at $0. Health is thus significantly down this week to $15.",
            }
        ],
        "ideal_response": "One item. key unchanged. insight: outflow lower → green category + amount, linked (e.g. g{[Health](/health/weekly)}); down this week to $15; ≤100 chars.",
    },
    {
        "name": "spend_vs_forecast_2026_02_sidegig",
        "batch": "spend_vs_forecast",
        "insight_input": [
            {
                "key": "14:spend_vs_forecast",
                "insight": "Sidegig income is significantly up this month at $1,200.",
            }
        ],
        "ideal_response": "One item. key unchanged. insight: inflow higher → green category + amount, linked (e.g. g{[Sidegig](/income_sidegig/monthly)}); up this month to $1,200; ≤100 chars.",
    },
    {
        "name": "spend_vs_forecast_2026_02_shopping",
        "batch": "spend_vs_forecast",
        "insight_input": [
            {
                "key": "14:spend_vs_forecast",
                "insight": "Clothing is significantly up this month at $450. Gadgets is significantly up this month at $800. Shopping is thus significantly up this month to $1,250.",
            }
        ],
        "ideal_response": "One item. key unchanged. insight: outflow higher → red category + amount, linked (e.g. r{[Shopping](/shopping/monthly)}); up this month to $1,250; ≤100 chars.",
    },
    {
        "name": "spend_vs_forecast_batch_transport_health_sidegig_shopping",
        "batch": "spend_vs_forecast",
        "insight_input": [
            {
                "key": "14:spend_vs_forecast",
                "insight": "Public Transit is significantly up this month at $120. Car and Fuel is slightly up this month at $350. Transport is thus significantly up this month to $470.",
            },
            {
                "key": "14:spend_vs_forecast",
                "insight": "Gym and Wellness is significantly down this week at $0. Health is thus significantly down this week to $15.",
            },
            {
                "key": "14:spend_vs_forecast",
                "insight": "Sidegig income is significantly up this month at $1,200.",
            },
            {
                "key": "14:spend_vs_forecast",
                "insight": "Clothing is significantly up this month at $450. Gadgets is significantly up this month at $800. Shopping is thus significantly up this month to $1,250.",
            },
        ],
        "ideal_response": "Four items. 1) Transport: red linked. 2) Health: green linked. 3) Sidegig: green linked. 4) Shopping: red linked. Each ≤100 chars.",
    },
    {
        "name": "uncat_txn_outflow_coffee",
        "batch": "uncat_txn",
        "insight_input": [
            {
                "key": "-100:uncat_txn",
                "insight": "Uncategorized outflow of $6 to BLUE BOTTLE COFFEE. Suggested category: Dining Out.",
            }
        ],
        "ideal_response": "One item. key unchanged. insight: amount in red r{$6}; transaction name; ask for confirmation on suggested category (Dining Out); category in plain text, no color; ≤100 chars.",
    },
    {
        "name": "uncat_txn_inflow_gift",
        "batch": "uncat_txn",
        "insight_input": [
            {
                "key": "-100:uncat_txn",
                "insight": "Uncategorized inflow of $100 from BIRTHDAY GIFT. No suggested category.",
            }
        ],
        "ideal_response": "One item. key unchanged. insight: amount in green g{$100}; transaction name; ask how it should be categorized; ≤100 chars.",
    },
    {
        "name": "mixed_scopes_medical_and_transport",
        "batch": "mixed",
        "insight_input": [
            {
                "key": "200:large_txn",
                "insight": "Outflow to CITY GENERAL HOSPITAL of $1,250 is larger than usual.",
            },
            {
                "key": "14:spend_vs_forecast",
                "insight": "Public Transit is significantly up this month at $120. Transport is thus significantly up this month to $470.",
            },
        ],
        "ideal_response": "Two items. First: large_txn format (amount red). Second: spend_vs_forecast format (red category+amount, linked). Keys preserved; each insight ≤100 chars.",
    },
]


def get_tests_by_batch(batch_type: str):
    """Return list of (index, test_case) for the given batch type (large_txn, uncat_txn, spend_vs_forecast, mixed)."""
    if batch_type not in BATCH_TYPES:
        return []
    return [(i, tc) for i, tc in enumerate(TEST_CASES) if tc.get("batch") == batch_type]


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


def run_test(test_name_or_index_or_dict, optimizer: PennyInsightsVerbalizerOptimizer = None):
    if optimizer is None:
        optimizer = PennyInsightsVerbalizerOptimizer()

    if isinstance(test_name_or_index_or_dict, dict):
        if "insight_input" not in test_name_or_index_or_dict:
            print("Invalid test dict: must contain 'insight_input' key.")
            return None
        name = test_name_or_index_or_dict.get("name", "custom_test")
        print(f"\n{'='*80}\nRunning test: {name}\n{'='*80}\n")
        result = _run_test_with_logging(
            test_name_or_index_or_dict["insight_input"],
            optimizer,
        )
        if test_name_or_index_or_dict.get("ideal_response"):
            print(
                "\n" + "=" * 80 + "\nIDEAL RESPONSE:\n" + "=" * 80 + "\n"
                + test_name_or_index_or_dict["ideal_response"] + "\n" + "=" * 80 + "\n"
            )
        return result

    tc = get_test_case(test_name_or_index_or_dict)
    if tc is None:
        print(f"Test case '{test_name_or_index_or_dict}' not found.")
        return None
    print(f"\n{'='*80}\nRunning test: {tc['name']}\n{'='*80}\n")
    result = _run_test_with_logging(tc["insight_input"], optimizer)
    if tc.get("ideal_response"):
        print(
            "\n" + "=" * 80 + "\nIDEAL RESPONSE:\n" + "=" * 80 + "\n"
            + tc["ideal_response"] + "\n" + "=" * 80 + "\n"
        )
    return result


def run_tests(test_names_or_indices_or_dicts, optimizer: PennyInsightsVerbalizerOptimizer = None):
    if optimizer is None:
        optimizer = PennyInsightsVerbalizerOptimizer()
    if test_names_or_indices_or_dicts is None:
        test_names_or_indices_or_dicts = list(range(len(TEST_CASES)))
    results = []
    for item in test_names_or_indices_or_dicts:
        results.append(run_test(item, optimizer))
    return results


def test_with_inputs(insight_input: list, optimizer: PennyInsightsVerbalizerOptimizer = None):
    if optimizer is None:
        optimizer = PennyInsightsVerbalizerOptimizer()
    return _run_test_with_logging(insight_input, optimizer)


def main(test: str = None, batch: str = None, no_thinking: bool = False):
    thinking_budget = 0 if no_thinking else 1024
    optimizer = PennyInsightsVerbalizerOptimizer(thinking_budget=thinking_budget)

    if batch is not None:
        batch_type = batch.strip().lower()
        indices = [i for i, _ in get_tests_by_batch(batch_type)]
        if not indices:
            print(f"Unknown or empty batch: {batch_type}. Use one of: {', '.join(BATCH_TYPES)}")
            return
        print(f"\n{'='*80}\nRunning batch: {batch_type} ({len(indices)} tests)\n{'='*80}\n")
        for idx, i in enumerate(indices):
            run_test(i, optimizer)
            if idx < len(indices) - 1:
                print("\n" + "-" * 80 + "\n")
        return

    if test is not None:
        if test.strip().lower() == "all":
            print(f"\n{'='*80}\nRunning ALL test cases\n{'='*80}\n")
            for i in range(len(TEST_CASES)):
                run_test(i, optimizer)
                if i < len(TEST_CASES) - 1:
                    print("\n" + "-" * 80 + "\n")
            return
        test_val = int(test) if test.isdigit() else test
        result = run_test(test_val, optimizer)
        if result is None:
            _print_usage()
        return

    _print_usage()


def _print_usage():
    print("Usage:")
    print("  Run a single test: --test <name_or_index>")
    print("  Run all tests: --test all")
    print("  Run by batch: --batch <large_txn|uncat_txn|spend_vs_forecast|mixed>")
    print("  Disable thinking: --no-thinking (thinking_budget=0)")
    print("\nBatches:")
    for bt in BATCH_TYPES:
        count = len(get_tests_by_batch(bt))
        print(f"  {bt}: {count} test(s)")
    print("\nAvailable test cases:")
    for i, tc in enumerate(TEST_CASES):
        print(f"  {i}: [{tc.get('batch', '?')}] {tc['name']}")
    print("  all: run all test cases")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run P:PennyInsightsVerbalizerWithLinksJsonKey optimizer tests")
    parser.add_argument("--test", type=str, help='Test name or index (e.g. "large_txn_outflow_larger_than_usual" or "0")')
    parser.add_argument("--batch", type=str, help="Run all tests in batch: large_txn, uncat_txn, spend_vs_forecast, mixed")
    parser.add_argument("--no-thinking", action="store_true", help="Set thinking_budget=0 (Thinking OFF)")
    args = parser.parse_args()
    main(test=args.test, batch=args.batch, no_thinking=args.no_thinking)
