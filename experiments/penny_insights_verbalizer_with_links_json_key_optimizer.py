from google import genai
from google.genai import types
from google.genai.errors import ClientError
import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()

GEMINI_2_5_FLASH_MODEL = "gemini-2.5-flash"

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
3. Use the appropriate tone depending on the insight.
4. Strictly keep to 100 characters only, including spaces and emojis. Remove pieces of information if necessary to follow the character limit.
5. Be mindful when using directional prepositions. Inflows are from an establishment, while outflows are to an establishment.
6. For monetary amounts, use commas, include currency, and exclude decimals.
7. Follow the proper format, depending on the context of the insight.
   - Text can be made green by enclosing it in g{} (ex: g{$1,234})
   - Text can be made green and linked by specifying the category and timeframe (ex: g{[food spending](/food/monthly)})
   - Text can be made red by enclosing it in r{} (ex: r{$1,234})
   - Text can be made red and linked by specifying the category and timeframe (ex: r{[food spending](/food/monthly)})

### `...large_txn` Guidelines
- **Messaging**: Include transaction name and monetary amount, and that it is smaller/larger than usual.
- **Format**
   - Inflow is smaller than usual: monetary amount in red
   - Outflow is larger than usual: monetary amount in red
   - Outflow is smaller than usual: monetary amount in green
   - Inflow is larger than usual: monetary amount in green

### `...spend_vs_forecast` and `...spent_vs_forecast` Guidelines
- **Messaging**: Include monetary amounts, timeframe (ie. weekly or monthly), and severity of divergence (or synonyms).
   - When referring to category totals, avoid using "higher by X", "up X", "X higher", and anything similar. Instead, use "higher at X", "up at X", "increased to X", and similar phrases. Note that "increased this week to $264" means that the total for the week is $264.
   - Specify if the insight is on an inflow/income or outflow/spending.
- **Format**
   - Outflow category is higher than forecasted: category and monetary amount in red, with category linked
   - Inflow category is lower than forecasted: category and monetary amount in red, with category linked
   - Inflow category is higher than forecasted: category and monetary amount in green, with category linked
   - Outflow category is lower than forecasted: category and monetary amount in green, with category linked

### `...uncat_txn` Guidelines
- **Messaging**: Include transaction name and monetary amount, then ask for confirmation on the suggested category, if any. Ask for how it should be categorized as if there is no suggested category.
- **Format**
   - Transaction is an outflow: monetary amount in red, category in plain text without color
   - Transaction is an inflow: monetary amount in green, category in plain text without color

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
        thinking_budget: int = 8196,
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
        "name": "large_txn_outflow_larger_than_usual",
        "batch": "large_txn",
        "insight_input": [
            {
                "key": "2025-02-10:25:large_txn",
                "insight": "Outflow to AMAZON MARKETPLACE of $847 is larger than usual.",
            }
        ],
        "ideal_response": "One item. key unchanged. insight: actionable message, no greeting; include transaction name (e.g. AMAZON MARKETPLACE) and amount; amount in red r{$847}; mention larger than usual; ≤100 chars; commas for amounts ≥1000; no decimals.",
    },
    {
        "name": "large_txn_outflow_smaller_than_usual",
        "batch": "large_txn",
        "insight_input": [
            {
                "key": "2025-02-09:12:large_txn",
                "insight": "Outflow to WHOLE FOODS of $62 is smaller than usual.",
            }
        ],
        "ideal_response": "One item. key unchanged. insight: amount in green g{$62}; smaller than usual; transaction name; ≤100 chars.",
    },
    {
        "name": "large_txn_inflow_larger_than_usual",
        "batch": "large_txn",
        "insight_input": [
            {
                "key": "2025-02-08:7:large_txn",
                "insight": "Inflow from EMPLOYER PAYROLL of $4,200 is larger than usual.",
            }
        ],
        "ideal_response": "One item. key unchanged. insight: amount in green g{$4,200}; inflow from establishment; larger than usual; ≤100 chars.",
    },
    {
        "name": "large_txn_inflow_smaller_than_usual",
        "batch": "large_txn",
        "insight_input": [
            {
                "key": "2025-02-07:3:large_txn",
                "insight": "Inflow from FREELANCE CLIENT of $150 is smaller than usual.",
            }
        ],
        "ideal_response": "One item. key unchanged. insight: amount in red r{$150}; inflow from establishment; smaller than usual; ≤100 chars.",
    },
    {
        "name": "spend_vs_forecast_2026_02_food",
        "batch": "spend_vs_forecast",
        "insight_input": [
            {
                "key": "spend_vs_forecast:2026-02:Food",
                "insight": "Dining Out is significantly down this week at $73.  Groceries is significantly down this week at $67.  Food is thus significantly down this week to $326.",
            }
        ],
        "ideal_response": "One item. key unchanged. insight: outflow category lower → green category + amount, linked (e.g. g{[Food spending](/food/weekly)}); down this week to $326 or similar; timeframe weekly; ≤100 chars.",
    },
    {
        "name": "spend_vs_forecast_2026_02_income",
        "batch": "spend_vs_forecast",
        "insight_input": [
            {
                "key": "spend_vs_forecast:2026-02:Income",
                "insight": "Interest is significantly down this month at $0.  Salary is significantly down this month at $0.  Income is thus down this month to $0.",
            }
        ],
        "ideal_response": "One item. key unchanged. insight: inflow category lower than forecasted → red category + amount, linked (e.g. r{[Income](/income/monthly)}); down this month to $0; ≤100 chars.",
    },
    {
        "name": "spend_vs_forecast_2026_02_leisure",
        "batch": "spend_vs_forecast",
        "insight_input": [
            {
                "key": "spend_vs_forecast:2026-02:Leisure",
                "insight": "Entertainment is significantly down this month at $21.",
            }
        ],
        "ideal_response": "One item. key unchanged. insight: outflow category lower → green category + amount, linked (e.g. g{[Entertainment spending](/leisure_entertainment/monthly)}); down this month at $21; ≤100 chars.",
    },
    {
        "name": "spend_vs_forecast_2026_02_bills",
        "batch": "spend_vs_forecast",
        "insight_input": [
            {
                "key": "spend_vs_forecast:2026-02:Bills",
                "insight": "Insurance is significantly down this month at $0.  Connectivity is significantly down this month at $0.  Bills is thus significantly down this month to $12.",
            }
        ],
        "ideal_response": "One item. key unchanged. insight: outflow category lower → green category + amount, linked (e.g. g{[Bills](/bills/monthly)}); down this month to $12; ≤100 chars.",
    },
    {
        "name": "spend_vs_forecast_2026_02_batch_food_income_leisure_bills",
        "batch": "spend_vs_forecast",
        "insight_input": [
            {
                "key": "spend_vs_forecast:2026-02:Food",
                "insight": "Dining Out is significantly down this week at $73.  Groceries is significantly down this week at $67.  Food is thus significantly down this week to $326.",
            },
            {
                "key": "spend_vs_forecast:2026-02:Income",
                "insight": "Interest is significantly down this month at $0.  Salary is significantly down this month at $0.  Income is thus down this month to $0.",
            },
            {
                "key": "spend_vs_forecast:2026-02:Leisure",
                "insight": "Entertainment is significantly down this month at $21.",
            },
            {
                "key": "spend_vs_forecast:2026-02:Bills",
                "insight": "Insurance is significantly down this month at $0.  Connectivity is significantly down this month at $0.  Bills is thus significantly down this month to $12.",
            },
        ],
        "ideal_response": "Four items. Keys unchanged. 1) Food: green linked category + amount, down this week to $326. 2) Income: red linked (inflow lower). 3) Leisure: green linked, down at $21. 4) Bills: green linked, down to $12. Each ≤100 chars.",
    },
    {
        "name": "uncat_txn_outflow",
        "batch": "uncat_txn",
        "insight_input": [
            {
                "key": "2025-02-11:42:uncat_txn",
                "insight": "Uncategorized outflow of $89 to VENMO JOHN DOE. Suggested category: Transfers.",
            }
        ],
        "ideal_response": "One item. key unchanged. insight: amount in red r{$89}; transaction name; ask for confirmation on suggested category (Transfers) or how to categorize; category in plain text, no color; ≤100 chars.",
    },
    {
        "name": "uncat_txn_inflow",
        "batch": "uncat_txn",
        "insight_input": [
            {
                "key": "2025-02-10:18:uncat_txn",
                "insight": "Uncategorized inflow of $500 from ZELLE REFUND. No suggested category.",
            }
        ],
        "ideal_response": "One item. key unchanged. insight: amount in green g{$500}; transaction name; ask how it should be categorized; ≤100 chars.",
    },
    {
        "name": "mixed_scopes_large_txn_and_spend_vs_forecast",
        "batch": "mixed",
        "insight_input": [
            {
                "key": "2025-02-10:25:large_txn",
                "insight": "Outflow to AMAZON MARKETPLACE of $847 is larger than usual.",
            },
            {
                "key": "25:spend_vs_forecast:Transport",
                "insight": "Transport spending was slightly reduced this month, now at $203.",
            },
        ],
        "ideal_response": "Two items. First: large_txn format (amount red, name, larger than usual). Second: spend_vs_forecast format (green category+amount, linked, lower at $203). Keys preserved; each insight ≤100 chars.",
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
    thinking_budget = 0 if no_thinking else 8196
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
