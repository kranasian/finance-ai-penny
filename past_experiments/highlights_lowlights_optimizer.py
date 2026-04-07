"""
Optimizer for highlights/lowlights text.

Input: list of items with id (1-based index), insight_text, largest_transactions_last_month (list),
largest_transactions_prev_month (list), and optionally largest_transactions_last_week / largest_transactions_prev_week,
and optionally sum_last_month, sum_prev_month, sum_last_week, sum_prev_week (totals for applicable category/categories).
Output: list of { bullet_headline, short_text }, one entry per input item (each entry independent).
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()

# Output: array of objects, one per input item; each has id (same as input), bullet_headline, short_text
OUTPUT_SCHEMA = types.Schema(
    type=types.Type.ARRAY,
    items=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "id": types.Schema(
                type=types.Type.NUMBER,
                description="Same id as the corresponding input item (1-based index).",
            ),
            "bullet_headline": types.Schema(
                type=types.Type.STRING,
                description="One short takeaway phrase (under 10 words); do not repeat in short_text. Markdown allowed. No greeting.",
            ),
            "short_text": types.Schema(
                type=types.Type.STRING,
                description="Explain pattern or likely cause from transactions; must not repeat headline. Longer than bullet_headline. Markdown allowed.",
            ),
        },
        required=["id", "bullet_headline", "short_text"],
    ),
)

SHORT_TEXT_WORD_LIMIT = 50
BULLET_HEADLINE_WORD_LIMIT = 10

# Prompt optimized for: (1) ideal-style without inlining examples, (2) no redundancy,
# (3) one main change + cause from last vs prev transactions, (4) markdown in bullet_headline,
# (5) headline = change only; cause only in short_text, (6) headline may add helpful descriptor; short_text concise. Chosen: iter 2—explicit "one clear sentence" and "name the standout difference".
SYSTEM_PROMPT = f"""**Objective:** For each insight item, output one object with bullet_headline and short_text. No greetings.

**Input:** JSON list. Each item: id (1-based), insight_text, and optionally largest_transactions_last_month/prev_month or largest_transactions_last_week/prev_week (top 3 txns; if there are 3, there may be more). Format: "Merchant for $X.XX." Optional sum_last_month, sum_prev_month, sum_last_week, sum_prev_week: total of transactions for the applicable category/categories (parent = sum of leaf categories; leaf = sum of mentioned leaf categories). Use these sums when helpful for amounts.

**bullet_headline:** The change only—what moved or stayed (e.g. spend down, income down, rent flat; utilities up). Use markdown, e.g. **Food spend down** or **Rent flat; utilities up**. You may add a brief descriptor that helps the user (e.g. magnitude or a second part like "X flat; Y up"). Do not put the cause or "because" in the headline. ≤ {BULLET_HEADLINE_WORD_LIMIT} words.

**short_text:** Focus on the cause. Compare largest_transactions last period vs previous period. Name the standout difference and the likely reason. Mention amounts that matter: from the transaction lists (e.g. one merchant at $X vs another at $Y) and, where it helps, the key amount from insight_text (e.g. the total or main figure for the change, like down to $1200 or down at $1440). You may briefly reference the change if it helps but do not restate the headline in a redundant way. Avoid vague phrases like "shift in spending sources"—be concrete: which merchants, which amounts, what that implies. Derive cause from the lists; do not infer totals or pay-period counts not present (top-3 list length is not total transaction count). ≤ {SHORT_TEXT_WORD_LIMIT} words; must be longer than bullet_headline.

**Rule:** Headline = change (optionally with helpful descriptor). Cause = short_text only. Never put cause in the headline.

**Output:** JSON array of {{ id, bullet_headline, short_text }}, one per input item, same order and id. Escape newlines as \\n. No invented numbers.

**Tone:** Spending down = highlight, up = lowlight. Income down = lowlight, up = highlight."""


def _parse_partial_json_array(raw: str) -> Optional[List[Dict[str, str]]]:
    """Extract list of { bullet_headline, short_text } from possibly truncated JSON array. Returns None if nothing found."""
    pattern = r'\{\s*"bullet_headline"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"short_text"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}'
    matches = re.findall(pattern, raw)
    if not matches:
        return None
    return [{"bullet_headline": m[0].replace("\\n", "\n").replace('\\"', '"'), "short_text": m[1].replace("\\n", "\n").replace('\\"', '"')} for m in matches]


def normalize_insights(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure each item has id 1,2,3,... and largest_transactions_* as lists; pass through sum_* when present."""
    out: List[Dict[str, Any]] = []
    for i, it in enumerate(items, start=1):
        row: Dict[str, Any] = {"id": i, "insight_text": it.get("insight_text", "")}
        for key in ("largest_transactions_last_month", "largest_transactions_prev_month", "largest_transactions_last_week", "largest_transactions_prev_week"):
            val = it.get(key)
            if val is None and key == "largest_transactions_last_month":
                val = it.get("largest_transaction_last_month")
            if val is None and key == "largest_transactions_prev_month":
                val = it.get("largest_transaction_prev_month")
            if val is not None:
                if isinstance(val, list):
                    lst = [str(v) for v in val if v]
                else:
                    lst = [str(val)] if str(val).strip() else []
                if lst:
                    row[key] = lst
        for key in ("sum_last_month", "sum_prev_month", "sum_last_week", "sum_prev_week"):
            val = it.get(key)
            if val is not None and isinstance(val, (int, float)):
                row[key] = round(float(val), 2)
        out.append(row)
    return out


class HighlightsLowlightsOptimizer:
    """Optimizes a list of insight items into a list of { bullet_headline, short_text }, one per item."""

    def __init__(self, model_name: str = "gemini-flash-lite-latest"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set. "
                "Set it in .env or your environment."
            )
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.thinking_budget = 2048
        self.temperature = 0.3
        self.top_p = 0.95
        self.max_output_tokens = 2048
        self.safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
        ]
        self.system_prompt = SYSTEM_PROMPT
        self.output_schema = OUTPUT_SCHEMA

    def optimize(self, items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Convert a list of insight items into a list of { bullet_headline, short_text }, one per item.

        Args:
            items: List of dicts with id, insight_text, and optionally
                largest_transactions_last_month, largest_transactions_prev_month (lists),
                or largest_transactions_last_week, largest_transactions_prev_week. IDs are normalized to 1,2,3,...

        Returns:
            List of dicts with keys bullet_headline and short_text (same length as input).
        """
        items = normalize_insights(items)
        request_text_str = f"""input:\n{json.dumps(items, indent=2)}\n\noutput: """
        request_text = types.Part.from_text(text=request_text_str)
        contents = [types.Content(role="user", parts=[request_text])]
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_output_tokens,
            safety_settings=self.safety_settings,
            system_instruction=[types.Part.from_text(text=self.system_prompt)],
            response_schema=self.output_schema,
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )
        output_text = (response.text or "").strip()
        if not output_text:
            raise ValueError(
                "Empty response from model. Check API key and model availability."
            )

        output_clean = output_text
        if output_clean.startswith("```"):
            first_newline = output_clean.find("\n")
            if first_newline != -1:
                output_clean = output_clean[first_newline + 1 :]
            closing = output_clean.find("```")
            if closing != -1:
                output_clean = output_clean[:closing]
        output_clean = output_clean.strip()
        try:
            result = json.loads(output_clean)
            if not isinstance(result, list):
                result = [result] if isinstance(result, dict) else []
            n = len(items)
            while len(result) < n:
                result.append({"id": items[len(result)]["id"], "bullet_headline": "", "short_text": ""})
            result = result[:n]
            for i in range(len(result)):
                result[i]["id"] = items[i]["id"]
            return result
        except json.JSONDecodeError:
            parsed = _parse_partial_json_array(output_clean)
            if parsed:
                n = len(items)
                while len(parsed) < n:
                    parsed.append({"id": items[len(parsed)]["id"], "bullet_headline": "", "short_text": ""})
                parsed = parsed[:n]
                for i in range(len(parsed)):
                    parsed[i]["id"] = items[i]["id"]
                return parsed
            raise ValueError(
                f"Failed to parse JSON response; response may be truncated.\nResponse text: {output_text}"
            )


def run_optimizer(
    items: List[Dict[str, Any]],
    optimizer: Optional[HighlightsLowlightsOptimizer] = None,
) -> List[Dict[str, str]]:
    """Run the optimizer on a list of insight items. Returns list of { bullet_headline, short_text }."""
    if optimizer is None:
        optimizer = HighlightsLowlightsOptimizer()
    return optimizer.optimize(items)


def _item(
    insight_text: str,
    largest_transactions_last_month: Optional[List[str]] = None,
    largest_transactions_prev_month: Optional[List[str]] = None,
    largest_transactions_last_week: Optional[List[str]] = None,
    largest_transactions_prev_week: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build one input item (id assigned 1-based when list is built). Format: 'Merchant for $X.'"""
    d: Dict[str, Any] = {"insight_text": insight_text}
    if largest_transactions_last_month is not None:
        d["largest_transactions_last_month"] = largest_transactions_last_month
    if largest_transactions_prev_month is not None:
        d["largest_transactions_prev_month"] = largest_transactions_prev_month
    if largest_transactions_last_week is not None:
        d["largest_transactions_last_week"] = largest_transactions_last_week
    if largest_transactions_prev_week is not None:
        d["largest_transactions_prev_week"] = largest_transactions_prev_week
    return d


# Test cases: list of dicts with "name", "items", and optional "ideal_response" (list of { id, bullet_headline, short_text }).
TEST_CASES: List[Dict[str, Any]] = [
    {
        "name": "mixed_food_income_shelter",
        "items": [
            _item(
                "Dining Out is significantly down last month at $606.  Groceries is significantly down last month at $594.  Delivered Food is significantly down last month at $0.  Food is thus significantly down last month to $1200.",
                largest_transactions_last_month=["Giant Eagle for $188.96.", "Costco for $174.72.", "Kroger for $160.23."],
                largest_transactions_prev_month=["Target for $192.95.", "Giant Eagle for $183.58.", "Giant Eagle for $182.37."],
            ),
            _item(
                "Salary is significantly down last month at $1440.",
                largest_transactions_last_month=["CA State Payroll for $1,440.00.", "Savings Interest for $3.34."],
                largest_transactions_prev_month=["CA State Payroll for $1,440.00.", "CA State Payroll for $1,440.00.", "Savings Interest for $1.67."],
            ),
            _item(
                "Utilities is slightly up last month at $377.",
                largest_transactions_last_month=["Mid-Carolina Electric Cooperative for $146.15.", "Dominion Energy for $131.35.", "Joint Municipal Water and Sewer Commission for $99.20."],
                largest_transactions_prev_month=["Mid-Carolina Electric Cooperative for $141.31.", "Dominion Energy for $121.08.", "Joint Municipal Water and Sewer Commission for $99.38."],
            ),
        ],
        "ideal_response": [
            {"id": 1, "bullet_headline": "Food spend down", "short_text": "Your total food spending was significantly down last month to $1200. One less Giant Eagle at the $180 range compared to the previous month likely explains why."},
            {"id": 2, "bullet_headline": "Income down", "short_text": "Salary was down last month. You had two CA State Payroll runs vs three in the previous month."},
            {"id": 3, "bullet_headline": "Utilities up", "short_text": "Shelter spend was slightly up. Mid-Carolina Electric and Dominion Energy each up a bit—likely usage or rate, not a structural change."},
        ],
    },
    {
        "name": "mixed_leisure_shopping_income",
        "items": [
            _item(
                "Entertainment is significantly down last month at $240.  Leisure is thus significantly down last month to $240.",
                largest_transactions_last_month=["Regal Cinemas for $121.63.", "Cinemark for $119.32.", "Fandango for $62.87."],
                largest_transactions_prev_month=["Fandango for $65.28.", "Fandango for $63.46.", "Spotify for $10.99."],
            ),
            _item(
                "Kids is significantly down last month at $0.  Clothing is significantly down last month at $0.",
                largest_transactions_last_month=[],
                largest_transactions_prev_month=["Kohl's for $25.46."],
            ),
            _item(
                "Salary is significantly down last month at $1440.",
                largest_transactions_last_month=["CA State Payroll for $1,440.00.", "Savings Interest for $3.34."],
                largest_transactions_prev_month=["CA State Payroll for $1,440.00.", "CA State Payroll for $1,440.00.", "Savings Interest for $1.67."],
            ),
        ],
        "ideal_response": [
            {"id": 1, "bullet_headline": "**Leisure spend down; more theater, less streaming.**", "short_text": "Entertainment dropped to $240. Last month went to Regal and Cinemark; previous month was mostly Fandango and Spotify—likely fewer streaming subs or a shift to in-theater spending."},
            {"id": 2, "bullet_headline": "**Kids and Clothing down**", "short_text": "Kids and Clothing at $0. No transactions last month compared to Kohl's ($25) previous month."},
            {"id": 3, "bullet_headline": "**Paycheck unchanged; fewer pay periods.**", "short_text": "Income dropped because you had one fewer payroll run (1 vs 2)."},
        ],
    },
    {
        "name": "highlights_only_food_education_leisure",
        "items": [
            _item(
                "Dining Out is significantly down last month at $606.  Groceries is significantly down last month at $594.  Delivered Food is significantly down last month at $0.  Food is thus significantly down last month to $1200.",
                largest_transactions_last_month=["Giant Eagle for $188.96.", "Costco for $174.72.", "Kroger for $160.23."],
                largest_transactions_prev_month=["Target for $192.95.", "Giant Eagle for $183.58.", "Giant Eagle for $182.37."],
            ),
            _item(
                "Tuition received refunds last month, totaling $1175.",
                largest_transactions_last_month=["Genentech Salary for $1,744.62.", "Daycare Center for $500.00.", "Music School for $101.32."],
                largest_transactions_prev_month=["Genentech Salary for $1,744.62.", "Genentech Salary for $1,744.62.", "After School Program for $500.00."],
            ),
            _item(
                "Entertainment is significantly down last month at $240.  Leisure is thus significantly down last month to $240.",
                largest_transactions_last_month=["Regal Cinemas for $121.63.", "Cinemark for $119.32.", "Fandango for $62.87."],
                largest_transactions_prev_month=["Fandango for $65.28.", "Fandango for $63.46.", "Spotify for $10.99."],
            ),
        ],
        "ideal_response": [
            {"id": 1, "bullet_headline": "Food spend down", "short_text": "Your total food spending was significantly down last month to $1200. One less Giant Eagle at the $180 range compared to the previous month likely explains why."},
            {"id": 2, "bullet_headline": "**Education refunds; payroll in category.**", "short_text": "Tuition refunds totaling $1175 mostly from Genentech Salary."},
            {"id": 3, "bullet_headline": "**Leisure spend down; more theater, less streaming.**", "short_text": "Entertainment dropped to $240. Last month went to Regal and Cinemark; previous month was mostly Fandango and Spotify—likely fewer streaming subs or a shift to in-theater spending."},
        ],
    },
    {
        "name": "lowlights_only_income_shelter",
        "items": [
            _item(
                "Salary is significantly down last month at $1440.",
                largest_transactions_last_month=["CA State Payroll for $1,440.00.", "Savings Interest for $3.34."],
                largest_transactions_prev_month=["CA State Payroll for $1,440.00.", "CA State Payroll for $1,440.00.", "Savings Interest for $1.67."],
            ),
            _item(
                "Utilities is slightly up last month at $377.",
                largest_transactions_last_month=["Property Group LLC for $2,850.00.", "Mid-Carolina Electric Cooperative for $146.15.", "Dominion Energy for $131.35."],
                largest_transactions_prev_month=["Property Group LLC for $2,850.00.", "Mid-Carolina Electric Cooperative for $141.31.", "Dominion Energy for $121.08."],
            ),
        ],
        "ideal_response": [
            {"id": 1, "bullet_headline": "**Salary down.**", "short_text": "Income dropped because you had one fewer payroll run (1 vs 2) from CA State Payroll."},
            {"id": 2, "bullet_headline": "Utilities up", "short_text": "Shelter spend was slightly up. Mid-Carolina Electric and Dominion Energy each up a bit—likely usage or rate, not a structural change."},
        ],
    },
    {
        "name": "mixed_monthly_weekly_and_lowlight",
        "items": [
            _item(
                "Car & Fuel is significantly down last month at $0.",
                largest_transactions_last_month=["QuikTrip for $68.65.", "QuikTrip for $60.10.", "Exxon for $47.36."],
                largest_transactions_prev_month=["Auto Zone for $562.12.", "QuikTrip for $68.28.", "Exxon for $65.92."],
            ),
            _item(
                "Bills is significantly down last week at $0.",
                largest_transactions_last_week=[],
                largest_transactions_prev_week=["AT&T Wireless for $88.43.", "AT&T Internet for $62.69."],
            ),
            _item(
                "Medical & Pharmacy is significantly down last week at $0.",
                largest_transactions_last_week=[],
                largest_transactions_prev_week=["CVS for $49.12."],
            ),
        ],
        "ideal_response": [
            {"id": 1, "bullet_headline": "**Car & Fuel down; no big repair.**", "short_text": "Spend dropped vs last month. Previous month had Auto Zone for $562; this month only QuikTrip and Exxon for fuel—the drop is likely from no one-off repair/maintenance, not less driving."},
            {"id": 2, "bullet_headline": "**Bills quiet last week.**", "short_text": "Bills were $0 last week. Previous week had AT&T Wireless and AT&T Internet."},
            {"id": 3, "bullet_headline": "**Health spend quiet last week.**", "short_text": "Medical & Pharmacy was $0 last week. Previous week had CVS ($49); no pharmacy or medical charge in the latest week."},
        ],
    },
]


def get_test_case(test_name_or_index: Any) -> Optional[Dict[str, Any]]:
    """Get a test case by name or index. Returns the test case dict or None if not found."""
    if isinstance(test_name_or_index, int):
        if 0 <= test_name_or_index < len(TEST_CASES):
            return TEST_CASES[test_name_or_index]
        return None
    if isinstance(test_name_or_index, str):
        for tc in TEST_CASES:
            if tc.get("name") == test_name_or_index:
                return tc
        return None
    return None


def _run_case(
    name: str,
    items: List[Dict[str, Any]],
    optimizer: HighlightsLowlightsOptimizer,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Run one test case; return result list. If verbose, prints name, INPUT, OUTPUT, and summary."""
    result = optimizer.optimize(items)
    if verbose:
        print(f"  {name}")
        print("  INPUT:")
        print(json.dumps(normalize_insights(items), indent=2))
        print("\n  OUTPUT:")
        print(json.dumps(result, indent=2))
        n = len(result)
        total_words = sum(
            len((e.get("short_text") or "").split()) + len((e.get("bullet_headline") or "").split())
            for e in result
        )
        print(f"  {name}: ran ({n} entries, words≈{total_words})")
    return result


def run_test(test_name_or_index_or_dict: Any, optimizer: Optional[HighlightsLowlightsOptimizer] = None) -> Optional[List[Dict[str, Any]]]:
    """
    Run a single test by name, index, or by passing test data directly.

    Args:
        test_name_or_index_or_dict: One of:
            - Test case name (str)
            - Test case index (int)
            - Test data dict: {"items": [...], "name": "custom_test", "ideal_response": [...] (optional)}
        optimizer: Optional HighlightsLowlightsOptimizer. If None, creates a new one.

    Returns:
        List of { id, bullet_headline, short_text }, or None if test not found.
    """
    if optimizer is None:
        optimizer = HighlightsLowlightsOptimizer()

    if isinstance(test_name_or_index_or_dict, dict):
        if "items" not in test_name_or_index_or_dict:
            print("Invalid test dict: must contain 'items' key.")
            return None
        test_name = test_name_or_index_or_dict.get("name", "custom_test")
        print("\n" + "=" * 80)
        print(f"Running test: {test_name}")
        print("=" * 80 + "\n")
        result = _run_case(
            test_name,
            test_name_or_index_or_dict["items"],
            optimizer,
            verbose=True,
        )
        if test_name_or_index_or_dict.get("ideal_response"):
            ideal = test_name_or_index_or_dict["ideal_response"]
            ideal_str = json.dumps(ideal, indent=2) if isinstance(ideal, list) else str(ideal)
            print("\n" + "=" * 80 + "\nIDEAL RESPONSE:\n" + "=" * 80 + "\n" + ideal_str + "\n" + "=" * 80 + "\n")
        return result

    test_case = get_test_case(test_name_or_index_or_dict)
    if test_case is None:
        print(f"Test case '{test_name_or_index_or_dict}' not found.")
        return None

    print("\n" + "=" * 80)
    print(f"Running test: {test_case['name']}")
    print("=" * 80 + "\n")
    result = _run_case(
        test_case["name"],
        test_case["items"],
        optimizer,
        verbose=True,
    )
    if test_case.get("ideal_response"):
        ideal_str = json.dumps(test_case["ideal_response"], indent=2)
        print("\n" + "=" * 80 + "\nIDEAL RESPONSE:\n" + "=" * 80 + "\n" + ideal_str + "\n" + "=" * 80 + "\n")
    return result


def run_tests(test_names_or_indices_or_dicts: Optional[List[Any]] = None, optimizer: Optional[HighlightsLowlightsOptimizer] = None) -> List[Optional[List[Dict[str, Any]]]]:
    """
    Run multiple tests by names, indices, or by passing test data dicts.

    Args:
        test_names_or_indices_or_dicts: None to run all from TEST_CASES, or list of names (str), indices (int), or dicts with "items".
        optimizer: Optional optimizer. If None, creates a new one.

    Returns:
        List of result lists (or None for not found).
    """
    if test_names_or_indices_or_dicts is None:
        test_names_or_indices_or_dicts = list(range(len(TEST_CASES)))
    results = []
    for test_item in test_names_or_indices_or_dicts:
        result = run_test(test_item, optimizer)
        results.append(result)
    return results


def test_with_inputs(items: List[Dict[str, Any]], optimizer: Optional[HighlightsLowlightsOptimizer] = None) -> List[Dict[str, Any]]:
    """Run optimizer with a custom list of items (no test case). Returns the result list."""
    if optimizer is None:
        optimizer = HighlightsLowlightsOptimizer()
    return optimizer.optimize(items)


def test_sample(optimizer: Optional[HighlightsLowlightsOptimizer] = None) -> List[Dict[str, Any]]:
    """Test with the first test case items (mixed_food_income_shelter)."""
    items = TEST_CASES[0]["items"]
    print("INPUT:")
    print(json.dumps(normalize_insights(items), indent=2))
    print("\n" + "=" * 80)
    result = run_optimizer(items, optimizer)
    print("\nOUTPUT:")
    print(json.dumps(result, indent=2))
    return result


def main(test: Optional[str] = None) -> None:
    """Main: run single test (name or index), all tests, or show usage. --input overrides and runs custom items."""
    import argparse
    parser = argparse.ArgumentParser(description="Run highlights/lowlights optimizer tests")
    parser.add_argument("--test", type=str, help='Test name or index (e.g. "mixed_food_income_shelter" or "0")')
    parser.add_argument("--input", type=str, default=None, help="JSON file path or JSON string of items list. If set, runs this instead of test cases.")
    parser.add_argument("--model", type=str, default="gemini-flash-lite-latest", help="Gemini model name.")
    args = parser.parse_args()

    opt = HighlightsLowlightsOptimizer(model_name=args.model)

    if args.input is not None:
        raw = args.input.strip()
        if raw.startswith("["):
            items = json.loads(raw)
        else:
            with open(raw, "r") as f:
                items = json.load(f)
        items = normalize_insights(items)
        print("INPUT:")
        print(json.dumps(items, indent=2))
        print("\n" + "=" * 80)
        result = opt.optimize(items)
        print("OUTPUT:")
        print(json.dumps(result, indent=2))
        return

    test_arg = test or args.test
    if test_arg is not None:
        if test_arg.strip().lower() == "all":
            print("\n" + "=" * 80)
            print("Running ALL test cases")
            print("=" * 80 + "\n")
            for i in range(len(TEST_CASES)):
                run_test(i, opt)
                if i < len(TEST_CASES) - 1:
                    print("\n" + "-" * 80 + "\n")
            return
        test_val = int(test_arg) if test_arg.isdigit() else test_arg
        result = run_test(test_val, opt)
        if result is None:
            print("\nAvailable test cases:")
            for i, tc in enumerate(TEST_CASES):
                print(f"  {i}: {tc['name']}")
            print("  all: run all test cases")
        return

    print("Usage:")
    print("  Run a single test: --test <name_or_index>")
    print("  Run all tests: --test all")
    print("  Custom input: --input <path_or_json_string>")
    print("  Model: --model <model_name>")
    print("\nAvailable test cases:")
    for i, tc in enumerate(TEST_CASES):
        print(f"  {i}: {tc['name']}")
    print("  all: run all test cases")


if __name__ == "__main__":
    main()
