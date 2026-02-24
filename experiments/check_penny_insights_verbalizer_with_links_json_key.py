from google import genai
from google.genai import types
import os
import sys
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are a checker verifying the output of a personal finance verbalizer (PennyInsightsVerbalizer).

## Input:
- **EVAL_INPUT**: A JSON array of raw insights (factual source of truth).
- **PAST_REVIEW_OUTCOMES**: An array of past review outcomes.
- **REVIEW_NEEDED**: The JSON output from the verbalizer (array of verbalized messages). 

## Correspondence Rule:
The insights in EVAL_INPUT correspond to the messages in REVIEW_NEEDED based on **order** (i.e., the first EVAL_INPUT insight is for the first REVIEW_NEEDED message, etc.). The `key` field signifies the topic and is not unique.

## Output:
**Return valid JSON ONLY.** Do not include markdown formatting, code blocks, or any text outside the JSON object.
Example format:
{"good_copy": true, "info_correct": true, "eval_text": ""}

- `good_copy`: True if REVIEW_NEEDED follows all **Formatting rules** (Part 2 & 3). This includes font colors, linking syntax, monetary formatting, and character limits. Phrasing should be appropriate (friendly openers and emojis are encouraged but not strictly required).
- `info_correct`: True if the **factual information** actually present in each message in REVIEW_NEEDED matches its corresponding insight in EVAL_INPUT (based on order). **STRICT RULE: NEVER mark info_correct: false for omitted information.** If EVAL_INPUT has details (like sub-categories, extra context, or aggregate totals) that are not in REVIEW_NEEDED, you MUST ignore the omission. Only mark `false` if the information that *is* present is factually wrong (e.g., wrong amount, wrong category name, wrong direction).
- `eval_text`: **MUST be an empty string "" if both good_copy and info_correct are True.** Otherwise, list each error. **Each line must start with "Insight <number>: "** (e.g., "Insight 1: ...", "Insight 2: ...") based on the order of the items. Use quick, concise phrases (max 15 words per error). Do not refer to rule numbers, labels like "Rule 1", or internal section names. Explain the error clearly so it is understandable independently. One line per erroneous item, separate with newline (`\\n`). Do not reference PAST_REVIEW_OUTCOMES. **STRICTLY FORBIDDEN: Do not include any internal reasoning, "Verification Steps", or "Re-evaluating" text in the output. The output must be valid JSON ONLY.**

## Strict Negative Constraints (DO NOT FLAG THESE AS ERRORS)
1. **Omitted Information**: Do NOT flag missing sub-categories, missing totals, or missing details from EVAL_INPUT as errors.
2. **Character Count**: Do NOT flag character count errors unless the visible text (excluding markup like `g{`, `r{`, `}`, `[]`, `()`) is strictly over 100 characters.
3. **Commas**: Do NOT flag comma errors if the amount already has a comma (e.g., `$1,000`).
4. **Friendly Openers**: Do NOT flag missing openers or emojis as errors; they are optional.

## Critical Priority: Learn from PAST_REVIEW_OUTCOMES
Use PAST_REVIEW_OUTCOMES as a knowledge base. If REVIEW_NEEDED repeats mistakes flagged in past outcomes, mark incorrect. Do not mention past outcomes in `eval_text`.

## Verification Steps (Internal Only - Do Not Output)
1. Check PAST_REVIEW_OUTCOMES: if REVIEW_NEEDED repeats past mistakes → mark False.
2. Verify good_copy: valid JSON array; each item follows formatting rules (Part 2 & 3); link timeframe matches text timeframe. **For `spend_vs_forecast`, ensure both category and amount are colored.**
3. Verify info_correct: For each item, check **only** that the factual information in REVIEW_NEEDED matches its corresponding insight in EVAL_INPUT (based on order). **Ignore any omitted information from EVAL_INPUT.** Ensure amounts are absolute values (e.g., "at $X", "to $X") and NOT relative (e.g., "$X higher").
4. eval_text: **Only when REVIEW_NEEDED is incorrect.** Every line must start with "Insight " then the item number (1, 2, 3...), then ": ". If REVIEW_NEEDED is correct, eval_text must be empty. **Be extremely careful not to flag correct items as incorrect.** Descriptions must be concise (max 15 words).

## Rules (apply to judge good_copy and info_correct)

### Part 1: Content Rules (info_correct)
1. **Factual Accuracy**: All information in REVIEW_NEEDED must match EVAL_INPUT.
2. **Required Info**:
    * `...large_txn`: Include name, amount, and larger/smaller.
    * `...spend_vs_forecast`: Include category, amount, higher/lower, and timeframe (weekly/monthly).
    * `...uncat_txn`: Include name, amount, and **must ask the user** for the category. IF a suggested category exists in EVAL_INPUT, ask for confirmation; otherwise, ask for the category.
3. **Category Asking**: ONLY ask the user for a transaction's category if the `key` includes "uncat_txn".
4. **Absolute Amounts**: Amounts must be communicated as absolute values and not relative (e.g., "higher at $100" or "increased to $100" is CORRECT; "$100 higher" or "up by $100" is WRONG).
5. **Prepositions**: Inflows *from* establishment, outflows *to* establishment.

            ### Part 2: Formatting Rules (good_copy)
            1. **Colors**:
                * **GREEN `g{...}`**: spending lower/expected, income higher/expected, uncategorized **inflow**; for `...large_txn`: **outflow smaller** than usual, **inflow larger** than usual.
                * **RED `r{...}`**: spending higher/expected, income lower/expected, uncategorized **outflow**; for `...large_txn`: **outflow larger** than usual, **inflow smaller** than usual.
                * **Forecast Insights**: For `...spend_vs_forecast`, **both the category and the amount** must be colored (same color per insight).
2. **Links**: `...spend_vs_forecast` category MUST be linked and colored (e.g., `g{[Food](/food/weekly)}`). The timeframe in the link (e.g., `/monthly`) MUST match the timeframe in the text.
3. **Amounts**: Use `$`, no decimals. **Commas only when absolute value is 1,000 or more** (e.g., `$999` no comma; `$1,000` with comma).
4. **Syntax**: Balanced `[]`, `{}`, `()`.
5. **Persona**: Friendly openers (e.g., "Note:", "Heads up!", "Great news!", "Awesome!", "Nice!", "Good job!") and emojis are encouraged.

### Part 3: Character Count Rule (good_copy)
1. **Strict Limit**: 100 characters maximum per insight. Count only **visible text and spaces**; exclude markup (`g{`, `r{`, `}`, `[]`, `()`, link syntax, and URL).
"""

class CheckJsonOptimizer:
  """Handles all Gemini API interactions for checking evaluation of insights against rules"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for checking evaluations"""
    # API Configuration
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file or environment.")
    self.client = genai.Client(api_key=api_key)
    
    # Model Configuration
    self.thinking_budget = 1024
    self.model_name = model_name
    
    # Generation Configuration Constants
    self.temperature = 0.6
    self.top_p = 0.95
    self.max_output_tokens = 8192
    
    # Safety Settings
    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
    ]
    
    # System Prompt
    self.system_prompt = SYSTEM_PROMPT

  
  def generate_response(self, eval_input: list, past_review_outcomes: list, review_needed: list) -> list:
    """
    Generate a response using Gemini API for checking evaluations.
    
    Args:
      eval_input: The original input items to be evaluated. An array of items with `key` and `insight` fields.
      past_review_outcomes: An array of past review outcomes, each containing `output`, `good_copy`, `info_correct`, and `eval_text`.
      review_needed: An array of items that need to be reviewed. Each item is an object with `key` and `insight` fields.
      
    Returns:
      List of dictionaries, each with good_copy, info_correct, and eval_text keys
    """
    # Create request text with the new input structure
    past_review_section = ""
    if past_review_outcomes:
      past_review_section = f"""<PAST_REVIEW_OUTCOMES>
{json.dumps(past_review_outcomes, indent=2)}
</PAST_REVIEW_OUTCOMES>

"""
    
    request_text = types.Part.from_text(text=f"""<EVAL_INPUT>
{json.dumps(eval_input, indent=2)}
</EVAL_INPUT>

{past_review_section}<REVIEW_NEEDED>
{json.dumps(review_needed, indent=2)}
</REVIEW_NEEDED>

Output:""")
    
    # Create content and configuration
    contents = [types.Content(role="user", parts=[request_text])]
    
    generate_content_config = types.GenerateContentConfig(
      temperature=self.temperature,
      top_p=self.top_p,
      max_output_tokens=self.max_output_tokens,
      safety_settings=self.safety_settings,
      system_instruction=[types.Part.from_text(text=self.system_prompt)],
      thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
    )

    # Generate response
    output_text = ""
    for chunk in self.client.models.generate_content_stream(
      model=self.model_name,
      contents=contents,
      config=generate_content_config,
    ):
      if chunk.text is not None:
        output_text += chunk.text
    
    # Check if response is empty
    if not output_text or not output_text.strip():
      raise ValueError(f"Empty response from model. Check API key and model availability.")
    
    # Parse JSON response
    try:
      # Remove markdown code blocks if present
      if "```json" in output_text:
        json_start = output_text.find("```json") + 7
        json_end = output_text.find("```", json_start)
        if json_end != -1:
          output_text = output_text[json_start:json_end].strip()
      elif "```" in output_text:
        # Try to find JSON in code blocks
        json_start = output_text.find("```") + 3
        json_end = output_text.find("```", json_start)
        if json_end != -1:
          output_text = output_text[json_start:json_end].strip()
      
      # Try parsing the whole response first
      parsed = json.loads(output_text.strip())
      
      # Return the response as-is (could be dict, list, etc.)
      return parsed
    except json.JSONDecodeError as e:
      raise ValueError(f"Failed to parse JSON response: {e}\\nResponse length: {len(output_text)}\\nResponse preview: {output_text[:500]}")


def test_with_inputs(eval_input: list, past_review_outcomes: list, review_needed: list, checker: CheckJsonOptimizer = None):
  """
  Convenient method to test the checker optimizer with custom inputs.
  
  Args:
    eval_input: The original input items to be evaluated. An array of items with `key` and `insight` fields.
    past_review_outcomes: An array of past review outcomes, each containing `output`, `good_copy`, `info_correct`, and `eval_text`.
    review_needed: An array of items that need to be reviewed. Each item is an object with `key` and `insight` fields.
    checker: Optional CheckJsonOptimizer instance. If None, creates a new one.
    
  Returns:
    List of dictionaries, each with good_copy, info_correct, and eval_text keys
  """
  if checker is None:
    checker = CheckJsonOptimizer()
  
  return checker.generate_response(eval_input, past_review_outcomes, review_needed)


# Test cases covering different scenarios
TEST_CASES = [
  {
    "name": "spend_vs_forecast_no_color_no_link",
    "eval_input": [
      {
        "key": "spend_vs_forecast:2026-02:Dining",
        "insight": "Dining Out spending increased this week to $150."
      }
    ],
    "review_needed": [
      {
        "key": "spend_vs_forecast:2026-02:Dining",
        "insight": "Heads up! Your Dining Out spending was higher at $150 this week. 🍔"
      }
    ],
    "past_review_outcomes": []
  },
  {
    "name": "spend_vs_forecast_missing_info_and_formatting",
    "eval_input": [
      {
        "key": "44:spend_vs_forecast",
        "insight": "Shopping spending slightly decreased this month to $219, with Gadgets increasing to $131 and Pets decreasing to $62."
      }
    ],
    "review_needed": [
      {
        "key": "44:spend_vs_forecast",
        "insight": "Your Shopping spending is lower at $219 this month. Gadgets is up at $131."
      }
    ],
    "past_review_outcomes": []
  },
  {
    "name": "provided_example_1_mixed",
    "eval_input": [
      {
        "key": "44:spend_vs_forecast",
        "insight": "Shopping spending significantly decreased this month, now at $49.\nThese are all compared to the forecasts based on average spending for this time.\n"
      },
      {
        "key": "44:spend_vs_forecast",
        "insight": "Shopping spending slightly decreased this month to $219, with Gadgets increasing to $131 and Pets decreasing to $62.\nThese are all compared to the forecasts based on average spending for this time.\n"
      }
    ],
    "review_needed": [
      {
        "key": "44:spend_vs_forecast",
        "insight": "Great news! Your g{[Shopping spending](/shopping/monthly)} is lower at g{$49} this month! 🥳"
      },
      {
        "key": "44:spend_vs_forecast",
        "insight": "Your Shopping spending is lower at $219 this month. Gadgets is up at $131."
      }
    ],
    "past_review_outcomes": []
  },
  {
    "name": "provided_example_2_factual_mismatch",
    "eval_input": [
      {
        "key": "spend_vs_forecast:2026-02:Food",
        "insight": "Dining Out is significantly down this week at $73.\n Groceries is significantly down this week at $67.\n Food is thus significantly down this week to $326.\n"
      }
    ],
    "review_needed": [
      {
        "key": "spend_vs_forecast:2026-02:Food",
        "insight": "Your g{[Dining Out](/meals_dining_out/weekly)} is lower at g{$85}! Your g{[Groceries](/meals_groceries/weekly)} is lower at g{$90}!"
      }
    ],
    "past_review_outcomes": []
  },
  {
    "name": "large_txn_correct",
    "eval_input": [
      {
        "key": "2025-02-14:2001:large_txn",
        "insight": "Outflow to APPLE STORE of $1,299 is larger than usual."
      }
    ],
    "review_needed": [
      {
        "key": "2025-02-14:2001:large_txn",
        "insight": "Heads up! r{$1,299} to APPLE STORE is larger than usual. 👀"
      }
    ],
    "past_review_outcomes": []
  },
  {
    "name": "uncat_txn_missing_ask",
    "eval_input": [
      {
        "key": "2025-02-14:3001:uncat_txn",
        "insight": "Uncategorized outflow of $85 to STEAMGAMES. Suggested category: Entertainment."
      }
    ],
    "review_needed": [
      {
        "key": "2025-02-14:3001:uncat_txn",
        "insight": "You spent r{$85} at STEAMGAMES. 🕹️"
      }
    ],
    "past_review_outcomes": []
  }
]


def run_test(test_case: dict, checker: CheckJsonOptimizer = None):
  """
  Run a single test case.
  
  Args:
    test_case: Test case dict with name, eval_item (or eval_input), eval_output (or review_needed), and optionally past_review_outcomes
    checker: Optional CheckJsonOptimizer instance. If None, creates a new one.
    
  Returns:
    Dictionary with good_copy, info_correct, and eval_text keys, or None if error occurred
  """
  if checker is None:
    checker = CheckJsonOptimizer()
  
  print(f"\\n{'='*80}")
  print(f"Running test: {test_case['name']}")
  print(f"{'='*80}")
  
  try:
    # Support both old format (eval_item/eval_output) and new format (eval_input/review_needed/past_review_outcomes)
    if "eval_input" in test_case:
      # New format
      eval_input = test_case["eval_input"]
      past_review_outcomes = test_case.get("past_review_outcomes", [])
      review_needed = test_case["review_needed"]
      # Ensure review_needed is a list
      if not isinstance(review_needed, list):
        review_needed = [review_needed]
    else:
      # Old format - convert to new format
      eval_input = test_case["eval_item"]
      past_review_outcomes = test_case.get("past_review_outcomes", [])
      # Use eval_output as review_needed (should be a list)
      if isinstance(test_case["eval_output"], list):
        review_needed = test_case["eval_output"]
      else:
        review_needed = [test_case["eval_output"]]
    
    # Print the exact input that will be passed to the LLM
    past_review_section = ""
    if past_review_outcomes:
      past_review_section = f"""<PAST_REVIEW_OUTCOMES>
{json.dumps(past_review_outcomes, indent=2)}
</PAST_REVIEW_OUTCOMES>

"""
    
    request_text = f"""<EVAL_INPUT>
{json.dumps(eval_input, indent=2)}
</EVAL_INPUT>

{past_review_section}<REVIEW_NEEDED>
{json.dumps(review_needed, indent=2)}
</REVIEW_NEEDED>

Output:"""
    print(f"Input passed to LLM:")
    print(request_text)
    print(f"\\n{'='*80}")
    
    result = checker.generate_response(eval_input, past_review_outcomes, review_needed)
    print(f"Result:")
    print(json.dumps(result, indent=2))
    print(f"{'='*80}")
    return result
  except Exception as e:
    print(f"ERROR: {type(e).__name__}: {str(e)}")
    print(f"{'='*80}")
    return None


def run_tests(test_names: list = None, checker: CheckJsonOptimizer = None):
  """
  Run multiple test cases.
  
  Args:
    test_names: List of test case names to run. If None, runs all tests.
    checker: Optional CheckJsonOptimizer instance. If None, creates a new one.
    
  Returns:
    List of results (None entries indicate failed tests)
  """
  if checker is None:
    checker = CheckJsonOptimizer()
  
  if test_names is None:
    tests_to_run = TEST_CASES
  else:
    tests_to_run = [tc for tc in TEST_CASES if tc["name"] in test_names]
  
  results = []
  passed = 0
  failed = 0
  
  for test_case in tests_to_run:
    result = run_test(test_case, checker)
    results.append(result)
    if result is None:
      failed += 1
    else:
      passed += 1
  
  print(f"\\n{'='*80}")
  print(f"Test Summary: {passed} passed, {failed} failed out of {len(tests_to_run)} tests")
  print(f"{'='*80}")
  
  return results


# Batches for testing
BATCHES = [
  ["spend_vs_forecast_no_color_no_link", "spend_vs_forecast_missing_info_and_formatting"],
  ["provided_example_1_mixed", "provided_example_2_factual_mismatch"],
  ["large_txn_correct", "uncat_txn_missing_ask"],
  ["spend_vs_forecast_no_color_no_link", "provided_example_1_mixed"]
]

def main():
  """Main function to test the checker optimizer. Pass batch 1..4 to run that batch."""
  checker = CheckJsonOptimizer()
  batch_num = int(sys.argv[1]) if len(sys.argv) > 1 else 1
  if batch_num < 1 or batch_num > len(BATCHES):
    print(f"Invalid batch number. Please choose from 1 to {len(BATCHES)}.")
    return
  batch_names = BATCHES[batch_num - 1]
  print(f"Running Batch {batch_num} ({len(batch_names)} tests): {batch_names}")
  run_tests(batch_names, checker=checker)


if __name__ == "__main__":
  main()
