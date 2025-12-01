from google import genai
from google.genai import types
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are a checker very good at verifying the correctness of evaluation of insights against a set of rules.

## Input:
1. **eval_item**: The item to be evaluated. An array of keys and insights.
2. **eval_output**: The evaluation output. An array of keys and insights.

## Output:
JSON object with the following keys:
- **good_copy**: boolean. `True` if the **eval_output** is a good copy of the **eval_item**, `False` if it is not.
- **info_correct**: boolean. `True` if the **eval_output** is correct, `False` if it is not.
- **eval_text**: string. Notes on the evaluation if *good_copy* is `False` or *info_correct* is `False`.

## Your Tasks

Check the **eval_item** against the **eval_output** following the rules below. For each item in **eval_item**, find the corresponding item in **eval_output** by matching the `key` field, then verify that the `insight` field in **eval_output** follows all applicable rules.

## Rules for Verification

### General Guidelines
1. **Exclude greeting**: The insight should not include a greeting.
2. **Use emojis conversationally**: Emojis should be used naturally and conversationally.
3. **Use appropriate tone**: The tone should match the insight type (positive for good news, neutral/concerned for issues).
4. **Character limit**: Strictly keep to 100 characters only, including spaces and emojis. Remove pieces of information if necessary to follow the character limit.
5. **Directional prepositions**: Be mindful when using directional prepositions. Inflows are from an establishment, while outflows are to an establishment.
6. **Monetary amounts**: Use commas, include currency, and exclude decimals (e.g., $1,000, not $1000.00).
7. **Formatting**:
   - Text can be made green by enclosing it in g{} (ex: g{$1,234})
   - Text can be made green and linked by specifying the category and timeframe (ex: g{[food spending](/food/monthly)})
   - Text can be made red by enclosing it in r{} (ex: r{$1,234})
   - Text can be made red and linked by specifying the category and timeframe (ex: r{[food spending](/food/monthly)})

### `...large_txn` Guidelines
- **Messaging**: Include transaction name and monetary amount, and that it is smaller/larger than usual.
- **Format**:
  - Inflow is smaller than usual: monetary amount in red
  - Outflow is larger than usual: monetary amount in red
  - Outflow is smaller than usual: monetary amount in green
  - Inflow is larger than usual: monetary amount in green

### `...spend_vs_forecast` and `...spent_vs_forecast` Guidelines
- **Messaging**: Include monetary amounts, timeframe (ie. weekly or monthly), and severity of divergence (or synonyms).
  - When referring to category totals, avoid using "higher by X", "up X", "X higher", and anything similar. Instead, use "higher at X", "up at X", "increased to X", and similar phrases. Note that "increased this week to $264" means that the total for the week is $264.
  - Specify if the insight is on an inflow/income or outflow/spending.
- **Format**:
  - Outflow category is higher than forecasted: category and monetary amount in red, with category linked
  - Inflow category is lower than forecasted: category and monetary amount in red, with category linked
  - Inflow category is higher than forecasted: category and monetary amount in green, with category linked
  - Outflow category is lower than forecasted: category and monetary amount in green, with category linked

### `...uncat_txn` Guidelines
- **Messaging**: Include transaction name and monetary amount, then ask for confirmation on the suggested category, if any. Ask for how it should be categorized as if there is no suggested category.
- **Format**:
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

## Verification Process

1. **Match keys**: For each item in **eval_item**, find the corresponding item in **eval_output** with the same `key`.
2. **Check good_copy**: Verify that the **eval_output** insight is a good copy/rewrite of the **eval_item** insight, maintaining the core information while following the formatting and messaging guidelines.
3. **Check info_correct**: Verify that the **eval_output** insight follows all applicable rules based on the key type:
   - If the key contains `spend_vs_forecast` or `spent_vs_forecast`, check the color formatting rules (green for lower outflow/higher inflow, red for higher outflow/lower inflow) and that the category is linked.
   - If the key contains `large_txn`, check the color formatting rules based on inflow/outflow and size.
   - If the key contains `uncat_txn`, check that the monetary amount is in the correct color (red for outflow, green for inflow) and category is in plain text.
4. **Generate eval_text**: If either `good_copy` or `info_correct` is `False`, provide specific notes about what is incorrect. Be concise and specific about the issues found.

<EXAMPLES>

input:
```json
{
  "eval_item": [
    {
      "key": "25:spend_vs_forecast",
      "insight": "Transport spending was slightly reduced this month, now at $203. Car & Fuel and Public Transit both saw slight increases, contributing to the overall change for this time. These are all compared to the forecasts based on average spending for this time."
    },
    {
      "key": "9:spent_vs_forecast",
      "insight": "Service Fees spending significantly decreased last week, now at $3000. These are all compared to forecasts based on average spending."
    }
  ],
  "eval_output": [
    {
      "key": "25:spend_vs_forecast",
      "insight": "Great news! 🎉 Your r{[Transport spending](/transportation/monthly)} was lower this month at g{$203}!"
    },
    {
      "key": "9:spent_vs_forecast",
      "insight": "Awesome! Your r{[Service Fees spending](/bills_service_fees/weekly)} was lower last week at g{$3,000}! 🥳"
    }
  ]
}
```

output:
```json
{
  "good_copy": false,
  "info_correct": false,
  "eval_text": "Transport spending should be in green. Service Fees spending should be in green"
}
```

</EXAMPLES>
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
    self.thinking_budget = 4096
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

  
  def generate_response(self, eval_item: list, eval_output: list) -> dict:
    """
    Generate a response using Gemini API for checking evaluations.
    
    Args:
      eval_item: The item to be evaluated. An array of keys and insights.
      eval_output: The evaluation output. An array of keys and insights.
      
    Returns:
      Dictionary with good_copy, info_correct, and eval_text keys
    """
    # Create request text with eval_item and eval_output
    request_text = types.Part.from_text(text=f"""Input:
{json.dumps({"eval_item": eval_item, "eval_output": eval_output}, indent=2)}

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
      
      # Extract JSON object from the response
      json_start = output_text.find('{')
      json_end = output_text.rfind('}') + 1
      
      if json_start != -1 and json_end > json_start:
        json_str = output_text[json_start:json_end]
        return json.loads(json_str)
      else:
        # Try parsing the whole response
        return json.loads(output_text.strip())
    except json.JSONDecodeError as e:
      raise ValueError(f"Failed to parse JSON response: {e}\nResponse length: {len(output_text)}\nResponse preview: {output_text[:500]}")


def test_with_inputs(eval_item: list, eval_output: list, checker: CheckJsonOptimizer = None):
  """
  Convenient method to test the checker optimizer with custom inputs.
  
  Args:
    eval_item: The item to be evaluated. An array of keys and insights.
    eval_output: The evaluation output. An array of keys and insights.
    checker: Optional CheckJsonOptimizer instance. If None, creates a new one.
    
  Returns:
    Dictionary with good_copy, info_correct, and eval_text keys
  """
  if checker is None:
    checker = CheckJsonOptimizer()
  
  return checker.generate_response(eval_item, eval_output)


# Test cases covering different scenarios
TEST_CASES = [
  {
    "name": "spend_vs_forecast_incorrect_colors",
    "eval_item": [
      {
        "key": "25:spend_vs_forecast",
        "insight": "Transport spending was slightly reduced this month, now at $203. Car & Fuel and Public Transit both saw slight increases, contributing to the overall change for this time. These are all compared to the forecasts based on average spending for this time."
      },
      {
        "key": "9:spent_vs_forecast",
        "insight": "Service Fees spending significantly decreased last week, now at $3000. These are all compared to forecasts based on average spending."
      }
    ],
    "eval_output": [
      {
        "key": "25:spend_vs_forecast",
        "insight": "Great news! 🎉 Your r{[Transport spending](/transportation/monthly)} was lower this month at g{$203}!"
      },
      {
        "key": "9:spent_vs_forecast",
        "insight": "Awesome! Your r{[Service Fees spending](/bills_service_fees/weekly)} was lower last week at g{$3,000}! 🥳"
      }
    ]
  },
  {
    "name": "spend_vs_forecast_correct_lower_outflow",
    "eval_item": [
      {
        "key": "25:spend_vs_forecast",
        "insight": "Transport spending was slightly reduced this month, now at $203."
      }
    ],
    "eval_output": [
      {
        "key": "25:spend_vs_forecast",
        "insight": "Great news! 🎉 Your g{[Transport spending](/transportation/monthly)} was lower this month at g{$203}!"
      }
    ]
  },
  {
    "name": "spend_vs_forecast_correct_higher_outflow",
    "eval_item": [
      {
        "key": "15:spend_vs_forecast",
        "insight": "Dining Out spending increased this month, now at $450."
      }
    ],
    "eval_output": [
      {
        "key": "15:spend_vs_forecast",
        "insight": "Heads up! Your r{[Dining Out spending](/meals_dining_out/monthly)} was higher this month at r{$450}."
      }
    ]
  },
  {
    "name": "spend_vs_forecast_correct_higher_inflow",
    "eval_item": [
      {
        "key": "30:spend_vs_forecast",
        "insight": "Salary income increased this month, now at $5,500."
      }
    ],
    "eval_output": [
      {
        "key": "30:spend_vs_forecast",
        "insight": "Awesome! Your g{[Salary income](/income_salary/monthly)} was higher this month at g{$5,500}! 🎉"
      }
    ]
  },
  {
    "name": "spend_vs_forecast_correct_lower_inflow",
    "eval_item": [
      {
        "key": "31:spend_vs_forecast",
        "insight": "Sidegig income decreased this month, now at $800."
      }
    ],
    "eval_output": [
      {
        "key": "31:spend_vs_forecast",
        "insight": "Note: Your r{[Sidegig income](/income_sidegig/monthly)} was lower this month at r{$800}."
      }
    ]
  },
  {
    "name": "large_txn_outflow_larger_red",
    "eval_item": [
      {
        "key": "-50:large_txn",
        "insight": "Outflow transaction to Amazon for $2,500 is larger than usual."
      }
    ],
    "eval_output": [
      {
        "key": "-50:large_txn",
        "insight": "Heads up! r{$2,500} to Amazon is larger than usual. 👀"
      }
    ]
  },
  {
    "name": "large_txn_outflow_smaller_green",
    "eval_item": [
      {
        "key": "-51:large_txn",
        "insight": "Outflow transaction to Target for $150 is smaller than usual."
      }
    ],
    "eval_output": [
      {
        "key": "-51:large_txn",
        "insight": "Nice! g{$150} to Target is smaller than usual. 👍"
      }
    ]
  },
  {
    "name": "large_txn_inflow_larger_green",
    "eval_item": [
      {
        "key": "50:large_txn",
        "insight": "Inflow transaction from Employer for $6,000 is larger than usual."
      }
    ],
    "eval_output": [
      {
        "key": "50:large_txn",
        "insight": "Awesome! g{$6,000} from Employer is larger than usual! 🎉"
      }
    ]
  },
  {
    "name": "large_txn_inflow_smaller_red",
    "eval_item": [
      {
        "key": "51:large_txn",
        "insight": "Inflow transaction from Freelance Client for $1,200 is smaller than usual."
      }
    ],
    "eval_output": [
      {
        "key": "51:large_txn",
        "insight": "Note: r{$1,200} from Freelance Client is smaller than usual."
      }
    ]
  },
  {
    "name": "uncat_txn_outflow_red",
    "eval_item": [
      {
        "key": "-100:uncat_txn",
        "insight": "Uncategorized outflow transaction for Chime for $1,000 with a likely category of Gadgets."
      }
    ],
    "eval_output": [
      {
        "key": "-100:uncat_txn",
        "insight": "Heads up! 🧐 r{$1,000} outflow to Chime. Is this for Gadgets? Let me know! 👇"
      }
    ]
  },
  {
    "name": "uncat_txn_inflow_green",
    "eval_item": [
      {
        "key": "100:uncat_txn",
        "insight": "Uncategorized inflow transaction from PayPal for $500 with a likely category of Sidegig."
      }
    ],
    "eval_output": [
      {
        "key": "100:uncat_txn",
        "insight": "Hey! g{$500} inflow from PayPal. Is this for Sidegig? 👇"
      }
    ]
  },
  {
    "name": "uncat_txn_no_category_suggestion",
    "eval_item": [
      {
        "key": "-101:uncat_txn",
        "insight": "Uncategorized outflow transaction for Unknown Merchant for $250."
      }
    ],
    "eval_output": [
      {
        "key": "-101:uncat_txn",
        "insight": "r{$250} to Unknown Merchant. How should I categorize this? 👇"
      }
    ]
  },
  {
    "name": "multiple_categories_mixed",
    "eval_item": [
      {
        "key": "20:spend_vs_forecast",
        "insight": "Groceries spending increased this week, now at $180."
      },
      {
        "key": "-60:large_txn",
        "insight": "Outflow transaction to Best Buy for $800 is larger than usual."
      },
      {
        "key": "-102:uncat_txn",
        "insight": "Uncategorized outflow transaction for Venmo for $75."
      }
    ],
    "eval_output": [
      {
        "key": "20:spend_vs_forecast",
        "insight": "Your r{[Groceries spending](/meals_groceries/weekly)} was higher this week at r{$180}."
      },
      {
        "key": "-60:large_txn",
        "insight": "r{$800} to Best Buy is larger than usual. 👀"
      },
      {
        "key": "-102:uncat_txn",
        "insight": "r{$75} to Venmo. How should I categorize this? 👇"
      }
    ]
  },
  {
    "name": "character_limit_exceeded",
    "eval_item": [
      {
        "key": "25:spend_vs_forecast",
        "insight": "Transport spending was slightly reduced this month, now at $203."
      }
    ],
    "eval_output": [
      {
        "key": "25:spend_vs_forecast",
        "insight": "Great news! 🎉 Your g{[Transport spending](/transportation/monthly)} was lower this month at g{$203}! This is excellent progress and shows you're managing your transportation costs well!"
      }
    ]
  },
  {
    "name": "missing_category_link",
    "eval_item": [
      {
        "key": "25:spend_vs_forecast",
        "insight": "Transport spending was slightly reduced this month, now at $203."
      }
    ],
    "eval_output": [
      {
        "key": "25:spend_vs_forecast",
        "insight": "Great news! 🎉 Your Transport spending was lower this month at g{$203}!"
      }
    ]
  },
  {
    "name": "wrong_timeframe",
    "eval_item": [
      {
        "key": "9:spent_vs_forecast",
        "insight": "Service Fees spending significantly decreased last week, now at $3000."
      }
    ],
    "eval_output": [
      {
        "key": "9:spent_vs_forecast",
        "insight": "Awesome! Your g{[Service Fees spending](/bills_service_fees/monthly)} was lower last week at g{$3,000}! 🥳"
      }
    ]
  },
  {
    "name": "monetary_format_incorrect",
    "eval_item": [
      {
        "key": "25:spend_vs_forecast",
        "insight": "Transport spending was slightly reduced this month, now at $203."
      }
    ],
    "eval_output": [
      {
        "key": "25:spend_vs_forecast",
        "insight": "Great news! 🎉 Your g{[Transport spending](/transportation/monthly)} was lower this month at g{$203.00}!"
      }
    ]
  }
]


def run_test(test_case: dict, checker: CheckJsonOptimizer = None):
  """
  Run a single test case.
  
  Args:
    test_case: Test case dict with name, eval_item, and eval_output
    checker: Optional CheckJsonOptimizer instance. If None, creates a new one.
    
  Returns:
    Dictionary with good_copy, info_correct, and eval_text keys, or None if error occurred
  """
  if checker is None:
    checker = CheckJsonOptimizer()
  
  print(f"\n{'='*80}")
  print(f"Running test: {test_case['name']}")
  print(f"{'='*80}")
  
  try:
    result = checker.generate_response(test_case["eval_item"], test_case["eval_output"])
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
  
  print(f"\n{'='*80}")
  print(f"Test Summary: {passed} passed, {failed} failed out of {len(tests_to_run)} tests")
  print(f"{'='*80}")
  
  return results


def main():
  """Main function to test the checker optimizer"""
  checker = CheckJsonOptimizer()
  
  # Run all tests
  run_tests(checker=checker)
  
  # Or run specific tests:
  # run_tests(["spend_vs_forecast_correct_lower_outflow", "uncat_txn_outflow_red"], checker=checker)


if __name__ == "__main__":
  main()
