from google import genai
from google.genai import types
import os
import sys
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are a checker verifying verbalizer outputs against rules.

## Input:
- **EVAL_INPUT**: Notable activity in personal finance (string). This is the **source of truth** for all factual checks.
- **PAST_REVIEW_OUTCOMES**: Array of past reviews, each with `output`, `good_copy`, `info_correct`, `eval_text`
- **REVIEW_NEEDED**: Highlights from EVAL_INPUT rewritten as actionable messages (string)

## Output:
JSON: `{"good_copy": boolean, "info_correct": boolean, "eval_text": string}`
- `good_copy`: **(FORMATTING ONLY)** True if REVIEW_NEEDED perfectly follows all formatting rules from the "Rules" section.
- `info_correct`: **(ACCURACY ONLY)** True if all information in REVIEW_NEEDED is factually accurate when compared to EVAL_INPUT. EVAL_INPUT is the sole source of truth.
- `eval_text`: Required ONLY if either boolean is False. For each issue: state the issue and **indicate why** (e.g. what is wrong and what it should be, or which rule). Use short phrases. Refer to insights by number (e.g., "Insight 1: ..."). **If text color is wrong**, state whether the error is in the **category** or **amount** color and why (e.g., "Insight 1: Wrong color on category (should be green for lower spending)", "Insight 2: Wrong color on amount (should be red for uncat outflow)"). No long paragraphs.

## Critical Priority: Strict Compliance
- **Evaluate only REVIEW_NEEDED**: Match each item in REVIEW_NEEDED to EVAL_INPUT by `key`; do not require or judge insights that are not in REVIEW_NEEDED.
- **Recall over Precision**: Flag ANY potential violation. False positives preferred over false negatives.
- **Learn from PAST_REVIEW_OUTCOMES**: If issues from past `eval_text` persist, mark as incorrect.
- **Conciseness**: Keep `eval_text` short. For each issue include a brief reason: e.g. "Wrong color on category (should be green for lower spending)", "Missing link (spend_vs_forecast category must be linked)", "Timeframe mismatch (text says 'last week' but link has '/monthly')", "Decimals in amount (use whole dollars)", "Comma in amount under 1000 (no comma for $999)", "Missing comma in amount 1000 or more ($1,000 required)".

## Rules

### Part 1: Content Rules
1.  **Factual Accuracy**: All information in REVIEW_NEEDED must match EVAL_INPUT. EVAL_INPUT is the source of truth.
2.  **Openers**: Conversational openers (e.g. "Great news!", "Heads up!", "Awesome!", "Hey!", "Note:") are **optional**. ONLY formal greetings like "Hi", "Hello", "Good morning", and "Good evening" are forbidden.
3.  **Required Info**: ONLY category, amount, and direction (high/low) are required. Magnitude (e.g., "slightly", "significantly") and other details (e.g. parent categories, breakdown of sub-categories) are NOT required. If the insight mentions a sub-category (e.g. "Medical & Pharmacy") but the parent category (e.g. "Health") is not mentioned in the output, this is ACCEPTABLE.
4.  **Prepositions**: Inflows *from* establishment, outflows *to* establishment.
5.  **Amounts**: Treat **$0** as a valid amount; it must appear correctly if present in EVAL_INPUT.
6.  **Insight-Specific**:
    *   `...large_txn`: Include name, amount, and larger/smaller.
    *   `...spend_vs_forecast`: Include category, amount, and higher/lower. Category MUST be linked.
    *   `...uncat_txn`: Include name, amount, and ask for category.

### Part 2: Formatting Rules (`good_copy` ONLY)
1.  **Colors**: Based on performance (EVAL_INPUT is source of truth):
    *   **GREEN `g{...}`** = good: spending lower/expected, income higher/expected, uncategorized **inflow**; for `...large_txn`: outflow **smaller** than usual, inflow **larger** than usual.
    *   **RED `r{...}`** = bad: spending higher than expected, income lower than expected, uncategorized **outflow**; for `...large_txn`: outflow **larger** than usual, inflow **smaller** than usual.
    *   For `...spend_vs_forecast`: category and amount share the same color per insight. For `...uncat_txn`: outflow amount RED, inflow amount GREEN. For `...large_txn`: only the amount is colored (no category link).
2.  **Links**: `...spend_vs_forecast` category MUST be linked and colored.
3.  **Amounts**: Use `$`, no decimals. **Commas only when the absolute value of the amount is 1,000 or more** (e.g., `$999` no comma; `$1,000` with comma; `$0` is valid, no comma).
4.  **Emojis**: Allowed.
5.  **Syntax**: Balanced `[]`, `{}`, `()`.

### Part 3: Character Count Rule (`good_copy` ONLY)
1.  **Strict Limit**: 100 characters maximum per insight. Count only **visible text and spaces**; exclude all markup (`g{`, `r{`, `}`, `[]`, `()`, link syntax, and URL). Flag only when the visible character count clearly exceeds 100.
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
      raise ValueError(f"Failed to parse JSON response: {e}\nResponse length: {len(output_text)}\nResponse preview: {output_text[:500]}")


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
    "name": "spend_vs_forecast_incorrect_colors",
    "eval_input": [
      {
        "key": "25:spend_vs_forecast",
        "insight": "Transport spending was slightly reduced this month, now at $203. Car & Fuel and Public Transit both saw slight increases, contributing to the overall change for this time. These are all compared to the forecasts based on average spending for this time."
      },
      {
        "key": "9:spent_vs_forecast",
        "insight": "Service Fees spending significantly decreased last week, now at $3000. These are all compared to forecasts based on average spending."
      }
    ],
    "review_needed": [{
      "key": "25:spend_vs_forecast",
      "insight": "Great news! 🎉 Your r{[Transport spending](/transportation/monthly)} was lower this month at g{$203}!"
    }, {
      "key": "9:spent_vs_forecast",
      "insight": "Awesome! Your r{[Service Fees spending](/bills_service_fees/weekly)} was lower last week at g{$3,000}! 🥳"
    }],
    "past_review_outcomes": []
  },
  {
    "name": "spend_vs_forecast_incorrect_colors_second_item",
    "eval_input": [
      {
        "key": "25:spend_vs_forecast",
        "insight": "Transport spending was slightly reduced this month, now at $203. Car & Fuel and Public Transit both saw slight increases, contributing to the overall change for this time. These are all compared to the forecasts based on average spending for this time."
      },
      {
        "key": "9:spent_vs_forecast",
        "insight": "Service Fees spending significantly decreased last week, now at $3000. These are all compared to forecasts based on average spending."
      }
    ],
    "review_needed": [{
      "key": "25:spend_vs_forecast",
      "insight": "Great news! 🎉 Your r{[Transport spending](/transportation/monthly)} was lower this month at g{$203}!"
    }, {
      "key": "9:spent_vs_forecast",
      "insight": "Awesome! Your r{[Service Fees spending](/bills_service_fees/weekly)} was lower last week at g{$3,000}! 🥳"
    }],
    "past_review_outcomes": []
  },
  {
    "name": "spend_vs_forecast_correct_lower_outflow",
    "eval_input": [
      {
        "key": "25:spend_vs_forecast",
        "insight": "Transport spending was slightly reduced this month, now at $203."
      }
    ],
    "review_needed": [{
      "key": "25:spend_vs_forecast",
      "insight": "Great news! 🎉 Your g{[Transport spending](/transportation/monthly)} was lower this month at g{$203}!"
    }],
    "past_review_outcomes": []
  },
  {
    "name": "spend_vs_forecast_correct_higher_outflow",
    "eval_input": [
      {
        "key": "15:spend_vs_forecast",
        "insight": "Dining Out spending increased this month, now at $450."
      }
    ],
    "review_needed": [{
      "key": "15:spend_vs_forecast",
      "insight": "Heads up! Your r{[Dining Out spending](/meals_dining_out/monthly)} was higher this month at r{$450}."
    }],
    "past_review_outcomes": [
      {
        "output": {
          "key": "15:spend_vs_forecast",
          "insight": "Your Dining Out spending was higher this month at $450."
        },
        "good_copy": False,
        "info_correct": False,
        "eval_text": "good_copy is False: Missing category link. info_correct is False: Category must be linked for spend_vs_forecast insights."
      }
    ]
  },
  {
    "name": "spend_vs_forecast_correct_higher_inflow",
    "eval_input": [
      {
        "key": "30:spend_vs_forecast",
        "insight": "Salary income increased this month, now at $5,500."
      }
    ],
    "review_needed": [{
      "key": "30:spend_vs_forecast",
      "insight": "Awesome! Your g{[Salary income](/income_salary/monthly)} was higher this month at g{$5,500}! 🎉"
    }],
    "past_review_outcomes": []
  },
  {
    "name": "spend_vs_forecast_correct_lower_inflow",
    "eval_input": [
      {
        "key": "31:spend_vs_forecast",
        "insight": "Sidegig income decreased this month, now at $800."
      }
    ],
    "review_needed": [{
      "key": "31:spend_vs_forecast",
      "insight": "Note: Your r{[Sidegig income](/income_sidegig/monthly)} was lower this month at r{$800}."
    }],
    "past_review_outcomes": []
  },
  {
    "name": "large_txn_outflow_larger_red",
    "eval_input": [
      {
        "key": "-50:large_txn",
        "insight": "Outflow transaction to Amazon for $2,500 is larger than usual."
      }
    ],
    "review_needed": [{
      "key": "-50:large_txn",
      "insight": "Heads up! r{$2,500} to Amazon is larger than usual. 👀"
    }],
    "past_review_outcomes": []
  },
  {
    "name": "large_txn_outflow_smaller_green",
    "eval_input": [
      {
        "key": "-51:large_txn",
        "insight": "Outflow transaction to Target for $150 is smaller than usual."
      }
    ],
    "review_needed": [{
      "key": "-51:large_txn",
      "insight": "Nice! g{$150} to Target is smaller than usual. 👍"
    }],
    "past_review_outcomes": []
  },
  {
    "name": "large_txn_inflow_larger_green",
    "eval_input": [
      {
        "key": "50:large_txn",
        "insight": "Inflow transaction from Employer for $6,000 is larger than usual."
      }
    ],
    "review_needed": [{
      "key": "50:large_txn",
      "insight": "Awesome! g{$6,000} from Employer is larger than usual! 🎉"
    }],
    "past_review_outcomes": []
  },
  {
    "name": "large_txn_inflow_smaller_red",
    "eval_input": [
      {
        "key": "51:large_txn",
        "insight": "Inflow transaction from Freelance Client for $1,200 is smaller than usual."
      }
    ],
    "review_needed": [{
      "key": "51:large_txn",
      "insight": "Note: r{$1,200} from Freelance Client is smaller than usual."
    }],
    "past_review_outcomes": []
  },
  {
    "name": "uncat_txn_outflow_red",
    "eval_input": [
      {
        "key": "-100:uncat_txn",
        "insight": "Uncategorized outflow transaction for Chime for $1,000 with a likely category of Gadgets."
      }
    ],
    "review_needed": [{
      "key": "-100:uncat_txn",
      "insight": "Heads up! 🧐 r{$1,000} outflow to Chime. Is this for Gadgets? Let me know! 👇"
    }],
    "past_review_outcomes": []
  },
  {
    "name": "uncat_txn_inflow_green",
    "eval_input": [
      {
        "key": "100:uncat_txn",
        "insight": "Uncategorized inflow transaction from PayPal for $500 with a likely category of Sidegig."
      }
    ],
    "review_needed": [{
      "key": "100:uncat_txn",
      "insight": "Hey! g{$500} inflow from PayPal. Is this for Sidegig? 👇"
    }],
    "past_review_outcomes": []
  },
  {
    "name": "uncat_txn_no_category_suggestion",
    "eval_input": [
      {
        "key": "-101:uncat_txn",
        "insight": "Uncategorized outflow transaction for Unknown Merchant for $250."
      }
    ],
    "review_needed": [{
      "key": "-101:uncat_txn",
      "insight": "r{$250} to Unknown Merchant. How should I categorize this? 👇"
    }],
    "past_review_outcomes": []
  },
  {
    "name": "multiple_categories_mixed_groceries",
    "eval_input": [
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
    "review_needed": [{
      "key": "20:spend_vs_forecast",
      "insight": "Your r{[Groceries spending](/meals_groceries/weekly)} was higher this week at r{$180}."
    }],
    "past_review_outcomes": []
  },
  {
    "name": "multiple_categories_mixed_large_txn",
    "eval_input": [
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
    "review_needed": [{
      "key": "-60:large_txn",
      "insight": "r{$800} to Best Buy is larger than usual. 👀"
    }],
    "past_review_outcomes": []
  },
  {
    "name": "multiple_categories_mixed_uncat_txn",
    "eval_input": [
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
    "review_needed": [{
      "key": "-102:uncat_txn",
      "insight": "r{$75} to Venmo. How should I categorize this? 👇"
    }],
    "past_review_outcomes": []
  },
  {
    "name": "character_limit_exceeded",
    "eval_input": [
      {
        "key": "25:spend_vs_forecast",
        "insight": "Transport spending was slightly reduced this month, now at $203."
      }
    ],
    "review_needed": [{
      "key": "25:spend_vs_forecast",
      "insight": "Great news! 🎉 Your g{[Transport spending](/transportation/monthly)} was lower this month at g{$203}! This is excellent progress and shows you're managing your transportation costs well!"
    }],
    "past_review_outcomes": [
      {
        "output": {
          "key": "25:spend_vs_forecast",
          "insight": "Great news! 🎉 Your g{[Transport spending](/transportation/monthly)} was lower this month at g{$203}! This is excellent progress and shows you're managing your transportation costs well!"
        },
        "good_copy": True,
        "info_correct": False,
        "eval_text": "info_correct is False: Exceeds 100 character limit. Current length: 145 characters."
      }
    ]
  },
  {
    "name": "missing_category_link",
    "eval_input": [
      {
        "key": "25:spend_vs_forecast",
        "insight": "Transport spending was slightly reduced this month, now at $203."
      }
    ],
    "review_needed": [{
      "key": "25:spend_vs_forecast",
      "insight": "Great news! 🎉 Your Transport spending was lower this month at g{$203}!"
    }],
    "past_review_outcomes": [
      {
        "output": {
          "key": "25:spend_vs_forecast",
          "insight": "Great news! 🎉 Your Transport spending was lower this month at g{$203}!"
        },
        "good_copy": True,
        "info_correct": False,
        "eval_text": "info_correct is False: Missing required category link for spend_vs_forecast. Category 'Transport spending' must be linked with format: g{[display text](/category/timeframe)} or r{[display text](/category/timeframe)}."
      }
    ]
  },
  {
    "name": "wrong_timeframe",
    "eval_input": [
      {
        "key": "9:spent_vs_forecast",
        "insight": "Service Fees spending significantly decreased last week, now at $3000."
      }
    ],
    "review_needed": [{
      "key": "9:spent_vs_forecast",
      "insight": "Awesome! Your g{[Service Fees spending](/bills_service_fees/monthly)} was lower last week at g{$3,000}! 🥳"
    }],
    "past_review_outcomes": [
      {
        "output": {
          "key": "9:spent_vs_forecast",
          "insight": "Awesome! Your g{[Service Fees spending](/bills_service_fees/monthly)} was lower last week at g{$3,000}! 🥳"
        },
        "good_copy": True,
        "info_correct": False,
        "eval_text": "info_correct is False: Timeframe mismatch. Insight mentions 'last week' but link uses '/monthly'. The timeframe in the link must match the timeframe mentioned in the insight text."
      }
    ]
  },
  {
    "name": "monetary_format_incorrect",
    "eval_input": [
      {
        "key": "25:spend_vs_forecast",
        "insight": "Transport spending was slightly reduced this month, now at $203."
      }
    ],
    "review_needed": [{
      "key": "25:spend_vs_forecast",
      "insight": "Great news! 🎉 Your g{[Transport spending](/transportation/monthly)} was lower this month at g{$203.00}!"
    }],
    "past_review_outcomes": [
      {
        "output": {
          "key": "25:spend_vs_forecast",
          "insight": "Great news! 🎉 Your g{[Transport spending](/transportation/monthly)} was lower this month at g{$203.00}!"
        },
        "good_copy": True,
        "info_correct": False,
        "eval_text": "info_correct is False: Incorrect monetary format. Should be '$203' (no decimals), not '$203.00'. Monetary format must be: $X,XXX (commas, currency, no decimals)."
      }
    ]
  },
  {
    "name": "past_review_outcomes_wrong_color_persists",
    "eval_input": [
      {
        "key": "spent_vs_forecast:2025-12:Health",
        "insight": "Medical & Pharmacy is significantly down last month at now $107.  Health is thus down last month to $169."
      }
    ],
    "review_needed": [{
      "key": "spent_vs_forecast:2025-12:Health",
      "insight": "Your r{[Medical & Pharmacy](/health_medical_pharmacy/monthly)} spending was lower last month at g{$107}! 🎉"
    }],
    "past_review_outcomes": [
      {
        "output": {
          "key": "spent_vs_forecast:2025-12:Health",
          "insight": "Your r{[Medical & Pharmacy spending](/health_medical_pharmacy/monthly)} was lower than expected at g{$107} last month! 🥳"
        },
        "good_copy": False,
        "info_correct": False,
        "eval_text": "good_copy is False: Insight omits core information (Health spending of $169). info_correct is False: Exceeds 100 characters. Category 'Medical & Pharmacy' is red, but should be green for an outflow lower than forecast."
      },
      {
        "output": {
          "key": "spent_vs_forecast:2025-12:Health",
          "insight": "Your r{[Medical & Pharmacy](/health_medical_pharmacy/monthly)} spending was lower last month at g{$107}! 🎉"
        },
        "good_copy": False,
        "info_correct": False,
        "eval_text": "Medical & Pharmacy category should be green."
      }
    ]
  },
  {
    "name": "past_review_outcomes_issue_fixed",
    "eval_input": [
      {
        "key": "spent_vs_forecast:2025-12:Health",
        "insight": "Medical & Pharmacy is significantly down last month at now $107.  Health is thus down last month to $169."
      }
    ],
    "review_needed": [{
      "key": "spent_vs_forecast:2025-12:Health",
      "insight": "Your g{[Medical & Pharmacy](/health_medical_pharmacy/monthly)} spending was lower last month at g{$107}! 🎉"
    }],
    "past_review_outcomes": [
      {
        "output": {
          "key": "spent_vs_forecast:2025-12:Health",
          "insight": "Your r{[Medical & Pharmacy](/health_medical_pharmacy/monthly)} spending was lower last month at g{$107}! 🎉"
        },
        "good_copy": False,
        "info_correct": False,
        "eval_text": "Medical & Pharmacy category should be green."
      }
    ]
  },
  {
    "name": "past_review_outcomes_missing_info_persists",
    "eval_input": [
      {
        "key": "spent_vs_forecast:2025-12:Health",
        "insight": "Medical & Pharmacy is significantly down last month at now $107.  Health is thus down last month to $169."
      }
    ],
    "review_needed": [{
      "key": "spent_vs_forecast:2025-12:Health",
      "insight": "Your g{[Medical & Pharmacy](/health_medical_pharmacy/monthly)} spending was lower last month at g{$107}! 🎉"
    }],
    "past_review_outcomes": [
      {
        "output": {
          "key": "spent_vs_forecast:2025-12:Health",
          "insight": "Your r{[Medical & Pharmacy spending](/health_medical_pharmacy/monthly)} was lower than expected at g{$107} last month! 🥳"
        },
        "good_copy": False,
        "info_correct": False,
        "eval_text": "good_copy is False: Insight omits core information (Health spending of $169). info_correct is False: Exceeds 100 characters. Category 'Medical & Pharmacy' is red, but should be green for an outflow lower than forecast."
      }
    ]
  },
  {
    "name": "zero_amount_valid",
    "eval_input": [{"key": "25:spend_vs_forecast", "insight": "Transport spending was zero this month, now at $0."}],
    "review_needed": [{"key": "25:spend_vs_forecast", "insight": "Your g{[Transport spending](/transportation/monthly)} was lower this month at g{$0}!"}],
    "past_review_outcomes": []
  },
  {
    "name": "comma_only_when_1000_or_more",
    "eval_input": [{"key": "25:spend_vs_forecast", "insight": "Transport spending was slightly reduced this month, now at $999."}],
    "review_needed": [{"key": "25:spend_vs_forecast", "insight": "Great news! Your g{[Transport spending](/transportation/monthly)} was lower this month at g{$999}!"}],
    "past_review_outcomes": []
  },
  {
    "name": "comma_required_when_1000_or_more",
    "eval_input": [{"key": "9:spent_vs_forecast", "insight": "Service Fees spending decreased last week, now at $1000."}],
    "review_needed": [{"key": "9:spent_vs_forecast", "insight": "Awesome! Your g{[Service Fees spending](/bills_service_fees/weekly)} was lower last week at g{$1000}!"}],
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
  
  print(f"\n{'='*80}")
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
    print(f"\n{'='*80}")
    
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
  
  print(f"\n{'='*80}")
  print(f"Test Summary: {passed} passed, {failed} failed out of {len(tests_to_run)} tests")
  print(f"{'='*80}")
  
  return results


def main():
  """Main function to test the checker optimizer. Pass batch 1..4 to run that batch."""
  checker = CheckJsonOptimizer()
  batch_num = int(sys.argv[1]) if len(sys.argv) > 1 else 1
  batch_names = BATCHES[batch_num - 1]
  print(f"Running Batch {batch_num} ({len(batch_names)} tests): {batch_names}")
  run_tests(batch_names, checker=checker)


if __name__ == "__main__":
  main()
