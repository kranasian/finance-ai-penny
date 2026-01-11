from google import genai
from google.genai import types
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are a checker verifying the output of a transaction name classifier.

## Input:
- **EVAL_INPUT**: A JSON array of transaction pairs. Each pair has a `match_id`, `left` transaction set, and `right` transaction set.
- **PAST_REVIEW_OUTCOMES**: An array of past review outcomes.
- **REVIEW_NEEDED**: The JSON output from the classifier that needs to be reviewed.

## Output:
JSON: `{"good_copy": boolean, "info_correct": boolean, "eval_text": string}`
- `good_copy`: True if REVIEW_NEEDED is a valid JSON array, contains an object for every `match_id` in EVAL_INPUT, and each object has the required fields (`match_id`, `reasoning`, `result`, `confidence`).
- `info_correct`: True if the `result` and `reasoning` for each item in REVIEW_NEEDED correctly apply the classification rules.
- `eval_text`: Required if either boolean is False; provide a single, direct sentence per issue.

## Critical Priority: Learn from PAST_REVIEW_OUTCOMES
**MANDATORY**: If PAST_REVIEW_OUTCOMES flags issues that still exist in REVIEW_NEEDED, mark as incorrect.
- Extract all issues from past `eval_text` fields.
- Check if REVIEW_NEEDED repeats the same mistakes.

## Verification Steps
1. **Check PAST_REVIEW_OUTCOMES first**: If REVIEW_NEEDED repeats past mistakes -> mark False.
2. **Verify good_copy**: Check for valid JSON, complete `match_id` coverage, and all required fields.
3. **Verify info_correct**: For each item, check if the `result` and `reasoning` are correct based on the Classifier Rules.
4. **Write eval_text**: If False, provide a single, direct sentence per issue.

## Classifier Rules for Verification
You must verify the classifier's output against these rules. The rules are ordered by importance.

**1. Mandatory Rules:**
- **Ignore Geographic Location:** Transactions at different physical locations of the same company **must** be classified as "same". (e.g., "McDonald's in LA" vs. "McDonald's in NY" -> "same").
- **Distinguish Online vs. Physical:** A company's online store is **different** from its physical store. (e.g., "Gap.com" vs. "Gap Store" -> "different").

**2. Core Task:**
The classifier's decision is based on two conditions being met:
1. The `short_name` from `left` must accurately describe all transactions in `right`.
2. The `short_name` from `right` must accurately describe all transactions in `left`.
- If **both** are true, the result is **"same"**. Otherwise, it is **"different"**.

**3. Analysis Heuristics:**
- **Marketplaces**: A marketplace transaction is **different** from a direct one (e.g., "Amazon: Nike" vs. "Nike Store").
- **Payment Processors**: The use of a payment processor should be **ignored** (e.g., "Stripe*MyWebsite" is the same as "MyWebsite").
- **Product/Service Distinction**: Different products or service tiers are **different** (e.g., "Spotify Silver" vs. "Spotify Gold").
- **Payment & Transfer Specificity**: More specific payments or transfers are **different** from general ones (e.g., "Zelle to Bob: Rent" vs. "Zelle to Bob").
- **Sub-brands**: Sub-brands are generally **same** as the parent company.

### Output Field Rules
- `reasoning`: Must be a brief 1-2 sentence explanation.
- `result`: Must be either "same" or "different".
- `confidence`: Must be "high", "medium", or "low".
"""

class CheckSameSummarizedNameClassifier:
  """Handles all Gemini API interactions for checking SameSummarizedNameClassifier outputs against rules"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for checking SameSummarizedNameClassifier evaluations"""
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
    self.max_output_tokens = 6000
    
    # Safety Settings
    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
    ]
    
    # System Prompt
    self.system_prompt = SYSTEM_PROMPT

  
  def generate_response(self, eval_input: list, past_review_outcomes: list, review_needed: list) -> dict:
    """
    Generate a response using Gemini API for checking SameSummarizedNameClassifier outputs.
    
    Args:
      eval_input: A JSON array of transaction pairs.
      past_review_outcomes: An array of past review outcomes, each containing `output`, `good_copy`, `info_correct`, and `eval_text`.
      review_needed: The SameSummarizedNameClassifier output that needs to be reviewed (JSON array).
      
    Returns:
      Dictionary with good_copy, info_correct, and eval_text keys
    """
    # Create request text with the new input structure
    request_text_str = f"""<EVAL_INPUT>

{json.dumps(eval_input, indent=2)}

</EVAL_INPUT>

<PAST_REVIEW_OUTCOMES>

{json.dumps(past_review_outcomes, indent=2)}

</PAST_REVIEW_OUTCOMES>

<REVIEW_NEEDED>

{json.dumps(review_needed, indent=2)}

</REVIEW_NEEDED>

Output:"""
    
    print(request_text_str)
    print(f"\n{'='*80}\n")
    
    request_text = types.Part.from_text(text=request_text_str)
    
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


def run_test_case(test_name: str, eval_input: list, review_needed: list, past_review_outcomes: list = None, checker: 'CheckSameSummarizedNameClassifier' = None):
  """
  Utility function to run a test case with custom inputs, common error handling, and output formatting.
  
  Args:
    test_name: Name of the test case
    eval_input: A list of transaction pairs to be evaluated.
    review_needed: The classifier output that needs to be reviewed (list of dicts).
    past_review_outcomes: An array of past review outcomes. Defaults to empty list.
    checker: Optional CheckSameSummarizedNameClassifier instance. If None, creates a new one.
    
  Returns:
    Dictionary with good_copy, info_correct, and eval_text keys, or None if error occurred
  """
  if past_review_outcomes is None:
    past_review_outcomes = []
  if checker is None:
    checker = CheckSameSummarizedNameClassifier()

  print(f"\n{'='*80}")
  print(f"Running test: {test_name}")
  print(f"{'='*80}")

  try:
    # Directly call the checker's response with the provided inputs.
    result = checker.generate_response(eval_input, past_review_outcomes, review_needed)
    print(f"Result:")
    print(json.dumps(result, indent=2))
    print(f"{'='*80}")
    return result
  except Exception as e:
    print(f"ERROR: {type(e).__name__}: {str(e)}")
    print(f"{'='*80}")
    return None


def run_correct_response(checker: CheckSameSummarizedNameClassifier = None):
  """
  Run the test case for a correct response.
  """
  eval_input = [
    {
      "match_id": "2112:2113",
      "left": {
        "short_name": "OpenAI",
        "raw_names": [
          "OPENAI",
          "OPENAI +14158799686 USA"
        ],
        "description": "sells access to its large language model API, including text generation, translation, and code completion",
        "amounts": [6.00, 5.32]
      },
      "right": {
        "short_name": "OpenAI Chatgpt",
        "raw_names": ["Openai Chatgpt Subscr"],
        "description": "sells subscriptions to use the AI chatbot for various tasks such as writing, coding, and problem-solving",
        "amounts": [21.28, 20.00]
      }
    },
    {
        "match_id": "SPOT-01",
        "left": {
            "short_name": "Spotify Silver Plan",
            "raw_names": ["Spotify Silver Plan"],
            "description": "Subscription for music streaming service.",
            "amounts": [9.99]
        },
        "right": {
            "short_name": "Spotify Platinum",
            "raw_names": ["SPOTIFY PLATINUM"],
            "description": "Premium subscription for music and podcast streaming.",
            "amounts": [15.99]
        }
    },
    {
        "match_id": "GRAB-01",
        "left": {
            "short_name": "Grab: Wendy's",
            "raw_names": ["GrabFood*Wendy's"],
            "description": "Food delivery from a fast-food chain.",
            "amounts": [12.50]
        },
        "right": {
            "short_name": "Wendy's",
            "raw_names": ["WENDY'S"],
            "description": "Fast-food restaurant specializing in hamburgers.",
            "amounts": [18.75]
        }
    }
  ]
  
  review_needed = [
    {
      "match_id": "2112:2113",
      "reasoning": "OpenAI API access and ChatGPT subscriptions are distinct services from the same company, making them different.",
      "result": "different",
      "confidence": "high"
    },
    {
      "match_id": "SPOT-01",
      "reasoning": "Spotify Silver and Platinum are different subscription tiers for the same service.",
      "result": "different",
      "confidence": "high"
    },
    {
      "match_id": "GRAB-01",
      "reasoning": "A transaction made through the Grab marketplace is different from a direct transaction with Wendy's.",
      "result": "different",
      "confidence": "high"
    }
  ]
  
  return run_test_case("correct_response", eval_input, review_needed, [], checker)


def run_edge_cases(checker: CheckSameSummarizedNameClassifier = None):
  """
  Run the test case for edge cases including online vs physical and different locations.
  """
  eval_input = [
    {
        "match_id": "AMZN-01",
        "left": {
            "short_name": "Amazon.com",
            "raw_names": ["AMAZON.COM"],
            "description": "Online retailer.",
            "amounts": [50.0]
        },
        "right": {
            "short_name": "Amazon Go",
            "raw_names": ["Amazon Go Store"],
            "description": "Physical convenience store.",
            "amounts": [12.50]
        }
    },
    {
        "match_id": "MGM-01",
        "left": {
            "short_name": "MGM Grand",
            "raw_names": ["MGM Grand Las Vegas"],
            "description": "Hotel and casino in Las Vegas.",
            "amounts": [300.0]
        },
        "right": {
            "short_name": "MGM Grand Detroit",
            "raw_names": ["MGM Grand Detroit"],
            "description": "Hotel and casino in Detroit.",
            "amounts": [250.0]
        }
    }
  ]
  
  review_needed = [
    {
      "match_id": "AMZN-01",
      "reasoning": "Amazon.com is an online store and Amazon Go is a physical store, so they are different.",
      "result": "different",
      "confidence": "high"
    },
    {
      "match_id": "MGM-01",
      "reasoning": "These are the same hotel chain, just different locations.",
      "result": "same",
      "confidence": "high"
    }
  ]
  
  return run_test_case("edge_cases", eval_input, review_needed, [], checker)


def run_test_group_1(checker: CheckSameSummarizedNameClassifier = None):
  """
  Test Group 1: Name Variations
  """
  eval_input = [
    {
        "match_id": "FB-01",
        "left": { "short_name": "Facebook.com", "raw_names": ["FACEBOOK.COM"], "description": "Social media platform.", "amounts": [10.00] },
        "right": { "short_name": "Facebook", "raw_names": ["Facebook"], "description": "Online social networking service.", "amounts": [15.50] }
    },
    {
        "match_id": "PP-01",
        "left": { "short_name": "Paypal Retry Payment", "raw_names": ["PAYPAL *RETRY PYMT"], "description": "A second attempt for a PayPal transaction.", "amounts": [45.00] },
        "right": { "short_name": "Paypal Payment", "raw_names": ["PayPal Payment"], "description": "A standard payment via PayPal.", "amounts": [45.00] }
    },
    {
      "match_id": "MARCO-01",
      "left": { "short_name": "Marco's Pizza", "raw_names": ["Marco's Pizza"], "description": "Pizza restaurant", "amounts": [70.92] },
      "right": { "short_name": "Marcos Pizza", "raw_names": ["MARCOS PIZZA"], "description": "A payment for food at a pizza restaurant", "amounts": [16.23] }
    }
  ]
  review_needed = [
    { "match_id": "FB-01", "reasoning": "Facebook.com and Facebook are different variations of the same entity.", "result": "same", "confidence": "high" },
    { "match_id": "PP-01", "reasoning": "A 'Retry Payment' is a different type of transaction than a standard 'Payment'.", "result": "different", "confidence": "medium" },
    { "match_id": "MARCO-01", "reasoning": "These are the same restaurant, just with a punctuation difference.", "result": "same", "confidence": "high" }
  ]
  return run_test_case("test_group_1_name_variations", eval_input, review_needed, [], checker)


def run_test_group_2(checker: CheckSameSummarizedNameClassifier = None):
  """
  Test Group 2: Sub-brands and Locations
  """
  eval_input = [
    {
      "match_id": "OLDNAVY-01",
      "left": { "short_name": "Old Navy Kids", "raw_names": ["Old Navy Kids"], "description": "Sells clothing for children.", "amounts": [39.09] },
      "right": { "short_name": "Old Navy", "raw_names": ["Old Navy"], "description": "Sells clothing for all ages.", "amounts": [145.54] }
    },
    {
      "match_id": "MARRIOTT-01",
      "left": { "short_name": "Cleveland Marriott", "raw_names": ["CLEVELAND MARRIOT"], "description": "Hotel in Cleveland.", "amounts": [250.00] },
      "right": { "short_name": "Marriott", "raw_names": ["Marriott"], "description": "Hotel chain.", "amounts": [189.00] }
    },
    {
      "match_id": "WALMART-01",
      "left": { "short_name": "Walmart Supercenter", "raw_names": ["WALMART SUPERCENTER"], "description": "Large format Walmart store.", "amounts": [120.00] },
      "right": { "short_name": "Walmart", "raw_names": ["Walmart"], "description": "General retailer.", "amounts": [80.00] }
    }
  ]
  review_needed = [
    { "match_id": "OLDNAVY-01", "reasoning": "'Old Navy Kids' is a specific sub-brand; it cannot accurately describe all transactions from the general 'Old Navy', making them different.", "result": "different", "confidence": "high" },
    { "match_id": "MARRIOTT-01", "reasoning": "These are the same hotel brand, just one specifies a location.", "result": "same", "confidence": "high" },
    { "match_id": "WALMART-01", "reasoning": "A Supercenter is a type of Walmart store, they are the same entity.", "result": "same", "confidence": "high" }
  ]
  return run_test_case("test_group_2_sub_brands_and_locations", eval_input, review_needed, [], checker)


def run_test_group_3(checker: CheckSameSummarizedNameClassifier = None):
  """
  Test Group 3: Edge Cases and Ambiguity
  """
  eval_input = [
    {
      "match_id": "ZELLE-01",
      "left": { "short_name": "Zelle to Gabby", "raw_names": ["Zelle Transfer to Gabby"], "description": "P2P transfer.", "amounts": [50.00] },
      "right": { "short_name": "Zelle to Gabby: Jollibee", "raw_names": ["Zelle to Gabby: Jollibee"], "description": "P2P transfer with memo.", "amounts": [25.30] }
    },
    {
      "match_id": "SALES-01",
      "left": { "short_name": "Sales", "raw_names": ["Sales"], "description": "Undetermined.", "amounts": [1581.33] },
      "right": { "short_name": "Seamless", "raw_names": ["Seamless"], "description": "Food delivery service.", "amounts": [84.29] }
    },
    {
      "match_id": "DOLLAR-01",
      "left": { "short_name": "Dollar Tree", "raw_names": ["Dollartree"], "description": "Discount store.", "amounts": [4.06] },
      "right": { "short_name": "Dollar Tree Payment", "raw_names": ["DOLLARTREE REF# 5023"], "description": "Payment to Dollar Tree.", "amounts": [4.06] }
    }
  ]
  review_needed = [
    { "match_id": "ZELLE-01", "reasoning": "A Zelle transfer with a specific memo is different from a general transfer.", "result": "different", "confidence": "high" },
    { "match_id": "SALES-01", "reasoning": "The name 'Sales' is too generic to be matched with a specific service like 'Seamless'.", "result": "different", "confidence": "high" },
    { "match_id": "DOLLAR-01", "reasoning": "The word 'Payment' is extraneous information; both refer to the same merchant.", "result": "same", "confidence": "high" }
  ]
  return run_test_case("test_group_3_edge_cases", eval_input, review_needed, [], checker)


def main(batch: int = 0):
  """Main function to test the SameSummarizedNameClassifier checker"""
  checker = CheckSameSummarizedNameClassifier()
  
  if batch == 0:
    # Run all tests if no specific batch is selected
    run_correct_response(checker)
    run_edge_cases(checker)
    run_test_group_1(checker)
    run_test_group_2(checker)
    run_test_group_3(checker)
  elif batch == 1:
    run_test_group_1(checker)
  elif batch == 2:
    run_test_group_2(checker)
  elif batch == 3:
    run_test_group_3(checker)
  else:
    print("Invalid batch number. Please choose 0, 1, 2, or 3.")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Run checker tests in batches')
  parser.add_argument('--batch', type=int, default=0, choices=[0, 1, 2, 3],
                      help='Batch number to run (1, 2, or 3). 0 runs all.')
  args = parser.parse_args()
  main(batch=args.batch)
