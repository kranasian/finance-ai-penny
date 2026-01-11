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
- `eval_text`: Required if either boolean is False; be specific and concise.

## Critical Priority: Learn from PAST_REVIEW_OUTCOMES
**MANDATORY**: If PAST_REVIEW_OUTCOMES flags issues that still exist in REVIEW_NEEDED, mark as incorrect.
- Extract all issues from past `eval_text` fields.
- Check if REVIEW_NEEDED repeats the same mistakes.

## Rules for `info_correct` Verification

You must verify the classifier's output against these rules:

**1. Core Task:**
The primary decision rule is: **Could the `short_name` from the left item AND the `short_name` from the right item EACH be used to accurately and completely describe all transactions in BOTH sets?**
- If **yes**, the result must be **"same"**.
- If **no**, the result must be **"different"**.

**2. Key Considerations:**
- **Ignore Geographic Locations**: Different physical locations should be classified as "same".

**3. Analysis Heuristics:**
- **Marketplaces vs. Payment Processors**:
    - Transactions via marketplaces (e.g., DoorDash, eBay) are **different** from direct merchant transactions.
    - The use of a payment processor (e.g., Stripe, Paypal) should be **ignored**.
- **Product/Service Distinction**:
    - Different products, service tiers, or charge types from the same company are **different**.
- **Payment & Transfer Specificity**:
    - A more specific payment type is **different** from a general one.
    - For P2P transfers, a transaction with a specific purpose (e.g., a memo) is **different** from a general transfer.
- **Sub-brands and Departments**:
    - Sub-brands or departments of the same parent company are generally **same**.

**4. Output Field Rules:**
- `reasoning`: Must be a brief 1-2 sentence explanation focusing on the most decisive factors.
- `result`: Must be either "same" or "different".
- `confidence`: Must be "high", "medium", or "low".

## Verification Steps

1. **Check PAST_REVIEW_OUTCOMES first**: Extract all flagged issues. If REVIEW_NEEDED repeats them -> mark False.
2. **Verify good_copy**: Is the output a valid JSON array? Does it have an entry for every `match_id` in the input? Does each entry have all required fields (`match_id`, `reasoning`, `result`, `confidence`) with valid enum values?
3. **Verify info_correct**: For each item in the output array, check if the classification (`result`) and `reasoning` are correct based on the Core Task and Analysis Heuristics.
4. **Write eval_text**: If False for any reason, provide a specific, concise explanation. If an issue from PAST_REVIEW_OUTCOMES is repeated, mention it.
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



def main():
  """Main function to test the SameSummarizedNameClassifier checker"""
  checker = CheckSameSummarizedNameClassifier()
  
  # Run all tests
  run_correct_response(checker)


if __name__ == "__main__":
  main()
