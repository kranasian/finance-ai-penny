from google import genai
from google.genai import types
import os
import json
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """**Objective:** Evaluate `ai_review_response` for red flags against prior turns, prioritizing response appropriateness to the latest Human intent.

**Input:**
Input is a list of conversation turns. `conversation_history` is all elements except the last; the last is always `ai_review_response`.

**Output:**
- Return a JSON object where each key is a red-flag key and each value is a concise rationale.
- Multiple keys are allowed only when truly necessary.
- If `good_response` applies, it must be the ONLY key.
- Prefer exactly one best-fitting key in normal cases.
- Keep rationale to one sentence, grounded in explicit evidence from the input turns.

**Decision rules (apply strictly):**
1) Judge response quality using only the conversation context provided. Do not assume missing backend/system limitations unless explicitly mentioned.
2) A polite processing acknowledgment (e.g., "give me a moment while I check") is `good_response` when it is context-appropriate and not contradictory.
3) A clear technical-error limitation message is `good_response` if it is transparent and helpful.
4) If AI repeats earlier AI wording verbatim or near-verbatim instead of paraphrasing in a continuing conversation, use `repetitive_information`.
5) Use `verbose` when response includes unnecessary internal/extra details (e.g., transaction IDs, unrelated categories, long assistant self-introductions, or leading with unrequested breakdown before the requested top-line answer).
6) Label precedence when overlapping: contradiction/illogical -> `non_sense`; missing requested parts -> `incomplete`; unnecessary added detail -> `verbose`; exact/near-exact historical repetition -> `repetitive_information`; otherwise `good_response`.
7) If the user asks for a single top-line value (for example "how much/total/forecasted spending"), label `verbose` when the response leads with unrequested category/itemized breakdown or gives an overly detailed breakdown. A brief breakdown may be acceptable only when it comes after the requested top-line value.

**Red flag keys:**
- `good_response`: Appropriate, direct response or valid limitation/processing communication. Use this when no other red flag applies.
- `non_sense`: Illogical/irrelevant response, invalid reasoning, or contradiction with earlier AI statements.
- `repetitive_information`: Repetition without added value, including repeating prior AI phrasing from conversation history.
- `incoherent_flow`: Breaks conversation progression or causes a conversational dead end.
- `verbose`: Includes unnecessary detail that weakens directness to the user's request.
- `incomplete`: Fails to answer all requested parts and does not provide a clear reason for missing parts.
"""

OUTPUT_SCHEMA = types.Schema(
  type=types.Type.OBJECT,
  properties={
    "good_response": types.Schema(type=types.Type.STRING),
    "non_sense": types.Schema(type=types.Type.STRING),
    "repetitive_information": types.Schema(type=types.Type.STRING),
    "incoherent_flow": types.Schema(type=types.Type.STRING),
    "verbose": types.Schema(type=types.Type.STRING),
    "incomplete": types.Schema(type=types.Type.STRING),
  },
)

CONFIG = {
  "json": True,
  "sanitize": True,
  "gen_config": {
    "top_k": 40,
    "top_p": 0.95,
    "temperature": 0.1,
    "thinking_budget": 1024,
    "max_output_tokens": 1152,
  },
  "model_name": "gemini-flash-lite-latest",
  "check_template": "Chk:VerbalizedResponseReviewer",
}

def _compare_flag_keys(actual: dict, ideal: dict) -> tuple[bool, str]:
  """Compare output by red-flag keys only (ignores rationale wording)."""
  actual_keys = set(actual.keys())
  ideal_keys = set(ideal.keys())
  if actual_keys != ideal_keys:
    return False, f"model_keys={sorted(actual_keys)} ideal_keys={sorted(ideal_keys)}"
  return True, f"matching keys: {sorted(actual_keys)}"

class VerbalizedResponseReviewer:
  """Handles all Gemini API interactions for checking AI responses for red flags"""
  
  def __init__(self, model_name=CONFIG["model_name"]):
    """Initialize the Gemini agent with API configuration for checking AI response red flags"""
    # API Configuration
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file or environment.")
    self.client = genai.Client(api_key=api_key)
    
    # Model Configuration
    self.thinking_budget = CONFIG["gen_config"]["thinking_budget"]
    self.model_name = model_name
    self.check_template = CONFIG["check_template"]
    self.json_enabled = CONFIG["json"]
    self.sanitize_enabled = CONFIG["sanitize"]
    
    # Generation Configuration Constants
    self.top_k = CONFIG["gen_config"]["top_k"]
    self.top_p = CONFIG["gen_config"]["top_p"]
    self.temperature = CONFIG["gen_config"]["temperature"]
    self.max_output_tokens = CONFIG["gen_config"]["max_output_tokens"]
    
    # Safety Settings
    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
    ]
    
    # System Prompt
    self.system_prompt = SYSTEM_PROMPT

  
  def evaluate_response(self, input_list_json_str: str) -> dict:
    """
    Evaluate an AI response for red flags using Gemini API.
    
    Args:
      input_list_json_str: JSON string of a list of conversation turns, where the final element is the ai_review_response.
      
    Returns:
      Dictionary containing the red flag key and rationale
    """
    # Parse the input JSON string into a Python list
    input_list = json.loads(input_list_json_str)

    # Extract ai_review_response (last element of the list)
    ai_review_response_dict = input_list[-1]
    ai_review_response = ai_review_response_dict["ai_review_response"]
    
    # Extract conversation history (all elements except the last)
    conversation_history_list = input_list[:-1]
    
    # Format conversation history for the prompt
    formatted_conversation_history = []
    for turn in conversation_history_list:
      if "Human" in turn:
        formatted_conversation_history.append({"speaker": "Human", "message": turn["Human"]})
      elif "AI" in turn:
        formatted_conversation_history.append({"speaker": "AI", "message": turn["AI"]})

    # Create request text with the new input structure
    request_payload = {
        "conversation_history": formatted_conversation_history,
        "ai_review_response": ai_review_response
    }
    request_text_str = f"""input: {json.dumps(request_payload, indent=2)}
output: """
    
    print(f"\n{'='*80}")
    print("INPUT JSON:")
    print(input_list_json_str)
    print("="*80)
    
    request_text = types.Part.from_text(text=request_text_str)
    
    # Create content and configuration
    contents = [types.Content(role="user", parts=[request_text])]
    
    generate_content_config = types.GenerateContentConfig(
      temperature=self.temperature,
      top_k=self.top_k,
      top_p=self.top_p,
      max_output_tokens=self.max_output_tokens,
      response_mime_type="application/json",
      response_schema=OUTPUT_SCHEMA,
      safety_settings=self.safety_settings,
      system_instruction=[types.Part.from_text(text=self.system_prompt)],
      thinking_config=types.ThinkingConfig(
        thinking_budget=self.thinking_budget,
        include_thoughts=True,
      ),
    )

    # Generate response
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
                  if hasattr(part, "thought") and part.thought:
                    if hasattr(part, "text") and part.text:
                      thought_summary += part.text
    except Exception as e:
      error_msg = f"API Error: {str(e)}";
      print(f"\n{'='*80}");
      print("ERROR:");
      print(error_msg);
      print("="*80);
      raise Exception(f"Failed to generate response from Gemini API: {error_msg}")
    
    # Check if response is empty
    if not output_text or not output_text.strip():
      raise ValueError(f"Empty response from model. Check API key and model availability.")
    
    print(f"{'='*80}")
    print("RESPONSE OUTPUT:")
    print(output_text.strip())
    print("="*80)
    if thought_summary.strip():
      print(f"{'=' * 80}")
      print("THOUGHT SUMMARY:")
      print(thought_summary.strip())
      print("=" * 80)

    # Try to parse as JSON, if it fails return as text
    try:
      cleaned_output = output_text.strip() # Initialize cleaned_output

      # Remove markdown code blocks if present
      if cleaned_output.startswith("```json"):
        json_start_idx = cleaned_output.find("```json") + len("```json")
        json_end_idx = cleaned_output.rfind("```")
        if json_start_idx != -1 and json_end_idx != -1 and json_end_idx > json_start_idx:
          cleaned_output = cleaned_output[json_start_idx:json_end_idx].strip()
      elif cleaned_output.startswith("```"):
        # Generic markdown block, just remove start and end lines
        lines = cleaned_output.split('\n')
        if len(lines) > 1 and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
          cleaned_output = '\n'.join(lines[1:-1]).strip()
        
      # Find the first opening brace and the last closing brace to extract the JSON string
      json_start = cleaned_output.find('{')
      json_end = cleaned_output.rfind('}') + 1

      if json_start != -1 and json_end > json_start:
        json_str = cleaned_output[json_start:json_end]
        return json.loads(json_str)
      else:
        # Fallback if no valid JSON object found
        raise json.JSONDecodeError("No JSON object found or invalid JSON structure", cleaned_output, 0)

    except json.JSONDecodeError as e:
      # If not valid JSON, return as a dict with the raw text and error info
      print(f"\nWarning: Failed to parse JSON response: {e}")
      print(f"Raw output preview: {output_text[:200]}...")
      return {"raw_output": output_text.strip(), "parse_error": str(e)}


TEST_CASES = [
  {
    "name": "complex_request_processing_ack_good_response",
    "input": [
      {
        "Human": "Can you compare my total income, total expenses, and savings rate for each quarter last year, then highlight the quarter with the biggest change and explain why?"
      },
      {
        "ai_review_response": "Absolutely, I can do that. Give me a moment to process all four quarters and I’ll get back to you with a clear breakdown. ⏳"
      }
    ],
    "output": {"good_response": "The response appropriately acknowledges the complex request and communicates a valid short processing delay."},
  },
  {
    "name": "food_spend_partial_answer_incomplete",
    "input": [
      {
        "Human": "How much did I spend on food last month and this month?"
      },
      {
        "ai_review_response": "You spent $412 on food last month."
      }
    ],
    "output": {"incomplete": "Only one part of a two-part request was answered, with no explanation for the missing current-month value."},
  },
  {
    "name": "account_balance_technical_error_good_response",
    "input": [
      {
        "Human": "Can you show me all my current account balances?"
      },
      {
        "ai_review_response": "I’m sorry — I’m having a technical issue retrieving live balances right now. Please try again in a bit, and I’ll fetch them as soon as the connection is restored."
      }
    ],
    "output": {"good_response": "A clear and polite technical-error explanation is a valid response when data cannot be retrieved."},
  },
  {
    "name": "historical_repeat_repetitive_information",
    "input": [
      {
        "Human": "Can you remind me what my emergency fund target is?"
      },
      {
        "AI": "Your emergency fund target is $12,000 based on your current monthly expenses."
      },
      {
        "Human": "Got it. And how much have I already saved toward that?"
      },
      {
        "AI": "You’ve currently saved $8,600 toward that emergency fund goal."
      },
      {
        "Human": "Before that, what did you say my emergency fund target was again?"
      },
      {
        "ai_review_response": "Your emergency fund target is $12,000 based on your current monthly expenses."
      }
    ],
    "output": {"repetitive_information": "The response repeats the same earlier AI wording verbatim instead of paraphrasing."},
  },
  {
    "name": "transactions_with_internal_ids_verbose",
    "input": [
      {
        "Human": "Can you list my transactions from yesterday?"
      },
      {
        "ai_review_response": "Sure — yesterday you had: 1) Grocery Mart (ID: 84192) - $64.20, 2) Coffee Spot (ID: 84195) - $5.80, 3) Ride Share (ID: 84201) - $14.40."
      }
    ],
    "output": {"verbose": "Internal identifiers are unnecessary for the user-facing answer and add excess detail."},
  },
  {
    "name": "forecast_category_then_total_verbose",
    "input": [
      {
        "Human": "What’s my forecasted spending next month?"
      },
      {
        "ai_review_response": "Next month forecast by category: groceries $420, dining $280, transport $190, subscriptions $95, utilities $240, entertainment $160. Total forecasted spending is $1,385."
      }
    ],
    "output": {"verbose": "The response should present the requested total first; category breakdown before the total adds unnecessary detail."},
  },
  {
    "name": "donation_with_extra_categories_verbose",
    "input": [
      {
        "Human": "How much did I spend on donations last week?"
      },
      {
        "ai_review_response": "You spent $85 on donations last week. Similar amounts from other categories: books $79, coffee $82, parking $88, and streaming $83."
      }
    ],
    "output": {"verbose": "The user only asked for donation spending, so other categories are unnecessary additions."},
  },
  {
    "name": "long_self_intro_before_answer_verbose",
    "input": [
      {
        "Human": "What was my highest expense this week?"
      },
      {
        "ai_review_response": "Hi! I’m Penny, your AI financial assistant. I help you track spending, monitor cash flow, understand trends, and make better money decisions. I can summarize transactions, spot unusual activity, and answer budget questions quickly. I’m designed to keep responses clear, practical, and personalized to your data. I’m always here to support your financial planning journey. Your highest expense this week was $310 at Midtown Grocer."
      }
    ],
    "output": {"verbose": "A long self-introduction is unnecessary and distracts from the direct answer."},
  },
]

def test_with_inputs(input_list, reviewer=None):
  """
  Convenient method to test the reviewer with custom inputs.
  
  Args:
    input_list: List of dictionaries representing conversation turns, with the final element containing ai_review_response.
    reviewer: Optional VerbalizedResponseReviewer instance. If None, creates a new one.
    
  Returns:
    Dictionary containing the red flag key and rationale
  """
  if reviewer is None:
    reviewer = VerbalizedResponseReviewer()
  
  return reviewer.evaluate_response(json.dumps(input_list, indent=2))


def _find_test_case(case_name: str):
  return next((case for case in TEST_CASES if case["name"] == case_name), None)


def run_batch_test(case_name: str, reviewer=None):
  """
  Run a specific test batch.
  """
  test_case = _find_test_case(case_name)
  if test_case is None:
    available_case_names = [case["name"] for case in TEST_CASES]
    raise ValueError(f"Batch '{case_name}' not found. Available batches: {available_case_names}")
  
  print(f"\n{'='*80}")
  print(f"RUNNING BATCH: {case_name}")
  print(f"{'='*80}\n")
  result = test_with_inputs(test_case["input"], reviewer)
  ideal_output = test_case["output"]
  print("IDEAL OUTPUT (REFERENCE):")
  print(json.dumps(ideal_output, indent=2))
  print("-" * 80)
  key_match, details = _compare_flag_keys(result, ideal_output)
  print(f"KEY MATCH: {'PASS' if key_match else 'FAIL'}")
  print(details)
  print("=" * 80)
  return result


def main(batch_id: str = None):
  """
  Main function to test the verbalized response reviewer.
  Optionally runs a specific batch if batch_id is provided.
  """
  try:
    reviewer = VerbalizedResponseReviewer()
    results = {}
    
    if batch_id:
      result = run_batch_test(batch_id, reviewer)
      results[batch_id] = result
    else:
      for case in TEST_CASES:
        case_name = case["name"]
        result = run_batch_test(case_name, reviewer)
        results[case_name] = result
    
    print(f"\n{'='*80}")
    print("FINAL RESULTS SUMMARY:")
    print(json.dumps(results, indent=2))
    print(f"{'='*80}\n")
    print("✅ Successfully evaluated the AI response(s)!")
    
    return results
  except ValueError as e:
    # Handle API key or initialization errors
    print(f"\n{'='*80}")
    print("ERROR OCCURRED:")
    print(str(e))
    print(f"{'='*80}\n")
    print("Troubleshooting:")
    print("1. Check that GEMINI_API_KEY is set in your .env file")
    print("2. Verify your API key is valid and has proper permissions")
    print("3. Ensure you have network connectivity")
    return None
  except Exception as e:
    # Handle API errors
    error_str = str(e)
    print(f"\n{'='*80}")
    print("ERROR OCCURRED:")
    print(error_str)
    print(f"{'='*80}\n")
    
    if "API key" in error_str or "PERMISSION_DENIED" in error_str or "403" in error_str:
      print("\n⚠️  API Key Issue Detected:")
      print("   Your GEMINI_API_KEY appears to be invalid or has been revoked.")
      print("   Please update your .env file with a valid API key.")
      print("   Get a new API key from: https://makersuite.google.com/app/apikey")
    else:
      print("\nTroubleshooting:")
      print("1. Check that GEMINI_API_KEY is set in your .env file")
      print("2. Verify your API key is valid and has proper permissions")
      print("3. Ensure you have network connectivity")
      print("4. Check the error message above for specific details")
    
    return None


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description="Run Verbalized Response Reviewer with optional batch ID.")
  parser.add_argument("--batch", type=str, help="ID of the batch to run (e.g., initial_greeting_good). If not provided, all batches will run.")
  args = parser.parse_args()
  main(batch_id=args.batch)