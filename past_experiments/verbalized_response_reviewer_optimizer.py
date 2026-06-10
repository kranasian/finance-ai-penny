from google import genai
from google.genai import types
import os
import json
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """**Objective:** Evaluate `ai_review_response` for red flags against prior turns, prioritizing whether the response appropriately addresses the Human's last message.

**Input:**
Input is a list of conversation turns. `conversation_history` is all elements except the last; the last is always `ai_review_response`.

**Output:**
Return one JSON object: a single red-flag key → one-sentence rationale. Prefer exactly one key. If `good_response` applies, it must be the only key. Rationales must be one sentence, evidence-based, and must not quote internal system identifiers or emoji characters.

**Red flag keys (six only):**
- `good_response`: Substantively addresses the Human's last message—direct answers, brief processing delays, communicated errors/limits, or acknowledged corrections.
- `incomplete`: Human's last message not substantively addressed (see rule 4).
- `verbose`: Extra information beyond the Human's ask lengthens the message without much added value (see rule 7).
- `repetitive_information`: Same phrases repeated within one response without added meaning.
- `incoherent_flow`: Breaks flow or dead-ends the conversation.
- `non_sense`: AI-shared figures or facts conflict with earlier AI on the same topic without acknowledging the discrepancy (see rule 6).

**Decision rules (apply strictly):**
1) Judge using only the conversation context provided. Do not assume unstated backend limitations.
2) **Responding vs completing:** `good_response` includes replies that substantively engage the Human's last message—even when the full data answer is not yet delivered. A brief processing acknowledgment with no data payload (e.g., "Sure, give me a second to check") is `good_response`: it addresses the Human, is not `incomplete`, and is not `verbose`.
3) **Errors and limits:** When the AI attempts the Human's request but fails, a clear communicated error or limitation is `good_response`—not `incomplete`. Examples: cannot retrieve balances, cannot save a budget. The Human was addressed even though the action did not succeed.
4) **`incomplete`:** Use only when the Human's last message is not substantively addressed—ignored, off-topic, or a multi-part question where an explicitly requested part is omitted with no explanation. Do not use merely because the full request was not completed; do not use for processing delays or communicated errors.
5) **Emojis:** Penny uses emojis for tone. Emojis are never a defect and must not be cited in rationales.
6) **`non_sense`:** Use when `ai_review_response` states figures or facts that conflict with earlier AI in the conversation on the same topic and does not acknowledge the discrepancy—leaving the Human with confusing contradictory information. **Correction exception:** if the response explicitly acknowledges the mistake and corrects it (e.g., "it was actually $230, not $180 — my bad for the hiccup"), that is `good_response`, not `non_sense`.
7) **`verbose`:** Use when information beyond what the Human asked for makes the message noticeably longer without much value for the ask. Includes: extra time periods or forecasts the Human did not request (e.g., Human asks last month only but response adds this month and next month); system reference numbers in a user-facing list; unrelated extra categories; a long self-introduction before the answer; or an itemized breakdown before a requested top-line total. Processing-only acknowledgments without extra data are not `verbose`.
8) **`repetitive_information`:** Same phrases repeated within one response without added meaning. Not when the Human asks for a reminder of earlier facts.
9) **Rationale hygiene:** One sentence only. Never write the words ID, txn, transaction number, or # in rationales; say "system reference numbers" instead.
10) **Precedence:** unacknowledged AI contradiction → `non_sense`; Human's message not addressed → `incomplete`; extra low-value length → `verbose`; within-response repetition → `repetitive_information`; otherwise `good_response`.
"""

OUTPUT_SCHEMA = types.Schema(
  type=types.Type.OBJECT,
  description="Exactly one red-flag key. If good_response applies, it must be the only key. Rationale: one sentence; no quoted internal IDs or emojis.",
  properties={
    "good_response": types.Schema(
      type=types.Type.STRING,
      description="Substantively addresses the Human's last message—direct answers, brief processing delays, communicated errors when an action fails, or acknowledged corrections of earlier AI mistakes.",
    ),
    "non_sense": types.Schema(
      type=types.Type.STRING,
      description="AI figures or facts conflict with earlier AI on the same topic without acknowledging the discrepancy. Not when the response apologizes and corrects the earlier mistake.",
    ),
    "repetitive_information": types.Schema(
      type=types.Type.STRING,
      description="Repeats the same phrases within the response itself without added meaning—not when the Human asked for a reminder of earlier facts.",
    ),
    "incoherent_flow": types.Schema(
      type=types.Type.STRING,
      description="Breaks conversation progression or causes a conversational dead end.",
    ),
    "verbose": types.Schema(
      type=types.Type.STRING,
      description="Extra information beyond the Human's ask (other periods/forecasts, system reference numbers, unrelated categories, long self-intro, breakdown before top-line total) lengthens the message without much value. Rationale must not contain ID, txn, or #.",
    ),
    "incomplete": types.Schema(
      type=types.Type.STRING,
      description="Human's last message is not substantively addressed—ignored, off-topic, or a requested part omitted with no explanation. Not when the Human was engaged via processing delay or a communicated error.",
    ),
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
        "Human": "How much did I spend on transport this week?"
      },
      {
        "ai_review_response": "Sure, give me a second to check. 📊"
      }
    ],
    "output": {"good_response": "The response appropriately addresses the Human's message with a brief processing acknowledgment."},
  },
  {
    "name": "food_spend_partial_answer_incomplete",
    "input": [
      {
        "Human": "How much did I spend on groceries and dining out in March and April?"
      },
      {
        "ai_review_response": "In March you spent $286 on groceries and $194 on dining out. 🛒"
      }
    ],
    "output": {"incomplete": "Only March was answered; April totals were not provided and no reason was given for the omission."},
  },
  {
    "name": "prior_ai_contradiction_non_sense",
    "input": [
      {
        "Human": "How much did I spend on groceries last month?"
      },
      {
        "AI": "You spent $180 on groceries last month."
      },
      {
        "Human": "How does that compare to this month so far?"
      },
      {
        "ai_review_response": "This month you’re at $95 so far, and last month was $240. 🛒"
      }
    ],
    "output": {"non_sense": "The response contradicts the AI’s earlier grocery total for last month without explanation, leaving the Human with conflicting figures."},
  },
  {
    "name": "prior_ai_correction_acknowledged_good_response",
    "input": [
      {
        "Human": "How much did I spend on groceries last month?"
      },
      {
        "AI": "You spent $180 on groceries last month."
      },
      {
        "Human": "How does that compare to this month so far?"
      },
      {
        "ai_review_response": "This month you’re at $95 so far — and it was actually $230 last month, not $180. My bad for the hiccup there! 📊"
      }
    ],
    "output": {"good_response": "The response transparently acknowledges and corrects the earlier figure while answering the comparison the Human asked for."},
  },
  {
    "name": "transactions_with_internal_ids_verbose",
    "input": [
      {
        "Human": "What did I buy over the weekend?"
      },
      {
        "ai_review_response": "Over the weekend you had: 1) River Market (txn #90214) — $47.80, 2) Blue Line Cafe (txn #90218) — $6.25, 3) City Parking (txn #90222) — $18.00. 🧾"
      }
    ],
    "output": {"verbose": "Internal transaction identifiers are unnecessary for the user-facing answer and add excess detail."},
  },
  {
    "name": "food_last_month_extra_forecasts_verbose",
    "input": [
      {
        "Human": "How much did I spend on food last month?"
      },
      {
        "ai_review_response": "You spent $412 on food last month. 🍽️ This month so far you’re at $318, and next month is forecasted at about $445."
      }
    ],
    "output": {"verbose": "The Human only asked for last month’s food spending; this-month and next-month figures add length without much value."},
  },
  {
    "name": "budget_set_error_good_response",
    "input": [
      {
        "Human": "Can you set a $500 monthly budget for dining out?"
      },
      {
        "ai_review_response": "Sorry — I ran into an error while trying to set that dining budget. Please try again in a moment and I’ll get it saved for you. 🔧"
      }
    ],
    "output": {"good_response": "A clear error message when the budget could not be set appropriately addresses the Human’s request."},
  },
  {
    "name": "long_self_intro_before_answer_verbose",
    "input": [
      {
        "Human": "What was my largest purchase this month?"
      },
      {
        "ai_review_response": "Hey! I’m Penny, your personal finance sidekick — I track spending, spot trends, flag unusual charges, and help you stay on top of budgets and goals. I’m built to keep answers practical, friendly, and tailored to your accounts. Whether you’re planning a trip or tightening grocery spend, I’ve got you. Your largest purchase this month was $892 at Westside Appliances. 🏠"
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