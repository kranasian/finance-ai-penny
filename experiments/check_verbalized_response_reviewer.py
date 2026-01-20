from google import genai
from google.genai import types
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are a checker verifying VerbalizedResponseReviewer outputs against rules.

## Input:
- **EVAL_INPUT**: JSON string containing conversation turns and `ai_review_response` (the AI response being evaluated)
- **PAST_REVIEW_OUTCOMES**: Array of past reviews, each with `output`, `good_copy`, `info_correct`, `eval_text`
- **REVIEW_NEEDED**: The VerbalizedResponseReviewer output to review (JSON string with `rating` and `rationale`)

## Output:
JSON: `{"good_copy": boolean, "info_correct": boolean, "eval_text": string}`
- `good_copy`: True if REVIEW_NEEDED is valid JSON with required `rating` and `rationale` keys, and rating is one of the valid keys
- `info_correct`: True if REVIEW_NEEDED follows all rules from the VerbalizedResponseReviewer template
- `eval_text`: Required if either boolean is False; be specific and concise

## Critical Priority: Learn from PAST_REVIEW_OUTCOMES
**MANDATORY**: If PAST_REVIEW_OUTCOMES flags issues that still exist in REVIEW_NEEDED, mark as incorrect.
- Extract all issues from past `eval_text` fields
- Check if REVIEW_NEEDED repeats the same mistakes
- If past reviews flag a missing element and it's still missing → mark `info_correct: False`

## Rules

### Output Format Requirements
1. **JSON Structure**: Must be valid JSON with exactly two keys: `rating` and `rationale`
2. **Rating Key**: Must be one of the following valid rating keys:
   - `good_response`: Straightforwardly answers the Human's last inquiry; good conversational response
   - `non_sense`: Information in response is misaligned with information in previous messages; response is not appropriate for the Human's last message; irrelevant, illogical, or has inconsistent computations
   - `repetitive_information`: Repeats information within its own response without value/context change
   - `incoherent_flow`: Conversation opener (e.g., Hello, Hi, Good morning, etc.) in the middle of an ongoing conversation; leads to dead end or breaks conversation flow
   - `verbose`: Response is too extensive for the Human's message/question; response is too long for it to be digestible for the Human; too many unnecessary details that prevent straightforwardly answering the Human's inquiry
   - `incomplete`: Does not answer all questions from the Human; too brief or lacks necessary information
3. **Rationale**: Must be present and provide a brief explanation for the rating

### Evaluation Rules (from VerbalizedResponseReviewer)
1. **Prioritize Relevance**: Must prioritize relevance to the preceding Human turn (the last Human message in conversation_turns)
2. **Use Latest Values**: Must ignore contradicting values from previous conversations; use latest values from `ai_review_response`
3. **Evaluate as Whole**: Must evaluate the response as a whole, considering full conversation context
4. **Appropriate Rating**: The rating must accurately reflect the quality issues (or lack thereof) in the `ai_review_response` based on the conversation history

### Content Quality Requirements
- **Rationale Quality**: The rationale must be brief, clear, and directly support the chosen rating
- **Context Awareness**: The evaluation must consider the full conversation context, not just the last turn in isolation
- **Accuracy**: The rating must accurately categorize the response based on the defined rating keys

## Verification Steps

1. **Check PAST_REVIEW_OUTCOMES first**: Extract all flagged issues. If REVIEW_NEEDED repeats them → mark False
2. **Verify good_copy**: 
   - Is REVIEW_NEEDED valid JSON?
   - Does it have both `rating` and `rationale` keys?
   - Is `rating` one of the valid rating keys?
   - Missing elements or invalid format = False
3. **Verify info_correct**: Apply all rules:
   - Does the rating accurately reflect the `ai_review_response` quality based on conversation context?
   - Does the evaluation prioritize relevance to the Human's last turn?
   - Does the rationale clearly explain the rating choice?
   - Is the evaluation based on the full conversation context, not just isolated turns?
   - Does it use the latest values from `ai_review_response` rather than contradicting previous conversations?
   - Is the rating appropriate for the identified issues (or lack thereof)?
   - **Specific error checks**:
     - If `incoherent_flow`: Does the response contain conversation openers (Hello, Hi, Good morning, etc.) in the middle of an ongoing conversation?
     - If `incomplete`: Does the response fail to answer all questions from the Human?
     - If `verbose`: Is the response too extensive/long for the Human's message/question, making it hard to digest?
     - If `non_sense`: Is the information in the response misaligned with previous messages, or is the response not appropriate for the Human's last message?
4. **Write eval_text**: If False, list specific issues. Reference unfixed PAST_REVIEW_OUTCOMES issues.
"""

class CheckVerbalizedResponseReviewer:
  """Handles all Gemini API interactions for checking VerbalizedResponseReviewer outputs against rules"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for checking VerbalizedResponseReviewer evaluations"""
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

  
  def generate_response(self, eval_input: str, past_review_outcomes: list, review_needed: str) -> dict:
    """
    Generate a response using Gemini API for checking VerbalizedResponseReviewer outputs.
    
    Args:
      eval_input: JSON string containing conversation turns and ai_review_response (the AI response being evaluated).
      past_review_outcomes: An array of past review outcomes, each containing `output`, `good_copy`, `info_correct`, and `eval_text`.
      review_needed: The VerbalizedResponseReviewer output that needs to be reviewed (JSON string with rating and rationale).
      
    Returns:
      Dictionary with good_copy, info_correct, and eval_text keys
    """
    # Create request text with the new input structure
    request_text_str = f"""<EVAL_INPUT>

{eval_input}

</EVAL_INPUT>

<PAST_REVIEW_OUTCOMES>

{json.dumps(past_review_outcomes, indent=2)}

</PAST_REVIEW_OUTCOMES>

<REVIEW_NEEDED>

{review_needed}

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


def run_test_case(test_name: str, eval_input: str, review_needed: str, past_review_outcomes: list = None, checker: 'CheckVerbalizedResponseReviewer' = None):
  """
  Utility function to run a test case with custom inputs, common error handling, and output formatting.
  
  Args:
    test_name: Name of the test case
    eval_input: JSON string containing conversation turns and ai_review_response (the AI response being evaluated).
    review_needed: The VerbalizedResponseReviewer output that needs to be reviewed (JSON string with rating and rationale).
    past_review_outcomes: An array of past review outcomes, each containing `output`, `good_copy`, `info_correct`, and `eval_text`. Defaults to empty list.
    checker: Optional CheckVerbalizedResponseReviewer instance. If None, creates a new one.
    
  Returns:
    Dictionary with good_copy, info_correct, and eval_text keys, or None if error occurred
  """
  if past_review_outcomes is None:
    past_review_outcomes = []
  if checker is None:
    checker = CheckVerbalizedResponseReviewer()

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


def run_correct_response(checker: CheckVerbalizedResponseReviewer = None):
  """
  Run the test case for correct_response.
  """
  eval_input = json.dumps({
    "conversation_turns": [
      {
        "speaker": "Human",
        "message": "How much cash do I have?"
      },
      {
        "speaker": "AI",
        "message": "Let me check your accounts for you."
      },
      {
        "speaker": "Human",
        "message": "What's my total cash balance?"
      }
    ],
    "ai_review_response": "You have $27,593 in total cash across your checking and savings accounts."
  }, indent=2)
  
  review_needed = json.dumps({
    "rating": "good_response",
    "rationale": "The response directly answers the Human's last inquiry about total cash balance with a specific dollar amount, providing clear and relevant information."
  }, indent=2)
  
  return run_test_case("correct_response", eval_input, review_needed, [], checker)


def run_greeting_response(checker: CheckVerbalizedResponseReviewer = None):
  """
  Run the test case for a response that starts with a greeting in an ongoing conversation.
  """
  eval_input = json.dumps({
    "conversation_turns": [
      {
        "speaker": "Human",
        "message": "I want to save $50,000 for a down payment."
      },
      {
        "speaker": "AI",
        "message": "That's a great goal! Let me help you create a savings plan. Based on your current income and expenses, you can save about $1,200 per month."
      },
      {
        "speaker": "Human",
        "message": "How long will it take to reach my goal?"
      },
      {
        "speaker": "AI",
        "message": "With your current savings rate, it will take approximately 42 months, or about 3.5 years, to reach $50,000."
      },
      {
        "speaker": "Human",
        "message": "What's my current savings balance?"
      }
    ],
    "ai_review_response": "Hello! Your current savings balance is $8,500 across your savings accounts."
  }, indent=2)
  
  review_needed = json.dumps({
    "rating": "incoherent_flow",
    "rationale": "The response contains a conversation opener ('Hello!') in the middle of an ongoing conversation, which breaks the conversation flow and is inappropriate for an established dialogue."
  }, indent=2)
  
  return run_test_case("greeting_response", eval_input, review_needed, [], checker)


def main():
  """Main function to test the VerbalizedResponseReviewer checker"""
  checker = CheckVerbalizedResponseReviewer()
  
  # Run all tests
  run_correct_response(checker)
  run_greeting_response(checker)


if __name__ == "__main__":
  main()
