from google import genai
from google.genai import types
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are an AI assistant that validates verbalizer outputs based on a strict set of rules.

## Input:
- **EVAL_INPUT**: JSON containing insights about a user's financial activity.
- **PAST_REVIEW_OUTCOMES**: A list of previous review outcomes for the same input.
- **REVIEW_NEEDED**: The verbalizer's JSON response that requires validation.

## Output:
Produce a single JSON object with the following structure: `{"good_copy": boolean, "info_correct": boolean, "eval_text": string}`
- `good_copy`: **(FORMATTING & STYLE)** Must be `true` only if `REVIEW_NEEDED` is perfectly formatted and styled according to all rules in Part 1.
- `info_correct`: **(CONTENT ACCURACY)** Must be `true` only if the information in `REVIEW_NEEDED` is factually correct and consistent based on `EVAL_INPUT` and rules in Part 2.
- `eval_text`: **(REQUIRED ON FAILURE)** If either `good_copy` or `info_correct` is `false`, provide a concise explanation here. Refer to insights by number if applicable (e.g., "Insight 1: ..."). Do not refer to rules by number.

## Core Directives:
1.  **Strictness is Paramount**: Prioritize recall over precision. If you have any doubt, mark the check as `false`. It is better to incorrectly flag a potential issue than to miss a real one.
2.  **Learn from History**: Before checking the current `REVIEW_NEEDED`, analyze `PAST_REVIEW_OUTCOMES`. If `REVIEW_NEEDED` repeats any mistake from a past `eval_text`, it is an automatic failure.

## Rules

### Part 1: Formatting and Copy Rules (`good_copy`)
1.  **Valid JSON**: The output must be a single, valid JSON array.
2.  **Currency Formatting**: All numbers representing currency must be formatted with commas and no decimals (e.g., "$1,234").
3.  **Tone**: The tone should be encouraging and light. It should be celebratory only for positive insights.
4.  **Title (`title`)**:
    -   Must be under 30 characters.
    -   Must accurately represent all key points in the `summary`.
5.  **Summary (`summary`)**:
    -   Must not contain conversational openers (e.g., "Hi", "Hello").
    -   Must clearly indicate the direction of financial changes (e.g., "higher than", "lower than").
    -   Must be clear and understandable on its own.

### Part 2: Content Rules (`info_correct`)
1.  **ID Matching**: The `id` in `REVIEW_NEEDED` must match the `id` and order from `EVAL_INPUT`.
2.  **Factual Accuracy**: All information must be accurate based on `EVAL_INPUT`. It is acceptable to omit some details from `EVAL_INPUT` for conciseness.
3.  **No External Information**: The response must be derived solely from `EVAL_INPUT`.
4.  **Internal Consistency**: All parts of the `REVIEW_NEEDED` output must be consistent with each other. For example, numbers or statements in the `summary` should not contradict each other.

## Verification Workflow:
1.  **Analyze Past Failures**: Review `PAST_REVIEW_OUTCOMES`. If `REVIEW_NEEDED` has repeated errors, fail it immediately.
2.  **Validate Formatting (`good_copy`)**: Scrutinize `REVIEW_NEEDED` against all rules in "Part 1".
3.  **Validate Content (`info_correct`)**: Scrutinize `REVIEW_NEEDED` against all rules in "Part 2".
4.  **Generate `eval_text`**: If any validation fails, write a clear and specific explanation.
"""

class CheckVerbalizerTextWithMemory:
  """Handles all Gemini API interactions for checking VerbalizerTextWithMemory outputs against rules"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for checking VerbalizerTextWithMemory evaluations"""
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
    Generate a response using Gemini API for checking P:Func:VerbalizerTextWithMemory outputs.
    
    Args:
      eval_input: String containing "**User request**:" followed by the user's financial goal/request, and "**Input Information from previous skill**:" followed by detailed financial data (savings balance, accounts, past transactions, forecasted patterns, savings rate).
      past_review_outcomes: An array of past review outcomes, each containing `output`, `good_copy`, `info_correct`, and `eval_text`.
      review_needed: The P:Func:VerbalizerTextWithMemory output that needs to be reviewed (string).
      
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
      thinking_config=types.ThinkingConfig(
        thinking_budget=self.thinking_budget,
        include_thoughts=True
      ),
    )

    # Generate response
    output_text = ""
    thought_summary = ""
    
    # According to Gemini API docs: iterate through chunks and check part.thought boolean
    for chunk in self.client.models.generate_content_stream(
      model=self.model_name,
      contents=contents,
      config=generate_content_config,
    ):
      # Extract text content (non-thought parts)
      if chunk.text is not None:
        output_text += chunk.text
      
      # Extract thought summary from chunk
      if hasattr(chunk, 'candidates') and chunk.candidates:
        for candidate in chunk.candidates:
          # Extract thought summary from parts (per Gemini API docs)
          # Check part.thought boolean to identify thought parts
          if hasattr(candidate, 'content') and candidate.content:
            if hasattr(candidate.content, 'parts') and candidate.content.parts:
              for part in candidate.content.parts:
                # Check if this part is a thought summary (per documentation)
                if hasattr(part, 'thought') and part.thought:
                  if hasattr(part, 'text') and part.text:
                    # Accumulate thought summary text (for streaming, it may come in chunks)
                    if thought_summary:
                      thought_summary += part.text
                    else:
                      thought_summary = part.text
    
    # Check if response is empty
    if not output_text or not output_text.strip():
      raise ValueError(f"Empty response from model. Check API key and model availability.")
    
    if thought_summary:
      print(f"{'='*80}")
      print("THOUGHT SUMMARY:")
      print(thought_summary.strip())
      print("="*80)
    
    print(f"{'='*80}")
    print("RESPONSE OUTPUT:")
    print(output_text.strip())
    print("="*80)
    
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


def run_test_case(test_name: str, eval_input: str, review_needed: str, past_review_outcomes: list = None, checker: 'CheckVerbalizerTextWithMemory' = None):
  """
  Utility function to run a test case with custom inputs, common error handling, and output formatting.
  
  Args:
    test_name: Name of the test case
    eval_input: String containing "**User request**:" followed by the user's financial goal/request, and "**Input Information from previous skill**:" followed by detailed financial data.
    review_needed: The P:Func:VerbalizerTextWithMemory output that needs to be reviewed (string).
    past_review_outcomes: An array of past review outcomes, each containing `output`, `good_copy`, `info_correct`, and `eval_text`. Defaults to empty list.
    checker: Optional CheckVerbalizerTextWithMemory instance. If None, creates a new one.
    
  Returns:
    Dictionary with good_copy, info_correct, and eval_text keys, or None if error occurred
  """
  if past_review_outcomes is None:
    past_review_outcomes = []
  if checker is None:
    checker = CheckVerbalizerTextWithMemory()

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


def run_correct_response(checker: CheckVerbalizerTextWithMemory = None):
  """
  Run the test case for correct_response.
  """
  eval_input = """[
  {
    "id": 1,
    "combined_insight": "Your shelter costs are way down this month to $1,248, mainly from less on home stuff, utilities, and upkeep. 🥳🏠 Oh em gee!  You got a huge surprise income boost of $8,800 this week, mostly from your business, and you're projected to spend only $68 by the end of the week!  Way to go, you savvy boss babe!"
  },
  {
    "id": 2,
    "combined_insight": "Looks like you spent less on food this month, down to $1,007, mostly from less eating out, deliveries, and groceries. 🍽️🚚🛒 Your transport costs are way down this month to just $46, mostly 'cause you took public transit less. 🚇"
  }
]"""
  
  review_needed = """[
  {
    "id": 1,
    "title": "Savings! 🏡💰 Boost! 🤩",
    "summary": "Yay! 🎉 Your shelter costs dropped to $1,248 this month, and you're crushing it with an $8,800 income boost and only $68 projected spending! 🤩💰"
  },
  {
    "id": 2,
    "title": "🍽️🚆 Food & Travel Win! 🥳👏",
    "summary": "Nice one! 🙌 You spent less on food, down to $1,007, and transport is only $46 this month! 🥳 Less eating out and fewer train trips pay off, girl! 👏"
  }
]"""
  
  return run_test_case("correct_response", eval_input, review_needed, [], checker)



def main():
  """Main function to test the HighlightsVerbalizerJson checker"""
  checker = CheckVerbalizerTextWithMemory()
  
  # Run all tests
  run_correct_response(checker)


if __name__ == "__main__":
  main()

