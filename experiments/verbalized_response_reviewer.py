from google import genai
from google.genai import types
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """Evaluate the `ai_review_response` for quality issues against the conversation history, prioritizing relevance to the Human's last turn.

**Input:** List of conversation turns. The final AI response to review is keyed as `ai_review_response`.

**Output:** Return JSON:
- `rating`: One rating key (see below)
- `rationale`: Brief explanation for the rating

**Rating Keys:**
- `good_response`: Straightforwardly answers the Human's last inquiry; good conversational response
- `non_sense`: Irrelevant, illogical, or has inconsistent computations
- `repetitive_information`: Repeats information within its own response without value/context change
- `incoherent_flow`: Leads to dead end or breaks conversation flow
- `verbose`: Too many unnecessary details that prevent straightforwardly answering the Human's inquiry
- `incomplete`: Too brief or lacks necessary information

**Evaluation Rules:**
- Prioritize relevance to the preceding Human turn
- Ignore contradicting values from previous conversations; use latest values from `ai_review_response`
- Evaluate the response as a whole, considering full conversation context
"""

class VerbalizedResponseReviewer:
  """Handles all Gemini API interactions for checking AI responses for red flags"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for checking AI response red flags"""
    # API Configuration
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file or environment.")
    self.client = genai.Client(api_key=api_key)
    
    # Model Configuration
    self.thinking_budget = 0
    self.model_name = model_name
    
    # Generation Configuration Constants
    self.temperature = 0.5
    self.top_p = 0.95
    self.max_output_tokens = 4096
    
    # Safety Settings
    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
    ]
    
    # System Prompt
    self.system_prompt = SYSTEM_PROMPT

  
  def evaluate_response(self, input_json: str) -> dict:
    """
    Evaluate an AI response for red flags using Gemini API.
    
    Args:
      input_json: JSON string containing conversation turns and ai_review_response.
      
    Returns:
      Dictionary containing the red flag key and rationale
    """
    # Create request text with the new input structure
    request_text_str = f"""input: {input_json}
output: """
    
    print(f"\n{'='*80}")
    print("INPUT JSON:")
    print(input_json)
    print("="*80)
    
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
    try:
      for chunk in self.client.models.generate_content_stream(
        model=self.model_name,
        contents=contents,
        config=generate_content_config,
      ):
        if chunk.text is not None:
          output_text += chunk.text
    except Exception as e:
      error_msg = f"API Error: {str(e)}"
      print(f"\n{'='*80}")
      print("ERROR:")
      print(error_msg)
      print("="*80)
      raise Exception(f"Failed to generate response from Gemini API: {error_msg}")
    
    # Check if response is empty
    if not output_text or not output_text.strip():
      raise ValueError(f"Empty response from model. Check API key and model availability.")
    
    print(f"{'='*80}")
    print("RESPONSE OUTPUT:")
    print(output_text.strip())
    print("="*80)

    # Try to parse as JSON, if it fails return as text
    try:
      # Clean up the output text - remove markdown code blocks if present
      cleaned_output = output_text.strip()
      
      # Remove markdown code blocks
      if "```json" in cleaned_output:
        json_start = cleaned_output.find("```json") + 7
        json_end = cleaned_output.find("```", json_start)
        if json_end != -1:
          cleaned_output = cleaned_output[json_start:json_end].strip()
      elif cleaned_output.startswith("```"):
        # Remove markdown code block markers
        lines = cleaned_output.split("\n")
        if lines[0].startswith("```"):
          lines = lines[1:]
        if lines and lines[-1].strip() == "```":
          lines = lines[:-1]
        cleaned_output = "\n".join(lines).strip()
      
      # Try to extract JSON object if it's embedded in text
      json_start = cleaned_output.find('{')
      json_end = cleaned_output.rfind('}') + 1
      
      if json_start != -1 and json_end > json_start:
        json_str = cleaned_output[json_start:json_end]
        return json.loads(json_str)
      else:
        # Try parsing the whole cleaned output
        return json.loads(cleaned_output)
    except json.JSONDecodeError as e:
      # If not valid JSON, return as a dict with the raw text and error info
      print(f"\nWarning: Failed to parse JSON response: {e}")
      print(f"Raw output preview: {output_text[:200]}...")
      return {"raw_output": output_text.strip(), "parse_error": str(e)}


def test_with_inputs(input_json: dict, reviewer: VerbalizedResponseReviewer = None):
  """
  Convenient method to test the reviewer with custom inputs.
  
  Args:
    input_json: Dictionary containing conversation turns and ai_review_response.
    reviewer: Optional VerbalizedResponseReviewer instance. If None, creates a new one.
    
  Returns:
    Dictionary containing the red flag key and rationale
  """
  if reviewer is None:
    reviewer = VerbalizedResponseReviewer()
  
  return reviewer.evaluate_response(json.dumps(input_json, indent=2))


def run_example_test(reviewer: VerbalizedResponseReviewer = None):
  """
  Run an example test case.
  """
  return test_with_inputs({
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
  }, reviewer)


def main():
  """Main function to test the verbalized response reviewer"""
  try:
    reviewer = VerbalizedResponseReviewer()
    result = run_example_test(reviewer)
    
    print(f"\n{'='*80}")
    print("FINAL RESULT:")
    print(json.dumps(result, indent=2))
    print("="*80)
    print("\n✅ Successfully evaluated the AI response!")
    
    return result
  except ValueError as e:
    # Handle API key or initialization errors
    print(f"\n{'='*80}")
    print("ERROR OCCURRED:")
    print(str(e))
    print("="*80)
    print("\nTroubleshooting:")
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
    print("="*80)
    
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
  main()
