from google import genai
from google.genai import types
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SCHEMA = types.Schema(
  type=types.Type.OBJECT,
  properties={
    "name_ideas": types.Schema(
      type=types.Type.ARRAY,
      items=types.Schema(type=types.Type.STRING),
      description="A ranked list of short name suggestions user would prefer to be called"),
  },
  required=["name_ideas",]
)

SYSTEM_PROMPT = """#### 1. Role & Goal
You are a heuristic name generator that analyzes user account information to suggest appropriate short names that the user might prefer to be called. Your goal is to generate a ranked list of safe, friendly, and appropriate name suggestions based on the provided account information.

#### 2. Core Task
Your task is to analyze the provided user account information and generate a ranked list of preferred short names. The names should:
- Be derived from the account_name (e.g., first name, common nickname)
- Be safe and appropriate for use in a financial advisor context
- Be friendly and personable
- Follow common naming conventions and preferences

#### 3. Input Data
You will be provided with a JSON object containing:
- `account_name`: The full account name of the user (e.g., "Dan Delima")
- `email_address`: The user's email address (e.g., "kranasian@gmail.com")

#### 4. Output Requirements
- **Structure:** Return a JSON object with a single key `name_ideas` containing an array of strings.
- **Ranking:** The most preferred/likely name should be first in the array.
- **Format:** Each name should be a simple string (e.g., "Dan", "Daniel").
- **Quantity:** Provide 1-3 name suggestions, prioritizing the most likely preferred name.
- **Safety:** Only suggest appropriate, professional names suitable for a financial advisor context.

#### 5. Critical Constraints
- **Use Input Data Only:** Base suggestions only on the provided account_name and email_address.
- **No External Knowledge:** Do not use information outside of the provided JSON object.
- **Common Patterns:** Prefer first names over last names, common nicknames over unusual variations.
- **Professional Context:** Ensure all suggestions are appropriate for a financial advisor to use.
"""

class UserInformationHeuristicsOptimizer:
  """Handles all Gemini API interactions for generating user name suggestions based on account information"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for generating user name suggestions"""
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
    
    # Output Schema - array of result objects
    self.output_schema = SCHEMA
  
  def generate_name_suggestions(self, input_json: dict) -> dict:
    """
    Generate name suggestions using Gemini API based on user account information.
    
    Args:
      input_json: Dictionary containing account_name and email_address.
      
    Returns:
      Dictionary containing name_ideas array with ranked name suggestions
    """
    # Create request text with the input structure
    request_text_str = f"""input: {json.dumps(input_json, indent=2)}
output: """
    
    print(f"\n{'='*80}")
    print("INPUT JSON:")
    print(json.dumps(input_json, indent=2))
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
      response_schema=self.output_schema,
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
    
    print(f"{'='*80}")
    print("RESPONSE OUTPUT:")
    print(output_text.strip())
    print("="*80)

    # Parse JSON response
    try:
      # Try to extract JSON from the response (in case there's extra text)
      output_text_clean = output_text.strip()
      # Remove markdown code blocks if present
      if output_text_clean.startswith("```"):
        lines = output_text_clean.split("\n")
        output_text_clean = "\n".join(lines[1:-1]) if len(lines) > 2 else output_text_clean
      if output_text_clean.startswith("```json"):
        lines = output_text_clean.split("\n")
        output_text_clean = "\n".join(lines[1:-1]) if len(lines) > 2 else output_text_clean
      
      result = json.loads(output_text_clean)
      return result
    except json.JSONDecodeError as e:
      raise ValueError(f"Failed to parse JSON response: {e}\nResponse text: {output_text}")


def test_with_inputs(input_json: dict, optimizer: UserInformationHeuristicsOptimizer = None):
  """
  Convenient method to test the optimizer with custom inputs.
  
  Args:
    input_json: Dictionary containing account_name and email_address.
    optimizer: Optional UserInformationHeuristicsOptimizer instance. If None, creates a new one.
    
  Returns:
    Dictionary containing name_ideas array with ranked name suggestions
  """
  if optimizer is None:
    optimizer = UserInformationHeuristicsOptimizer()
  
  return optimizer.generate_name_suggestions(input_json)


def run_test_dan_delima(optimizer: UserInformationHeuristicsOptimizer = None):
  """
  Run the test case for Dan Delima.
  """
  return test_with_inputs({
    "account_name": "Dan Delima",
    "email_address": "kranasian@gmail.com"
  }, optimizer)


def run_test_angel_rodriguez(optimizer: UserInformationHeuristicsOptimizer = None):
  """
  Run the test case for Angel Rodriguez.
  """
  return test_with_inputs({
    "account_name": "Angel Rodriguez",
    "email_address": "angel.r@gmail.com"
  }, optimizer)


def run_test_michael_johnson(optimizer: UserInformationHeuristicsOptimizer = None):
  """
  Run the test case for Michael Johnson.
  """
  return test_with_inputs({
    "account_name": "Michael Johnson",
    "email_address": "mike.johnson@example.com"
  }, optimizer)


def main():
  """Main function to test the UserInformationHeuristicsOptimizer"""
  optimizer = UserInformationHeuristicsOptimizer()
  
  print("\n" + "="*80)
  print("TEST 1: Dan Delima")
  print("="*80)
  result1 = run_test_dan_delima(optimizer)
  print(f"\nResult: {json.dumps(result1, indent=2)}")
  
  print("\n" + "="*80)
  print("TEST 2: Angel Rodriguez")
  print("="*80)
  result2 = run_test_angel_rodriguez(optimizer)
  print(f"\nResult: {json.dumps(result2, indent=2)}")
  
  print("\n" + "="*80)
  print("TEST 3: Michael Johnson")
  print("="*80)
  result3 = run_test_michael_johnson(optimizer)
  print(f"\nResult: {json.dumps(result3, indent=2)}")


if __name__ == "__main__":
  main()

