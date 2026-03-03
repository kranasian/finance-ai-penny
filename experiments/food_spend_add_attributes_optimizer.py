from google import genai
from google.genai import types
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SCHEMA = types.Schema(
  type=types.Type.ARRAY,
  items=types.Schema(
    type=types.Type.OBJECT,
    properties={
      "id": types.Schema(type=types.Type.NUMBER),
      "primary": types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(type=types.Type.STRING),
        description="List containing at least one of: Fast food, Restaurant, Beverage, Grocery"
      ),
      "secondary": types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(type=types.Type.STRING),
        description="List of 2-5 descriptive tags (Cuisine, Dish, Style)"
      ),
    },
    required=["id", "primary", "secondary"]
  )
)

SYSTEM_PROMPT = """Task: Analyze food establishment data to determine primary category and secondary attributes.

Input: JSON list of establishments (id, name, description).
Output: JSON list of attributes (id, primary, secondary).

Definitions:
- `primary`: Select one or more from ["Fast food", "Restaurant", "Beverage", "Grocery"].
- `secondary`: Extract 3-7 specific tags (Cuisine, Dish, Style, or Type) from the Name and Description.

Constraints:
- Use singular nouns for `secondary` tags.
- STRICTLY EXCLUDE standalone generic terms: "Food", "Drink", "Fare", "Meal", "Dish", "Beverage", "Cuisine", "Restaurant", "Store".
- Allow compound terms (e.g., "Frozen meal", "Health food").
- Infer attributes from the Name if relevant.
- MUST NOT repeat `primary` categories in `secondary`.
- MANDATORY: Ensure at least 3 `secondary` tags. If specific details are missing, infer broad categories (e.g., "Chain", "Retail", "Shop", "Dining", "Asian", "European").
- Output JSON only."""

class FoodSpendAddAttributesOptimizer:
  """Handles all Gemini API interactions for generating food establishment attributes based on establishment information"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for generating establishment attributes"""
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
    self.top_k = 40
    self.max_output_tokens = 4096
    self.response_mime_type = "application/json"
    
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
  
  def generate(self, establishments: list) -> list:
    """
    Generate attributes using Gemini API based on establishment information.
    
    Args:
      establishments: List of dictionaries, each containing id, name, and description.
      
    Returns:
      List of dictionaries, each containing id, primary, and secondary
    """
    # Create request text with the input structure
    request_text_str = f"""input: {json.dumps(establishments, indent=2)}
output: """
    
    print(f"\n{'='*80}")
    print("INPUT JSON:")
    print(json.dumps(establishments, indent=2))
    print("="*80)
    
    request_text = types.Part.from_text(text=request_text_str)
    
    # Create content and configuration
    contents = [types.Content(role="user", parts=[request_text])]
    
    generate_content_config = types.GenerateContentConfig(
      temperature=self.temperature,
      top_p=self.top_p,
      top_k=self.top_k,
      max_output_tokens=self.max_output_tokens,
      response_mime_type=self.response_mime_type,
      safety_settings=self.safety_settings,
      system_instruction=[types.Part.from_text(text=self.system_prompt)],
      thinking_config=types.ThinkingConfig(
        thinking_budget=self.thinking_budget,
        include_thoughts=True
      ),
      response_schema=self.output_schema,
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


def test_with_inputs(establishments: list, optimizer: FoodSpendAddAttributesOptimizer = None):
  """
  Convenient method to test the optimizer with custom inputs.
  
  Args:
    establishments: List of dictionaries, each containing id, name, and description.
    optimizer: Optional FoodSpendAddAttributesOptimizer instance. If None, creates a new one.
    
  Returns:
    List of dictionaries, each containing id, primary, and secondary
  """
  if optimizer is None:
    optimizer = FoodSpendAddAttributesOptimizer()
  
  return optimizer.generate(establishments)


def run_test_set_1(optimizer: FoodSpendAddAttributesOptimizer = None):
  return test_with_inputs([
    { "id": 101, "name": "Erewhon Market", "description": "sells organic and natural foods, groceries, produce, and health and wellness products" },
    { "id": 102, "name": "Waffle House", "description": "A restaurant chain serving American diner fare." },
    { "id": 103, "name": "Blue Apron", "description": "sends boxes of food to houses for customers to cook" }
  ], optimizer)

def run_test_set_2(optimizer: FoodSpendAddAttributesOptimizer = None):
  return test_with_inputs([
    { "id": 201, "name": "7-Eleven", "description": "convenience store chain sells snacks, drinks, groceries, hot food, and other everyday items" },
    { "id": 202, "name": "Korean House", "description": "A purchase from a Korean restaurant for food and beverages." },
    { "id": 203, "name": "Nobu", "description": "Japanese restaurant that serves sushi, sashimi, and other Japanese-Peruvian fusion dishes" }
  ], optimizer)

def run_test_set_3(optimizer: FoodSpendAddAttributesOptimizer = None):
  return test_with_inputs([
    { "id": 901, "name": "Costco", "description": "sells bulk groceries, electronics, furniture, and other goods at discounted prices with a membership fee" },
    { "id": 902, "name": "Daily Harvest", "description": "delivers frozen plant-based meals" },
    { "id": 903, "name": "Lidl", "description": "Payment for groceries and household items from a discount supermarket." }
  ], optimizer)

def run_test_set_4(optimizer: FoodSpendAddAttributesOptimizer = None):
  return test_with_inputs([
    { "id": 2001, "name": "Shake Shack", "description": "fast-casual restaurant chain that serves burgers, hot dogs, fries, and shakes" },
    { "id": 2002, "name": "Marugame Udon", "description": "This is a credit from a Japanese restaurant specializing in udon noodles and tempura." },
    { "id": 2003, "name": "Harry & David", "description": "sends fruit baskets and gourmet gifts" }
  ], optimizer)


def main(batch: int = 1):
  """
  Main function to test the FoodSpendAddAttributesOptimizer
  
  Args:
    batch: Batch number (1-4) to determine which tests to run
  """
  print(f"Testing FoodSpendAddAttributesOptimizer - Batch {batch}\n")
  
  optimizer = FoodSpendAddAttributesOptimizer()
  
  if batch == 1:
    run_test_set_1(optimizer)
  elif batch == 2:
    run_test_set_2(optimizer)
  elif batch == 3:
    run_test_set_3(optimizer)
  elif batch == 4:
    run_test_set_4(optimizer)
  else:
    raise ValueError("batch must be between 1 and 4")
  
  print("Test completed!")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Run food spend attribute tests in batches')
  parser.add_argument('--batch', type=int, default=1, choices=[1, 2, 3, 4],
                      help='Batch number to run (1-4)')
  args = parser.parse_args()
  main(batch=args.batch)
