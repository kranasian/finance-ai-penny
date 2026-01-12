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

SYSTEM_PROMPT = """Task: Transform Food Establishment JSON to Attribute JSON.

Mapping Rules:
1. Copy `id`.
2. `primary`: List [1+ items]. Options: "Fast food", "Restaurant", "Beverage", "Grocery".
   - Inference: "Tea/Coffee" -> "Beverage". "Market/Mart" -> "Grocery". "Fast-casual" -> "Fast food".
3. `secondary`: List [3-7 items]. Extract specific Cuisine, Dish, or Style.
   - Example: "Sells tacos" -> ["Mexican", "Tacos"].
   - Use singular nouns and avoid plurals.

Output: JSON Array only."""

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
      max_output_tokens=self.max_output_tokens,
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


def run_test_first_set(optimizer: FoodSpendAddAttributesOptimizer = None):
  """
  Run the test case for Starbucks.
  """
  return test_with_inputs([
    {
      "id": 342,
      "name": "Snack Tiger Tea",
      "description": "A purchase of snacks and beverages from a cafe."
    },
    {
      "id": 567,
      "name": "San Froyo",
      "description": "This establishment sells frozen yogurt and other dessert items."
    },
    {
      "id": 891,
      "name": "Manila Bay Cuisine",
      "description": "sells Filipino dishes"
    },
    {
      "id": 234,
      "name": "Boudin Stonestown",
      "description": "This establishment sells baked goods and cafe items."
    },
    {
      "id": 678,
      "name": "Chipotle",
      "description": "Fast-casual Mexican restaurant serving burritos, bowls, and tacos"
    }
  ], optimizer)


def run_test_second_set(optimizer: FoodSpendAddAttributesOptimizer = None):
  """
  Run the test case with multiple establishments.
  """
  return test_with_inputs([
    {
      "id": 123,
      "name": "Trader Joe's",
      "description": "sells a variety of groceries, including private-label products, organic produce, and prepared foods"
    },
    {
      "id": 456,
      "name": "Nijiya Market",
      "description": "sells Japanese groceries, including fresh produce, seafood, meat, snacks, and other items"
    },
    {
      "id": 789,
      "name": "Panda Express",
      "description": "fast-food restaurant chain that serves American Chinese cuisine"
    },
    {
      "id": 135,
      "name": "Burning Mouth",
      "description": "This establishment sells spicy food."
    },
    {
      "id": 246,
      "name": "Goldilocks Bakeshop",
      "description": "sells cakes, pastries, breads, and other baked goods"
    },
    {
      "id": 369,
      "name": "Ramen Nagi",
      "description": "sells ramen noodles and other Japanese dishes"
    }
  ], optimizer)


def main(batch: int = 1):
  """
  Main function to test the FoodSpendAddAttributesOptimizer
  
  Args:
    batch: Batch number (1 or 2) to determine which tests to run
  """
  print("Testing FoodSpendAddAttributesOptimizer\n")
  
  if batch == 1:
    # Batch 1: Basic establishment types
    optimizer = FoodSpendAddAttributesOptimizer()
    print("Test 1: Starbucks")
    print("-" * 80)
    run_test_first_set(optimizer)
    
    print("Test 2: Multiple Establishments")
    print("-" * 80)
    run_test_second_set(optimizer)
        
  # elif batch == 2:
    # # Batch 2: Restaurant and multiple establishments
    # optimizer = FoodSpendAddAttributesOptimizer()
    # print("Test 1: Italian Restaurant")
    # print("-" * 80)
    # result4 = run_test_italian_restaurant(optimizer)
    # print(f"\nResult: {json.dumps(result4, indent=2)}")
    # print("\n")
  else:
    raise ValueError("batch must be 1 or 2")
  
  print("All tests completed!")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Run food spend attribute tests in batches')
  parser.add_argument('--batch', type=int, default=1, choices=[1, 2],
                      help='Batch number to run (1 for basic types, 2 for restaurant and multiple establishments)')
  args = parser.parse_args()
  main(batch=args.batch)

