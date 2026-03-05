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
        description="List containing at least one of: Fast food, Restaurant, Beverage, Grocery, Dessert. Borderline places can have multiple (e.g., both Restaurant and Beverage)."
      ),
      "secondary": types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(type=types.Type.STRING),
        description="List of 2-5 HIGH-VALUE business categories or precise descriptors. MUST NOT contain: food, dish, cuisine, snack, meal, heat, eatery, entree."
      ),
    },
    required=["id", "primary", "secondary"]
  )
)

SYSTEM_PROMPT = """Task: Transform Food Establishment JSON to Attribute JSON utilizing standard business taxonomy.

Mapping Rules:
1. Copy `id`.
2. `primary`: List [1+ items]. Options: "Fast food", "Restaurant", "Beverage", "Grocery", "Dessert".
   - Inference: "Tea/Coffee" -> "Beverage". "Market/Mart" -> "Grocery". "Fast-casual" -> "Fast food".
   - IF borderline (e.g., cafe with bakery), INCLUDE ALL relevant (e.g., "Restaurant", "Beverage").
3. `secondary`: List [3-7 items]. Extract highly specific, high-value Google Places/Yelp-style category tags.
   - For Fast food & Restaurants: Extract specific Cuisine (e.g. "Mexican", "Filipino"), key recognizable items (e.g. "Burgers", "Tacos"), or Dining Style.
   - For Beverage: Extract specific Types (e.g., "Coffee Shop", "Boba", "Juice Bar").
   - For Grocery: Extract specialty (e.g., "Japanese Market", "Organic").
   - For Dessert: Extract specific type (e.g., "Frozen Yogurt", "Donut Shop").
   - Sparse Descriptions: If info is sparse, infer high-value descriptive characteristics (e.g., flavor profiles like "Spicy") without generic nouns.
   - CRITICAL NEGATIVE CONSTRAINTS: Tags MUST NOT contain the following words (even as substrings):
     "food", "dish", "cuisine", "snack", "meal", "eatery", "appetizer", "entree", "heat", "place".
   - ONLY use standard singular nouns for items (e.g. "Burger"), flavor descriptors, or established category names (e.g. "Coffee Shop").

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


TEST_CASES = [
  {
    "name": "Test 1: Basic establishment types",
    "establishments": [
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
      }
    ],
    "ideal_response": """[
  {"id": 342, "primary": ["Beverage"], "secondary": ["Boba", "Tea", "Cafe"]},
  {"id": 567, "primary": ["Dessert"], "secondary": ["Frozen yogurt", "Frozen dessert"]},
  {"id": 891, "primary": ["Restaurant"], "secondary": ["Filipino", "Rice dish", "Family style"]}
]
Key validations:
- Expansive attributes that list attributes of the establishment
- Too "generic" tags like "Snack", "Dish" must not be included."""
  },
  {
    "name": "Test 2: Basic establishment types",
    "establishments": [
      {
        "id": 234,
        "name": "Boudin Stonestown",
        "description": "This establishment sells baked goods and cafe items."
      },
      {
        "id": 678,
        "name": "Chipotle",
        "description": "Fast-casual Mexican restaurant serving burritos, bowls, and tacos"
      },
      {
        "id": 123,
        "name": "Trader Joe's",
        "description": "sells a variety of groceries, including private-label products, organic produce, and prepared foods"
      }
    ],
    "ideal_response": """[
  {"id": 234, "primary": ["Restaurant", "Beverage"], "secondary": ["Baked goods", "Coffee", "Pastry", "Bread"]},
  {"id": 678, "primary": ["Fast food"], "secondary": ["Mexican", "Burrito", "Bowl", "Taco"]},
  {"id": 123, "primary": ["Grocery"], "secondary": ["Private-label", "Organic", "Prepared food"]}
]
Key validations:
- Singular nouns only
- Borderline primary can have multiple primary attributes."""
  },
  {
    "name": "Test 2: Multiple Establishments",
    "establishments": [
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
      }
    ],
    "ideal_response": """[
  {"id": 456, "primary": ["Grocery"], "secondary": ["Japanese", "Specialty"]},
  {"id": 789, "primary": ["Fast food"], "secondary": ["American Chinese"]},
  {"id": 135, "primary": ["Restaurant"], "secondary": ["Spicy food", "Burgers""]}
]
Key validations:
- Relevant 2-5 tags that is diverse and might not even be in the title.
- Too "generic" tags like "Food" or "Cuisine" must not be included."""
  },
#   {
#     "name": "Test 3: Multiple Establishments",
#     "establishments": [
#       {
#         "id": 246,
#         "name": "Goldilocks Bakeshop",
#         "description": "sells cakes, pastries, breads, and other baked goods"
#       },
#       {
#         "id": 369,
#         "name": "Ramen Nagi",
#         "description": "sells ramen noodles and other Japanese dishes"
#       }
#     ],
#     "ideal_response": """# [
#   {"id": 246, "primary": ["Restaurant", "Fast food"], "secondary": ["Cake", "Pastry", "Bread", "Baked goods"]},
#   {"id": 369, "primary": ["Restaurant"], "secondary": ["Ramen", "Japanese"]}
# ]
# Key validations:
# - Singular nouns
# - Relevant 2-5 tags"""
#   }
]


def test_with_inputs(test_name_or_index_or_dict, optimizer: FoodSpendAddAttributesOptimizer = None):
  """
  Convenient method to test the optimizer with custom inputs.
  
  Args:
    test_name_or_index_or_dict: Test case index, name or dict (Wait, could be a list of establishments directly for backward compatibility)
    optimizer: Optional FoodSpendAddAttributesOptimizer instance. If None, creates a new one.
    
  Returns:
    List of dictionaries, each containing id, primary, and secondary
  """
  if optimizer is None:
    optimizer = FoodSpendAddAttributesOptimizer()
    
  test_case = None
  
  # Backward compatibility: handle a direct list of establishments
  if isinstance(test_name_or_index_or_dict, list):
    return optimizer.generate(test_name_or_index_or_dict)
    
  if isinstance(test_name_or_index_or_dict, dict):
    if "establishments" in test_name_or_index_or_dict:
      test_case = test_name_or_index_or_dict
  elif isinstance(test_name_or_index_or_dict, int):
    if 0 <= test_name_or_index_or_dict < len(TEST_CASES):
      test_case = TEST_CASES[test_name_or_index_or_dict]
  elif isinstance(test_name_or_index_or_dict, str):
    for tc in TEST_CASES:
      if tc["name"] == test_name_or_index_or_dict:
        test_case = tc
        break
        
  if not test_case:
    print(f"Test case '{test_name_or_index_or_dict}' not found.")
    return None


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
<<<<<<< HEAD
    batch: Batch number (1-4) to determine which tests to run
=======
    batch: Batch number to run
>>>>>>> 868b18209c1e7b8725a147144a0138406ac8a3eb
  """
  print(f"Testing FoodSpendAddAttributesOptimizer - Batch {batch}\n")
  
  optimizer = FoodSpendAddAttributesOptimizer()
  
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
