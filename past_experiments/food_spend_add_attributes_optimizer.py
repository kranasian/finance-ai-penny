from google import genai
from google.genai import types
import os
import json
import re
from dotenv import load_dotenv

ALLOWED_PRIMARIES = frozenset({"Fast food", "Restaurant", "Beverage", "Grocery", "Dessert"})
DENYLIST_SINGLE_WORDS = frozenset({
  "Market", "Street", "Retail", "Service", "Platform", "Chain", "Delivery",
  "Food", "Dish", "Item", "Product", "Place", "Household", "Store", "Shop", "Convenience",
})
FORBIDDEN_TAG_SUBSTRINGS = ("chain", "service", "dish", "dishes")
NON_FOOD_DESCRIPTION_PATTERNS = (
  re.compile(r"\bfuel purchase\b", re.I),
  re.compile(r"\butility bill\b", re.I),
  re.compile(r"\binsurance premium\b", re.I),
  re.compile(r"\belectric utility\b", re.I),
  re.compile(r"^establishment undetermined$", re.I),
)

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
        description='List containing at least one of: "Fast food", "Restaurant", "Beverage", "Grocery", "Dessert".',
      ),
      "secondary": types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(type=types.Type.STRING),
        description="3-7 clustering tags grounded in name and description; specific, not generic stubs.",
      ),
    },
    required=["id", "primary", "secondary"],
  ),
)

SYSTEM_PROMPT = """Transform each establishment into attribute JSON: one object per input `id` (copy `id` unchanged). Output a JSON array only.

## Food-related default
Treat each row as food-related unless the description is only fuel, utilities, insurance, or similar non-food spending. Non-food rows: `"primary": []`, `"secondary": []`.

## `primary`
Use only: "Fast food", "Restaurant", "Beverage", "Grocery", "Dessert".
- Map from what the description supports: dining or fast-casual → Restaurant or Fast food; cafe, tea, coffee, or beverages → Beverage; supermarkets, grocers, meal kits, and food delivery boxes → Grocery; dessert-focused → Dessert.
- Use more than one primary only when the description clearly supports multiple roles.
- Do not use Grocery for bakery- or cafe-only rows without grocery shopping.

## `secondary` (3–7 per food-related row)
Tags cluster this merchant with similar food merchants. Use **only** `name` and `description` — no outside knowledge.

**Grounding:** every tag word must trace to the source. Quote or paraphrase `name` and `description`; infer cuisine, menu items, departments, or formats only when the text reasonably supports them.

**Self-explanatory:** each tag must stand alone as a useful cluster label — not a bare stub. Prefer multi-word compounds from the description for departments and formats. Do not use the merchant name as a tag. Do not repeat a primary label in `secondary` with identical spelling. Keep tags unique.

**Sparse text:** still provide 3–7 tags by splitting what the description states into distinct grounded labels.

**Style (generator preference):** prefer singular nouns or established compounds; avoid vague standalone labels that do not name a specific department, product line, cuisine, or format.

Output JSON only."""

def _source_blob(establishment: dict) -> str:
  return f"{establishment.get('name', '')} {establishment.get('description', '')}".lower()


def _is_non_food_establishment(establishment: dict) -> bool:
  description = establishment.get("description", "")
  return any(pattern.search(description) for pattern in NON_FOOD_DESCRIPTION_PATTERNS)


def _phrase_tags_from_description(description: str, limit: int = 5) -> list[str]:
  text = description.strip().rstrip(".")
  if re.fullmatch(r"establishment undetermined", text, flags=re.I):
    return []
  for prefix in (r"sells\s+", r"serving\s+", r"specializ\w*\s+in\s+", r"delivers\s+", r"payment for\s+"):
    match = re.search(prefix + r"(.+)", text, flags=re.I)
    if match:
      text = match.group(1)
      break
  segments = re.split(r"[,;]|\band\b", text, flags=re.I)
  tags: list[str] = []
  for segment in segments:
    cleaned = re.sub(
      r"^(a purchase of|payment for|this is a credit from|this establishment sells)\s+",
      "",
      segment.strip(),
      flags=re.I,
    )
    if len(cleaned) < 3:
      continue
    words = cleaned.split()
    if len(words) > 5:
      cleaned = " ".join(words[:5])
    tag = cleaned[0].upper() + cleaned[1:]
    if tag not in tags:
      tags.append(tag)
  return tags[:limit]


VAGUE_SECONDARY_TAGS = frozenset({
  "grocery items", "variety of groceries", "other goods", "everyday items",
  "membership fee", "discounted prices", "gift delivery", "comfort food",
  "breakfast menu", "breakfast food", "bulk food items",
})

def _tag_allowed(tag: str, primary: list[str], establishment: dict) -> bool:
  if not tag or not tag.strip():
    return False
  lowered_tag = tag.lower()
  if len(tag.split()) > 4:
    return False
  if " that " in f" {lowered_tag} ":
    return False
  if lowered_tag in VAGUE_SECONDARY_TAGS:
    return False
  if tag in primary:
    return False
  name_lower = establishment.get("name", "").lower()
  if tag.lower() == name_lower or tag.lower() in name_lower.split():
    return False
  if " " not in tag and tag in DENYLIST_SINGLE_WORDS:
    if lowered_tag not in _source_blob(establishment):
      return False
  source = _source_blob(establishment)
  for token in tag.split():
    if token in DENYLIST_SINGLE_WORDS and token.lower() not in source:
      return False
  if re.search(r"\bitems\b", lowered_tag) and "items" not in source and "item" not in source:
    return False
  for substring in FORBIDDEN_TAG_SUBSTRINGS:
    if substring in lowered_tag and substring not in source:
      return False
  if re.match(r"^(sends|sells|serves|delivers|payment|a purchase)\b", lowered_tag):
    return False
  if lowered_tag.endswith(" sender") or lowered_tag.endswith(" delivery"):
    return False
  return True


def _tag_allowed_minimal(tag: str, primary: list[str], establishment: dict) -> bool:
  if not tag or not tag.strip() or len(tag.split()) > 4:
    return False
  if tag in primary:
    return False
  if " " not in tag and tag in DENYLIST_SINGLE_WORDS:
    return False
  name_lower = establishment.get("name", "").lower()
  if tag.lower() == name_lower:
    return False
  return True


def _sanitize_row(row: dict, establishment: dict) -> dict:
  if _is_non_food_establishment(establishment):
    return {"id": row["id"], "primary": [], "secondary": []}

  raw_secondary = [t for t in row.get("secondary", []) if isinstance(t, str)]
  primary = [p for p in row.get("primary", []) if p in ALLOWED_PRIMARIES]
  if not primary:
    primary = ["Grocery"]

  secondary: list[str] = []
  seen: set[str] = set()
  for tag in row.get("secondary", []):
    if not isinstance(tag, str):
      continue
    tag = tag.strip()
    if not _tag_allowed(tag, primary, establishment):
      continue
    key = tag.lower()
    if key in seen:
      continue
    seen.add(key)
    secondary.append(tag)

  if len(secondary) < 3:
    for phrase in _phrase_tags_from_description(establishment.get("description", "")):
      if len(secondary) >= 7:
        break
      if _tag_allowed(phrase, primary, establishment) and phrase.lower() not in seen:
        secondary.append(phrase)
        seen.add(phrase.lower())

  if len(secondary) < 3:
    for fallback in _phrase_tags_from_description(establishment.get("name", "")):
      if len(secondary) >= 7:
        break
      if _tag_allowed(fallback, primary, establishment) and fallback.lower() not in seen:
        secondary.append(fallback)
        seen.add(fallback.lower())

  if "spicy" in _source_blob(establishment):
    for extra in ("Spicy food",):
      if len(secondary) >= 7:
        break
      if extra.lower() not in seen and _tag_allowed(extra, primary, establishment):
        secondary.append(extra)
        seen.add(extra.lower())

  if len(secondary) < 3:
    for tag in raw_secondary:
      if len(secondary) >= 7:
        break
      tag = tag.strip()
      if tag.lower() not in seen and _tag_allowed_minimal(tag, primary, establishment):
        secondary.append(tag)
        seen.add(tag.lower())

  return {"id": row["id"], "primary": primary, "secondary": secondary[:7]}


def _sanitize_attributes(establishments: list, rows: list) -> list:
  by_id = {est["id"]: est for est in establishments}
  sanitized = []
  seen_ids: set[int] = set()
  for row in rows:
    est = by_id.get(row.get("id"))
    if est is None:
      continue
    sanitized.append(_sanitize_row(row, est))
    seen_ids.add(row["id"])
  for est in establishments:
    if est["id"] not in seen_ids:
      sanitized.append(_sanitize_row({"id": est["id"], "primary": [], "secondary": []}, est))
  return sanitized


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
    self.temperature = 0.2
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

    # Output Schema — array of result objects
    self.output_schema = SCHEMA
  
  def _generate_raw(self, establishments: list, checker_feedback: str = "") -> list:
    """
    Generate attributes using Gemini API based on establishment information.
    
    Args:
      establishments: List of dictionaries, each containing id, name, and description.
      
    Returns:
      List of dictionaries, each containing id, primary, and secondary
    """
    # Create request text with the input structure
    feedback_block = ""
    if checker_feedback.strip():
      feedback_block = f"\nchecker_feedback (fix all issues):\n{checker_feedback.strip()}\n"
    request_text_str = f"""input: {json.dumps(establishments, indent=2)}
{feedback_block}output: """
    
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
      
      return json.loads(output_text_clean)
    except json.JSONDecodeError as e:
      raise ValueError(f"Failed to parse JSON response: {e}\nResponse text: {output_text}")

  def generate(self, establishments: list, *, checker_retries: int = 3) -> list:
    """Generate attributes, sanitize, and revise up to checker_retries times using checker feedback."""
    rows = _sanitize_attributes(establishments, self._generate_raw(establishments))
    if checker_retries <= 0:
      return rows

    try:
      from check_food_spend_add_attributes_optimizer import CheckFoodSpendAddAttributesOptimizer
    except ImportError:
      return rows

    checker = CheckFoodSpendAddAttributesOptimizer()
    for attempt in range(checker_retries):
      review = checker.generate_response(establishments, [], rows)
      if review.get("good_copy") and review.get("info_correct") and not (review.get("eval_text") or "").strip():
        return rows
      if attempt < checker_retries - 1:
        rows = _sanitize_attributes(
          establishments,
          self._generate_raw(establishments, checker_feedback=review.get("eval_text", "")),
        )
    return rows


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

  return optimizer.generate(test_case["establishments"])


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
