from google import genai
from google.genai import types
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are a checker verifying the output of a model (FoodSpendAddAttributesOptimizer) that transforms food establishment data into structured attributes.

## Input:
- **EVAL_INPUT**: A JSON array of food establishments. Each has `id`, `name`, and `description`.
- **PAST_REVIEW_OUTCOMES**: An array of past review outcomes.
- **REVIEW_NEEDED**: The JSON output from the optimizer that needs to be reviewed (array of result objects).

## Output:
Return valid JSON only. Put each top-level key on its own line (line break after each of good_copy, info_correct, eval_text). Example format:
```
{"good_copy": true,
"info_correct": true,
"eval_text": ""}
```

- `good_copy`: True if REVIEW_NEEDED is a valid JSON array and each item has the required fields: `id`, `primary`, `secondary`. Every `id` in REVIEW_NEEDED must exist in EVAL_INPUT. There must be exactly one output item per establishment in EVAL_INPUT.
- `info_correct`: True if the `primary` and `secondary` attributes for each item in REVIEW_NEEDED are correct according to the rules.
- `eval_text`: Empty string when good_copy and info_correct are both True. Otherwise, explain why REVIEW_NEEDED is incorrect. Each line must start with "Establishment <id>: ". **Crucial: The explanation must be self-contained and descriptive (e.g., "contains forbidden generic term 'Food'" instead of "violates Rule 3").** One line per erroneous item (max 25 words per line). **NEVER reference rule numbers in your output.**

## Critical Rules for FoodSpendAddAttributes:
1. `primary`: Must be a list containing at least one of: "Fast food", "Restaurant", "Beverage", "Grocery", "Dessert".
2. `secondary`: Must be a list of 3-7 tags that define the establishment and set it apart from others.
3. **Definition Axis**: Tags are only acceptable if they contribute to describing the establishment (e.g., "Burger" is okay, "Item" is unacceptable as it is too vague).
4. **Independently Understandable**: Tags must be understandable on their own. (e.g., "Convenience Store" is okay, but "Convenience" alone is unacceptable as it requires context).
5. **Source Grounding**: Tags must be based solely on the provided `name` and `description`. Do not use external knowledge or infer details not present in the text.
6. **Multi-word Tags**: Two-word tags are acceptable if a single word cannot capture the essence (e.g., "Frozen Yogurt").
7. **Singular Nouns**: Use singular nouns for `secondary` tags (e.g., "Taco" not "Tacos").
8. **No Repetition**: `primary` categories must not be repeated in `secondary`.
9. **Minimum Tags**: Every establishment MUST have at least 3 `secondary` tags. If fewer are provided, it is a failure.
10. **No External Inference**: If the description says "sells spicy food", do not tag "Burger" unless "Burger" is mentioned in the name or description.
11. **Negative Constraints**: `secondary` tags MUST NOT contain: "food", "dish", "cuisine", "snack", "meal", "eatery", "appetizer", "entree", "heat", "place".
12. **Generic Terms**: Strictly exclude standalone generic terms like "Food", "Drink", "Fare", "Meal", "Dish", "Beverage", "Cuisine", "Restaurant", "Store", "Cafe", "Tea", "Boba".

## Verification Steps:
1. Check PAST_REVIEW_OUTCOMES for repeated mistakes.
2. Verify good_copy: structure, required fields, one-to-one mapping of IDs.
3. Verify info_correct: Check primary/secondary choices against the axes and rules above.
4. eval_text: Only when incorrect. Each line starts with "Establishment <id>: ". **NEVER reference rule numbers in eval_text. Use descriptive language only (e.g., "contains forbidden term 'Cuisine'" or "tag 'Burger' is not grounded in source text").**
"""

class CheckFoodSpendAddAttributesOptimizer:
  """Handles all Gemini API interactions for checking FoodSpendAddAttributesOptimizer outputs"""

  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set.")
    self.client = genai.Client(api_key=api_key)

    self.model_name = model_name
    self.top_k = 40
    self.top_p = 0.95
    self.temperature = 0.5
    self.thinking_budget = 0
    self.max_output_tokens = 4096

    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
    ]
    self.system_prompt = SYSTEM_PROMPT

  def generate_response(self, eval_input: list, past_review_outcomes: list, review_needed: list) -> dict:
    """Generate a response using Gemini API for checking optimizer outputs."""
    request_text_str = f"""<EVAL_INPUT>
{json.dumps(eval_input, indent=2)}
</EVAL_INPUT>

<PAST_REVIEW_OUTCOMES>
{json.dumps(past_review_outcomes, indent=2)}
</PAST_REVIEW_OUTCOMES>

<REVIEW_NEEDED>
{json.dumps(review_needed, indent=2)}
</REVIEW_NEEDED>

Output:"""

    request_text = types.Part.from_text(text=request_text_str)
    contents = [types.Content(role="user", parts=[request_text])]

    generate_content_config = types.GenerateContentConfig(
      top_k=self.top_k,
      top_p=self.top_p,
      temperature=self.temperature,
      max_output_tokens=self.max_output_tokens,
      safety_settings=self.safety_settings,
      system_instruction=[types.Part.from_text(text=self.system_prompt)],
      thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
      response_mime_type="application/json",
    )

    output_text = ""
    for chunk in self.client.models.generate_content_stream(
      model=self.model_name,
      contents=contents,
      config=generate_content_config,
    ):
      if chunk.text is not None:
        output_text += chunk.text

    try:
      return json.loads(output_text.strip())
    except json.JSONDecodeError:
      # Fallback for potential markdown or extra text
      json_start = output_text.find('{')
      json_end = output_text.rfind('}') + 1
      if json_start != -1 and json_end > json_start:
        return json.loads(output_text[json_start:json_end])
      raise ValueError(f"Failed to parse JSON: {output_text}")

def run_test_case(test_name: str, eval_input: list, review_needed: list, past_review_outcomes: list = None, checker: 'CheckFoodSpendAddAttributesOptimizer' = None):
  if past_review_outcomes is None:
    past_review_outcomes = []
  if checker is None:
    checker = CheckFoodSpendAddAttributesOptimizer()

  print(f"\n{'='*80}\nRunning test: {test_name}\n{'='*80}")
  try:
    result = checker.generate_response(eval_input, past_review_outcomes, review_needed)
    print("Result:")
    print(json.dumps(result, indent=2))
    return result
  except Exception as e:
    print(f"ERROR: {str(e)}")
    return None

def run_correct_response(checker: CheckFoodSpendAddAttributesOptimizer = None):
  eval_input = [
    {"id": 1, "name": "Snack Tiger Tea", "description": "A purchase of snacks and beverages from a cafe."}
  ]
  review_needed = [
    {"id": 1, "primary": ["Beverage"], "secondary": ["Boba", "Tea", "Cafe"]}
  ]
  return run_test_case("correct_response", eval_input, review_needed, [], checker)

def run_wrong_secondary_count(checker: CheckFoodSpendAddAttributesOptimizer = None):
  eval_input = [
    {"id": 2, "name": "San Froyo", "description": "Frozen yogurt shop."}
  ]
  review_needed = [
    {"id": 2, "primary": ["Dessert"], "secondary": ["Frozen yogurt"]} # Only 1 tag, needs 3-7
  ]
  return run_test_case("wrong_secondary_count", eval_input, review_needed, [], checker)

def run_forbidden_words(checker: CheckFoodSpendAddAttributesOptimizer = None):
  eval_input = [
    {"id": 3, "name": "Manila Bay Cuisine", "description": "sells Filipino dishes"}
  ]
  review_needed = [
    {"id": 3, "primary": ["Restaurant"], "secondary": ["Filipino Cuisine", "Rice dish", "Family style"]} # "Cuisine" and "dish" are forbidden
  ]
  return run_test_case("forbidden_words", eval_input, review_needed, [], checker)

def run_external_inference(checker: CheckFoodSpendAddAttributesOptimizer = None):
  eval_input = [
    {"id": 4, "name": "Burning Mouth", "description": "This establishment sells spicy food."}
  ]
  review_needed = [
    {"id": 4, "primary": ["Restaurant"], "secondary": ["Spicy", "Burger", "Chicken"]} # "Burger" and "Chicken" are not in description
  ]
  return run_test_case("external_inference", eval_input, review_needed, [], checker)

def main(batch: int = 0):
  checker = CheckFoodSpendAddAttributesOptimizer()
  if batch == 0:
    run_correct_response(checker)
    run_wrong_secondary_count(checker)
    run_forbidden_words(checker)
    run_external_inference(checker)
  elif batch == 1:
    run_correct_response(checker)
  elif batch == 2:
    run_wrong_secondary_count(checker)
  elif batch == 3:
    run_forbidden_words(checker)
  elif batch == 4:
    run_external_inference(checker)

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch', type=int, default=0)
  args = parser.parse_args()
  main(batch=args.batch)
