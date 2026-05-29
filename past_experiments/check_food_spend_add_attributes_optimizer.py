from __future__ import annotations

from google import genai
from google.genai import types
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

RUN_SETTINGS = {
  "json": True,
  "sanitize": True,
  "gen_config": {
    "top_p": 0.95,
    "top_k": 40,
    "temperature": 0.2,
    "max_output_tokens": 4096,
    "response_mime_type": "application/json",
    "thinking_budget": 0,
  },
  "model_name": "gemini-flash-lite-latest",
  "output_schema": {
    "type": 6,
    "items": None,
    "required": ["eval_text", "good_copy", "info_correct"],
    "properties": {
      "eval_text": {
        "type": 1,
        "description": (
          "Empty string when no issues. Otherwise one line per bad establishment id, starting "
          "with 'Establishment <id>: '. Max 25 words per line. State the concrete flaw in "
          "plain language. Never cite rule numbers."
        ),
      },
      "good_copy": {
        "type": 4,
        "description": (
          "True only when REVIEW_NEEDED is a JSON array with one object per EVAL_INPUT id, each "
          "having id, primary (list), and secondary (list). False only for structural gaps — "
          "not for tag quality."
        ),
      },
      "info_correct": {
        "type": 4,
        "description": (
          "True when food/non-food classification is correct, secondary count is 3-7, tags are "
          "grounded, and every secondary tag is self-explanatory (no rejected single-word stubs)."
        ),
      },
    },
  },
}

CHECKER_OUTPUT_SCHEMA = types.Schema(
  type=types.Type.OBJECT,
  properties={
    "eval_text": types.Schema(
      type=types.Type.STRING,
      description=RUN_SETTINGS["output_schema"]["properties"]["eval_text"]["description"],
    ),
    "good_copy": types.Schema(
      type=types.Type.BOOLEAN,
      description=RUN_SETTINGS["output_schema"]["properties"]["good_copy"]["description"],
    ),
    "info_correct": types.Schema(
      type=types.Type.BOOLEAN,
      description=RUN_SETTINGS["output_schema"]["properties"]["info_correct"]["description"],
    ),
  },
  required=["eval_text", "good_copy", "info_correct"],
)

SYSTEM_PROMPT = """You are a checker for structured merchant attributes. Tags are used to cluster similar food-related establishments; they need not be food words but must discriminate one merchant from unrelated food merchants.

## Inputs
- **EVAL_INPUT**: array of `{id, name, description}`
- **PAST_REVIEW_OUTCOMES**: prior checker results
- **REVIEW_NEEDED**: array of `{id, primary, secondary}` to validate

## Output
Return JSON only, fields in order: `eval_text`, `good_copy`, `info_correct`.

## Step 1 — Food-related classification
Default **food-related** for every row unless the purchase cannot connect to any of:
- **Groceries** — food sold for home use (markets, supermarkets, food-forward general merchandise)
- **Delivered food** — prepared or grocery food brought to the customer
- **Dining out** — prepared food or drinks consumed away from home, including inflight or venue food service

If **not** food-related: correct output is `"primary": []`, `"secondary": []`. Fail if REVIEW_NEEDED assigns food categories to a non-food purchase, or leaves food merchants blank without cause.

Discount or variety retailers whose description emphasizes non-food merchandise (apparel, toys, general household goods) with no food angle are usually **not** food-related.

## Step 2 — Structure (`good_copy`)
- REVIEW_NEEDED is an array with exactly one object per EVAL_INPUT `id`
- Each object has `id`, `primary` (list), `secondary` (list)
- Fail `good_copy` only for missing/extra ids or wrong field types — not for tag semantics

## Step 3 — Food-related tag rules (`info_correct`)

**Denylist reminder:** the denylist applies only when the full tag has **no space**. Any tag with a space skips denylist checks entirely.

### Primary
- Non-empty list; **multiple primaries allowed** when the merchant spans more than one allowed category
- Allowed values only: "Fast food", "Restaurant", "Beverage", "Grocery", "Dessert"
- **Never** fail because primary has more than one entry
- When primary is **Beverage**, tags naming a specific drink or snack format are grounded if `description` mentions beverages, drinks, cafe, or snacks — including when the tag names a beverage type consistent with those words

### Secondary count and uniqueness
- **3 to 7** tags inclusive; three tags is sufficient — do not require more
- Tags unique within the row
- **Primary repeat rule:** fail only if a secondary tag equals a primary tag with **identical spelling** (same characters). Never fail because a secondary tag is plural, synonymous, or shares a root with a primary label

### Grounding (name + description together)
- Use **both** `name` and `description`. Merchant-name tokens count. Words or phrases from the description count.
- **Description phrases**: multi-word tags that quote or closely paraphrase phrasing from `description` are grounded.
- **Specialty inference**: a tag is grounded when reasonably linked to what the name or description establishes — not for unrelated dishes, cuisines, or flavors. A product or menu-style tag implied by the merchant **name** passes when `description` supports the same retail context and does not contradict it.
- **Flavor descriptors**: grounded when the source describes that taste profile.
- **Format/channel/venue tags**: grounded when stated in the source, or as a multi-word compound naming the food-related role. Do not assume takeout or delivery without support.
- **Mixed purchases** on food-related merchants: tags naming non-food departments or product lines are grounded when `description` includes them as part of the same purchase.
- Reject tags with no reasonable link to the source.

### Self-explanatory (clustering test)
*Would this tag alone cluster this merchant with similar food merchants?*

**Denylist gate (apply first per tag):** if the tag contains a space character, **stop** — do not run denylist logic on that tag; do not split the tag into words; never combine two denylist words into a failure. Judge only grounding and whether the full phrase is specific enough. Tags paraphrasing `description` pass.

For tags with **no** space: fail only when the **entire tag string** exactly equals one denylist token (case-sensitive). Words not on the list are never denylist failures. Never treat a substring or a word inside a multi-word tag as a denylist hit.

**Multi-word tags** copied or closely paraphrased from `description` pass the clustering test when they name a purchase type, department, venue, or channel stated in the source — including non-food words that describe where or how the food purchase occurred.

**Single-word denylist** — fail only when the entire tag equals one of:
`Market`, `Street`, `Retail`, `Service`, `Platform`, `Chain`, `Delivery`, `Food`, `Dish`, `Item`, `Product`, `Place`, `Household`, `Store`, `Shop`, `Convenience`
Exception: allow that exact token when it appears in `name` or `description` with clear food-retail meaning.

**Other single-word tags** — pass when grounded as cuisine, menu item, product department, beverage, flavor, venue/format, or name token supported by the source (including reasonable specialty inference from the merchant name).

### Strictness balance
- Fail every unsupported or non-self-explanatory tag on a row, even if other tags are strong
- Do not fail for alternate valid wording, count of exactly three, or multiple primaries
- Do not fail subjective quality opinions when all tags pass grounding and the self-explanatory test

## Field semantics
- `eval_text`: empty if no issues; else one line per bad `id`, prefix `Establishment <id>: `, max 25 words, no rule numbers
- `good_copy`: structural validity only
- `info_correct`: false if any row fails classification or tag rules above

## Procedure
1. Read PAST_REVIEW_OUTCOMES for recurring failure patterns
2. Per id: classify food-related vs not; check structure; for each secondary tag, check grounding, then self-explanatory (denylist only if the tag has no spaces)
3. Write `eval_text` listing every distinct issue
4. Set `good_copy`, then `info_correct`
"""

class CheckFoodSpendAddAttributesOptimizer:
  """Handles all Gemini API interactions for checking FoodSpendAddAttributesOptimizer outputs"""

  def __init__(
    self,
    model_name: str | None = None,
    thinking_budget: int | None = None,
    json_output: bool | None = None,
    sanitize: bool | None = None,
  ):
    """Initialize the Gemini agent with API configuration"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set.")
    self.client = genai.Client(api_key=api_key)

    gc = RUN_SETTINGS["gen_config"]
    self.model_name = model_name if model_name is not None else RUN_SETTINGS["model_name"]
    self.thinking_budget = (
      thinking_budget if thinking_budget is not None else gc["thinking_budget"]
    )
    self.top_k = gc["top_k"]
    self.top_p = gc["top_p"]
    self.temperature = gc["temperature"]
    self.max_output_tokens = gc["max_output_tokens"]
    self.json_output = json_output if json_output is not None else RUN_SETTINGS["json"]
    self.sanitize = sanitize if sanitize is not None else RUN_SETTINGS["sanitize"]
    self.response_mime_type = gc.get("response_mime_type", "application/json")
    self.output_schema = CHECKER_OUTPUT_SCHEMA

    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
    ]
    self.system_prompt = SYSTEM_PROMPT

  def _parse_checker_json(self, output_text: str) -> dict:
    text = output_text.strip()
    if self.sanitize:
      if text.startswith("```json"):
        text = text[7:].strip()
        if text.endswith("```"):
          text = text[:-3].strip()
      elif text.startswith("```"):
        text = text[3:].strip()
        if text.endswith("```"):
          text = text[:-3].strip()
    else:
      json_start = text.find("{")
      json_end = text.rfind("}") + 1
      if json_start != -1 and json_end > json_start:
        text = text[json_start:json_end]

    parsed = json.loads(text.strip())
    if not isinstance(parsed, dict):
      raise ValueError(f"Expected dict, got {type(parsed)}")
    if "eval_text" not in parsed:
      parsed["eval_text"] = ""
    return parsed

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

    config_kwargs = dict(
      top_k=self.top_k,
      top_p=self.top_p,
      temperature=self.temperature,
      max_output_tokens=self.max_output_tokens,
      safety_settings=self.safety_settings,
      system_instruction=[types.Part.from_text(text=self.system_prompt)],
      thinking_config=types.ThinkingConfig(
        thinking_budget=self.thinking_budget,
        include_thoughts=True,
      ),
    )
    if self.json_output:
      config_kwargs["response_mime_type"] = self.response_mime_type
      config_kwargs["response_schema"] = self.output_schema
    generate_content_config = types.GenerateContentConfig(**config_kwargs)

    output_text = ""
    thought_summary = ""

    for chunk in self.client.models.generate_content_stream(
      model=self.model_name,
      contents=contents,
      config=generate_content_config,
    ):
      if chunk.text is not None:
        output_text += chunk.text

      if hasattr(chunk, 'candidates') and chunk.candidates:
        for candidate in chunk.candidates:
          if hasattr(candidate, 'content') and candidate.content:
            if hasattr(candidate.content, 'parts') and candidate.content.parts:
              for part in candidate.content.parts:
                if hasattr(part, 'thought') and part.thought:
                  if hasattr(part, 'text') and part.text:
                    if thought_summary:
                      thought_summary += part.text
                    else:
                      thought_summary = part.text

    if not output_text or not output_text.strip():
      raise ValueError("Empty response from model. Check API key and model availability.")

    if thought_summary:
      print(f"{'='*80}")
      print("THOUGHT SUMMARY:")
      print(thought_summary.strip())
      print("="*80)

    try:
      return self._parse_checker_json(output_text)
    except json.JSONDecodeError as e:
      raise ValueError(f"Failed to parse JSON: {e}\nResponse: {output_text}") from e


TEST_CASES = [
  # --- Batch 1: correct passes across establishment types ---
  {
    "name": "beverage_boba_cafe",
    "batch": 1,
    "eval_input": [
      {"id": 101, "name": "Snack Tiger Tea", "description": "Snacks and beverages purchased at a cafe."}
    ],
    "review_needed": [
      {"id": 101, "primary": ["Beverage"], "secondary": ["Boba", "Tea", "Cafe"]}
    ],
    "past_review_outcomes": [],
    "ideal_output": {"eval_text": "", "good_copy": True, "info_correct": True},
  },
  {
    "name": "grocery_supermarket",
    "batch": 1,
    "eval_input": [
      {
        "id": 102,
        "name": "Trader Joe's",
        "description": "Groceries including organic produce, private-label items, and prepared foods.",
      }
    ],
    "review_needed": [
      {
        "id": 102,
        "primary": ["Grocery"],
        "secondary": ["Organic produce", "Private-label", "Prepared foods"],
      }
    ],
    "past_review_outcomes": [],
    "ideal_output": {"eval_text": "", "good_copy": True, "info_correct": True},
  },
  {
    "name": "fast_food_mexican",
    "batch": 1,
    "eval_input": [
      {
        "id": 103,
        "name": "Chipotle",
        "description": "Fast-casual Mexican restaurant serving burritos, bowls, and tacos.",
      }
    ],
    "review_needed": [
      {"id": 103, "primary": ["Fast food"], "secondary": ["Mexican", "Burrito", "Bowl", "Taco"]}
    ],
    "past_review_outcomes": [],
    "ideal_output": {"eval_text": "", "good_copy": True, "info_correct": True},
  },
  {
    "name": "restaurant_sushi_bar",
    "batch": 1,
    "eval_input": [
      {
        "id": 104,
        "name": "Nobu",
        "description": "Japanese restaurant serving sushi, sashimi, and Japanese-Peruvian fusion dishes.",
      }
    ],
    "review_needed": [
      {
        "id": 104,
        "primary": ["Restaurant"],
        "secondary": ["Japanese", "Sushi", "Sashimi", "Peruvian fusion"],
      }
    ],
    "past_review_outcomes": [],
    "ideal_output": {"eval_text": "", "good_copy": True, "info_correct": True},
  },
  # --- Batch 2: semantic / tag-quality failures ---
  {
    "name": "dessert_too_few_secondary",
    "batch": 2,
    "eval_input": [
      {"id": 201, "name": "San Froyo", "description": "Frozen yogurt shop selling frozen yogurt and toppings."}
    ],
    "review_needed": [
      {"id": 201, "primary": ["Dessert"], "secondary": ["Frozen yogurt"]}
    ],
    "past_review_outcomes": [],
    "ideal_output": {
      "eval_text": "Establishment 201: fewer than 3 secondary tags provided.",
      "good_copy": True,
      "info_correct": False,
    },
  },
  {
    "name": "primary_repeated_in_secondary",
    "batch": 2,
    "eval_input": [
      {
        "id": 202,
        "name": "Shake Shack",
        "description": "Fast-casual chain serving burgers, hot dogs, fries, and shakes.",
      }
    ],
    "review_needed": [
      {
        "id": 202,
        "primary": ["Fast food"],
        "secondary": ["Fast food", "Burger", "Milkshake"],
      }
    ],
    "past_review_outcomes": [],
    "ideal_output": {
      "eval_text": "Establishment 202: secondary repeats primary value 'Fast food'.",
      "good_copy": True,
      "info_correct": False,
    },
  },
  {
    "name": "vague_food_and_dish_tags",
    "batch": 2,
    "eval_input": [
      {"id": 203, "name": "Manila Bay Kitchen", "description": "Family-style Filipino restaurant with rice plates."}
    ],
    "review_needed": [
      {"id": 203, "primary": ["Restaurant"], "secondary": ["Food", "Dish", "Family style"]}
    ],
    "past_review_outcomes": [],
    "ideal_output": {
      "eval_text": "Establishment 203: secondary tags 'Food' and 'Dish' are too vague.",
      "good_copy": True,
      "info_correct": False,
    },
  },
  {
    "name": "vague_household_tag",
    "batch": 2,
    "eval_input": [
      {
        "id": 205,
        "name": "Target",
        "description": "General store purchase including groceries and household essentials.",
      }
    ],
    "review_needed": [
      {
        "id": 205,
        "primary": ["Grocery"],
        "secondary": ["General store", "Groceries", "Household"],
      }
    ],
    "past_review_outcomes": [],
    "ideal_output": {
      "eval_text": "Establishment 205: tag 'Household' is not self-explanatory; use a specific compound such as 'Household essentials'.",
      "good_copy": True,
      "info_correct": False,
    },
  },
  {
    "name": "household_essentials_ok",
    "batch": 2,
    "eval_input": [
      {
        "id": 206,
        "name": "Target",
        "description": "General store purchase including groceries and household essentials.",
      }
    ],
    "review_needed": [
      {
        "id": 206,
        "primary": ["Grocery"],
        "secondary": ["General store", "Groceries", "Household essentials"],
      }
    ],
    "past_review_outcomes": [],
    "ideal_output": {"eval_text": "", "good_copy": True, "info_correct": True},
  },
  {
    "name": "meal_kit_delivery_correct",
    "batch": 2,
    "eval_input": [
      {
        "id": 204,
        "name": "Blue Apron",
        "description": "Subscription boxes of ingredients and recipes delivered for home cooking.",
      }
    ],
    "review_needed": [
      {
        "id": 204,
        "primary": ["Grocery"],
        "secondary": ["Meal kit", "Home cooking", "Subscription delivery"],
      }
    ],
    "past_review_outcomes": [],
    "ideal_output": {"eval_text": "", "good_copy": True, "info_correct": True},
  },
  # --- Batch 3: non-food vs food-adjacent ---
  {
    "name": "non_food_fuel_blank",
    "batch": 3,
    "eval_input": [
      {"id": 301, "name": "Shell Gas Station", "description": "Fuel purchase at a gas station."}
    ],
    "review_needed": [
      {"id": 301, "primary": [], "secondary": []}
    ],
    "past_review_outcomes": [],
    "ideal_output": {"eval_text": "", "good_copy": True, "info_correct": True},
  },
  {
    "name": "non_food_utility_tagged_as_grocery",
    "batch": 3,
    "eval_input": [
      {"id": 302, "name": "PG&E", "description": "Monthly electric utility bill payment."}
    ],
    "review_needed": [
      {"id": 302, "primary": ["Grocery"], "secondary": ["Utility", "Electric", "Bill"]}
    ],
    "past_review_outcomes": [],
    "ideal_output": {
      "eval_text": "Establishment 302: non-food transaction should have empty primary and secondary.",
      "good_copy": True,
      "info_correct": False,
    },
  },
  {
    "name": "non_food_insurance_blank",
    "batch": 3,
    "eval_input": [
      {"id": 303, "name": "State Farm", "description": "Auto insurance premium payment."}
    ],
    "review_needed": [
      {"id": 303, "primary": [], "secondary": []}
    ],
    "past_review_outcomes": [],
    "ideal_output": {"eval_text": "", "good_copy": True, "info_correct": True},
  },
  {
    "name": "airline_inflight_food",
    "batch": 3,
    "eval_input": [
      {
        "id": 305,
        "name": "United Airlines",
        "description": "Inflight meal and beverage purchase on a domestic flight.",
      }
    ],
    "review_needed": [
      {
        "id": 305,
        "primary": ["Restaurant"],
        "secondary": ["Inflight meal", "Airline catering", "Domestic flight"],
      }
    ],
    "past_review_outcomes": [],
    "ideal_output": {"eval_text": "", "good_copy": True, "info_correct": True},
  },
  {
    "name": "convenience_store_with_food",
    "batch": 3,
    "eval_input": [
      {
        "id": 304,
        "name": "7-Eleven",
        "description": "Convenience store purchase of snacks, drinks, and hot food.",
      }
    ],
    "review_needed": [
      {
        "id": 304,
        "primary": ["Grocery"],
        "secondary": ["Convenience Store", "Snacks", "Fountain drinks"],
      }
    ],
    "past_review_outcomes": [],
    "ideal_output": {"eval_text": "", "good_copy": True, "info_correct": True},
  },
  # --- Batch 4: inference and structural edge cases ---
  {
    "name": "supported_spicy_inference",
    "batch": 4,
    "eval_input": [
      {"id": 401, "name": "Burning Mouth", "description": "Restaurant known for spicy food."}
    ],
    "review_needed": [
      {"id": 401, "primary": ["Restaurant"], "secondary": ["Spicy", "Bold flavor", "Heat-forward"]}
    ],
    "past_review_outcomes": [],
    "ideal_output": {"eval_text": "", "good_copy": True, "info_correct": True},
  },
  {
    "name": "ungrounded_burger_and_chicken",
    "batch": 4,
    "eval_input": [
      {"id": 402, "name": "Burning Mouth", "description": "Restaurant known for spicy food."}
    ],
    "review_needed": [
      {"id": 402, "primary": ["Restaurant"], "secondary": ["Spicy", "Burger", "Chicken"]}
    ],
    "past_review_outcomes": [],
    "ideal_output": {
      "eval_text": "Establishment 402: tags 'Burger' and 'Chicken' are not grounded in name or description.",
      "good_copy": True,
      "info_correct": False,
    },
  },
  {
    "name": "warehouse_grocery_mixed_retailer",
    "batch": 4,
    "eval_input": [
      {
        "id": 403,
        "name": "Costco",
        "description": "Bulk groceries, electronics, and household goods with membership.",
      }
    ],
    "review_needed": [
      {
        "id": 403,
        "primary": ["Grocery"],
        "secondary": ["Warehouse club", "Bulk groceries", "Membership retailer"],
      }
    ],
    "past_review_outcomes": [],
    "ideal_output": {"eval_text": "", "good_copy": True, "info_correct": True},
  },
  {
    "name": "missing_establishment_in_review",
    "batch": 4,
    "eval_input": [
      {"id": 501, "name": "Marugame Udon", "description": "Japanese noodle restaurant specializing in udon and tempura."},
      {"id": 502, "name": "Harry & David", "description": "Gift retailer shipping fruit baskets and gourmet gifts."},
    ],
    "review_needed": [
      {"id": 501, "primary": ["Restaurant"], "secondary": ["Udon", "Tempura", "Japanese noodle"]}
    ],
    "past_review_outcomes": [],
    "ideal_output": {
      "eval_text": "Establishment 502: missing from REVIEW_NEEDED; expected one result per EVAL_INPUT id.",
      "good_copy": False,
      "info_correct": False,
    },
  },
]


def _compare_checker_result(actual: dict | None, ideal_output: dict) -> tuple[bool, str]:
  if actual is None:
    return False, "no model output"

  actual_good_copy = bool(actual.get("good_copy"))
  actual_info_correct = bool(actual.get("info_correct"))
  ideal_good_copy = bool(ideal_output.get("good_copy"))
  ideal_info_correct = bool(ideal_output.get("info_correct"))

  if actual_good_copy != ideal_good_copy or actual_info_correct != ideal_info_correct:
    return (
      False,
      f"good_copy model={actual_good_copy} ideal={ideal_good_copy}; "
      f"info_correct model={actual_info_correct} ideal={ideal_info_correct}",
    )

  if not ideal_good_copy or not ideal_info_correct:
    actual_eval_text = (actual.get("eval_text") or "").strip()
    if not actual_eval_text:
      return False, "expected non-empty eval_text when a boolean is false"
  return True, "booleans match ideal output"


def run_test_case(
  test_name: str,
  eval_input: list,
  review_needed: list,
  past_review_outcomes: list | None = None,
  checker: CheckFoodSpendAddAttributesOptimizer | None = None,
  ideal_output: dict | None = None,
):
  if past_review_outcomes is None:
    past_review_outcomes = []
  if checker is None:
    checker = CheckFoodSpendAddAttributesOptimizer()

  print(f"\n{'='*80}\nRunning test: {test_name}\n{'='*80}")
  if ideal_output is not None:
    print("IDEAL OUTPUT:")
    print(json.dumps(ideal_output, indent=2))
  try:
    result = checker.generate_response(eval_input, past_review_outcomes, review_needed)
    print("CHECKER OUTPUT:")
    print(json.dumps(result, indent=2))
    if ideal_output is not None:
      ok, detail = _compare_checker_result(result, ideal_output)
      print("OUTPUT MATCHES IDEAL:")
      print("YES" if ok else "NO")
      print(detail)
    print("="*80)
    return result
  except Exception as e:
    print(f"ERROR: {str(e)}")
    return None


def run_test_from_case(test_case: dict, checker: CheckFoodSpendAddAttributesOptimizer | None = None):
  return run_test_case(
    test_case["name"],
    test_case["eval_input"],
    test_case["review_needed"],
    test_case.get("past_review_outcomes", []),
    checker,
    test_case.get("ideal_output"),
  )


def test_with_inputs(test_name_or_index_or_dict, checker: CheckFoodSpendAddAttributesOptimizer | None = None):
  if checker is None:
    checker = CheckFoodSpendAddAttributesOptimizer()

  test_case = None
  if isinstance(test_name_or_index_or_dict, dict):
    if "eval_input" in test_name_or_index_or_dict and "review_needed" in test_name_or_index_or_dict:
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

  return run_test_from_case(test_case, checker)


def run_batch(batch: int, checker: CheckFoodSpendAddAttributesOptimizer | None = None):
  """Run all test cases in a batch (1–4)."""
  if checker is None:
    checker = CheckFoodSpendAddAttributesOptimizer()
  cases = [tc for tc in TEST_CASES if tc.get("batch") == batch]
  if not cases:
    raise ValueError(f"No test cases for batch {batch}")
  results = []
  for tc in cases:
    results.append(run_test_from_case(tc, checker))
  return results


def main(batch: int = 0):
  checker = CheckFoodSpendAddAttributesOptimizer()
  if batch == 0:
    cases = TEST_CASES
  else:
    cases = [tc for tc in TEST_CASES if tc.get("batch") == batch]
    if not cases:
      raise ValueError("batch must be 0 (all) or 1–4")
  for tc in cases:
    run_test_from_case(tc, checker)

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch', type=int, default=0)
  args = parser.parse_args()
  main(batch=args.batch)
