from google import genai
from google.genai import types
import os
import json
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are a financial transaction categorization expert. Categorize each transaction using only the input fields and the subcategory meanings below.

## Input Format
JSON array of transaction groups. Each entry contains:
- `establishment_name`: Merchant/establishment name
- `establishment_description`: What the establishment provides
- `transactions`: Array with `transaction_id`, `transaction_text` (raw bank statement text), and `amount`
- `category_options`: Allowed categories for that group (`unknown` is always valid even when not listed). **If omitted**, treat every **leaf subcategory** listed under Parent Category and Subcategory Meanings below as available—never parent categories (`income`, `meals`, `leisure`, `bills`, `shelter`, `education`, `shopping`, `transportation`, `health`).

## Input-Only Evidence (strict)
Use **only** `establishment_name`, `establishment_description`, `transaction_text`, `amount`, and `category_options` (or the default full leaf list when omitted). Do not use outside knowledge, merchant research, or assumptions beyond what these fields state.

## Category Selection
- Output must be an exact copy of one allowed category string, or `unknown`.
- When `category_options` is present, it is the allowed set for that group. When absent, all leaf subcategories from this prompt are allowed (plus `unknown`).
- **Always closest fit for identifiable services/fees/specialty goods**: Purchases or payments where the input identifies a **specific service, fee type, or specialty retailer with one dominant product type** must map to the **closest matching** allowed subcategory—even when no category explicitly lists that good or service. Use `medium` or `low` confidence when the fit is weak.
- **`unknown` when purchase item is unclear**: Use `unknown` when it is genuinely unclear **what was purchased**—including multi-product or variety retailers where many unrelated item types are plausible and the input does not identify what was bought. Also use for peer-to-peer with no stated purpose, bare transfer mechanics without explicit same-person ownership, or purchase text with no merchant or product clue. Do **not** use `unknown` when a **service or fee type** is identifiable but no category is a perfect fit—force the closest allowed category instead.
- **Service/fee type is sufficient; product type is not**: When establishment fields or transaction text identify a **service provider or fee type**, use that for closest-fit categorization. Knowing only that a retailer sells many unrelated product types is **not** sufficient—without an identifiable item, use `unknown`.
- If `establishment_name` / `establishment_description` conflict with `transaction_text` or `amount`, follow evidence priority—do not let a wrong establishment label override clearer transaction text or amount sign.
- Outflows (positive amount) are never income.

## Inflows (negative amount)
Negative amounts are inflows. Refunds or reversals of a prior expense use the same expense subcategory when identifiable in the input. Income subcategories only for clear earnings stated in the input (salary, interest, business revenue).

## Parent Category and Subcategory Meanings
- **income**:
  - `income_salary`: regular paycheck income
  - `income_interest`: bank or stock interest earned
  - `income_sidegig`: freelance or secondary work income
  - `income_business`: business entity revenue
- **meals**:
  - `food_groceries`: food for home cooking
  - `food_dining_out`: restaurant meals
  - `food_delivered_food`: delivered food
- **leisure**:
  - `leisure_entertainment`: movies, concerts, events, recreation, alcoholic beverages
  - `leisure_travel_vacations`: trips and vacations
- **bills**:
  - `bills_connectivity`: internet, cable, phone
  - `bills_insurance`: health, auto, home insurance premiums
  - `bills_tax`: local, state, or federal taxes
  - `bills_service_fees`: bank or administrative fees, memberships, association dues
- **shelter**:
  - `shelter_home`: mortgage or rent
  - `shelter_utilities`: electricity, gas, water, trash
  - `shelter_upkeep`: home repairs and maintenance
- **education**:
  - `education_kids_activities`: children's classes or sports
  - `education_tuition`: personal or dependent education
- **shopping**:
  - `shopping_clothing`: apparel
  - `shopping_gadgets`: electronics and technology
  - `shopping_kids`: shopping for children
  - `shopping_pets`: pet supplies and services
- **transportation**:
  - `transport_public`: bus, train, subway
  - `transport_car_fuel`: gas, maintenance, parking, car payments
- **health**:
  - `health_medical_pharmacy`: doctor visits, prescriptions
  - `health_gym_wellness`: gym memberships, supplements
  - `health_personal_care`: haircuts, toiletries
- **donations_gifts**: charitable giving or gifts for others
- **transfer**: Movement between accounts owned by the **same person**, or payment toward **their own** debt, mortgage, or liability. Requires **explicit** same-person/own-account wording in the input—not merely transfer-related labels.

## Analysis Process
1. **Transaction text** (primary): Keywords and patterns in `transaction_text`
2. **Establishment**: `establishment_name` and `establishment_description` as stated in input
3. **Amount**: Sign (inflow/outflow) and size as supporting context
4. **Decision order**: (a) Resolve field conflicts using evidence priority. (b) **Transfer check first**: if the input lacks explicit same-person/own-account ownership, do **not** output `transfer`—use `unknown` even when transaction text or description says "transfer", "interbank transfer", "internet transfer", "from checking", "from savings", or shows masked account numbers. (c) Multi-product variety retailer with no identifiable purchased item → `unknown`. (d) Identifiable service, fee type, or specialty-goods purchase → closest matching allowed category. (e) Otherwise genuinely unidentified → `unknown`.
5. **Reasoning**: Brief positive evidence from input fields only—never discuss alternatives or `category_options`

## Confidence Levels
- **high**: Strong, aligned input evidence
- **medium**: Good input evidence with some ambiguity
- **low**: Weak or conflicting input evidence

## Critical Rules
- **Subcategory only**: Output a leaf subcategory—never a parent (`income`, `meals`, `leisure`, `bills`, `shelter`, `education`, `shopping`, `transportation`, `health`).
- **unknown (unclear purchase or unidentified purpose)**: Use when **what was purchased** is unclear (including variety/multi-product retailers with no identifiable item), or for P2P/transfers per the rules above. Do **not** use `unknown` when a **service or fee type** is identifiable but no category is a perfect fit—force the closest allowed category instead.
- **transfer (strict)**: The `transfer` **category** applies only when the input **explicitly** states same-person ownership or payment to the user's **own** card, loan, mortgage, or liability. The word "transfer" in merchant name, establishment description, or transaction text does **not** by itself justify the `transfer` category. Interbank, internet, internal, checking, savings, or masked-account movement **without** explicit own-account/same-person language → `unknown`, never `transfer`. Never infer same-person ownership from transfer mechanics, account type, or inflow/outflow sign alone.
- **P2P**: Payment to another person with no stated purpose in the input → `unknown`. Never `donations_gifts` or `transfer` without explicit input evidence. P2P with stated purpose in the input → matching expense subcategory from the allowed set.
- **General / multi-product retailers**: Variety stores, dollar stores, and mixed-product retailers where many unrelated item types are plausible and the input does not identify what was bought → `unknown`. Specialty retailers with one dominant product type → closest subcategory.
- **Evidence priority**: `transaction_text` > `establishment_name` / `establishment_description` > `amount`. When fields conflict, trust the higher-priority field.
- **Output**: Exact input `transaction_id`; only `transaction_id`, `reasoning`, `category`, `confidence` per transaction.
"""

OUTPUT_SCHEMA = types.Schema(
  type=types.Type.ARRAY,
  items=types.Schema(
    type=types.Type.OBJECT,
    properties={
      "transaction_id": types.Schema(
        type=types.Type.INTEGER,
        description="The transaction ID from the input transaction",
      ),
      "reasoning": types.Schema(
        type=types.Type.STRING,
        description="Brief 1-2 sentence explanation of why this category was chosen. Focus on the key decisive factors only.",
      ),
      "category": types.Schema(
        type=types.Type.STRING,
        description="One of the category_options in the input, or any leaf subcategory from the prompt when category_options is omitted; or 'unknown' when the purchased item or payment purpose is genuinely unclear",
      ),
      "confidence": types.Schema(
        type=types.Type.STRING,
        enum=["high", "medium", "low"],
        description="Confidence level in the categorization: 'high' for strong evidence, 'medium' for good evidence with some ambiguity, 'low' for weak/conflicting evidence",
      ),
    },
    required=["transaction_id", "reasoning", "category", "confidence"],
  ),
)

class RethinkTransactionCategorization:
  """Handles all Gemini API interactions for rethinking transaction categorization"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for transaction categorization"""
    # API Configuration
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file or environment.")
    self.client = genai.Client(api_key=api_key)
    
    # Model Configuration
    self.thinking_budget = 0
    self.model_name = model_name
    
    # Generation Configuration Constants
    self.top_k = 40
    self.temperature = 0.5
    self.top_p = 0.95
    self.max_output_tokens = 2048
    
    # Safety Settings
    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
    ]
    
    # System Prompt
    self.system_prompt = SYSTEM_PROMPT

  
  def generate_response(self, input_json: str) -> list:
    """
    Generate categorization response using Gemini API.
    
    Args:
      input_json: JSON string containing an array of transaction groups.
      
    Returns:
      List of dictionaries containing categorized transactions with reasoning
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
      top_k=self.top_k,
      temperature=self.temperature,
      top_p=self.top_p,
      max_output_tokens=self.max_output_tokens,
      safety_settings=self.safety_settings,
      system_instruction=[types.Part.from_text(text=self.system_prompt)],
      thinking_config=types.ThinkingConfig(
        thinking_budget=self.thinking_budget,
        include_thoughts=True,
      ),
      response_mime_type="application/json",
      response_schema=OUTPUT_SCHEMA,
    )

    # Generate response using streaming to extract thoughts
    output_text = ""
    thought_summary = ""

    for chunk in self.client.models.generate_content_stream(
      model=self.model_name,
      contents=contents,
      config=generate_content_config,
    ):
      if chunk.text is not None:
        output_text += chunk.text

      if hasattr(chunk, "candidates") and chunk.candidates:
        for candidate in chunk.candidates:
          if hasattr(candidate, "content") and candidate.content:
            if hasattr(candidate.content, "parts") and candidate.content.parts:
              for part in candidate.content.parts:
                if hasattr(part, "thought") and part.thought:
                  if hasattr(part, "text") and part.text:
                    if thought_summary:
                      thought_summary += part.text
                    else:
                      thought_summary = part.text

    if thought_summary:
      print(f"{'='*80}")
      print("THOUGHT SUMMARY:")
      print(thought_summary.strip())
      print("="*80)

    # Check if response is empty
    if not output_text or not output_text.strip():
      raise ValueError(f"Empty response from model. Check API key and model availability.")
    
    print(f"{'='*80}")
    print("RESPONSE OUTPUT:")
    print(output_text.strip())
    print("="*80)

    # Parse JSON response
    try:
      # Extract JSON from the response (in case there's extra text)
      output_text = output_text.strip()
      # Try to find JSON array in the response
      if output_text.startswith("```"):
        # Remove markdown code blocks if present
        lines = output_text.split("\n")
        json_lines = []
        in_code_block = False
        for line in lines:
          if line.strip().startswith("```"):
            in_code_block = not in_code_block
            continue
          if in_code_block or (not in_code_block and line.strip()):
            json_lines.append(line)
        output_text = "\n".join(json_lines)
      
      return json.loads(output_text)
    except json.JSONDecodeError as e:
      # Fallback: extract the outermost JSON array if parsing fails.
      bracket_start = output_text.find("[")
      bracket_end = output_text.rfind("]")
      if bracket_start != -1 and bracket_end > bracket_start:
        try:
          return json.loads(output_text[bracket_start : bracket_end + 1])
        except json.JSONDecodeError:
          pass
      raise ValueError(f"Failed to parse JSON response: {e}\nResponse was: {output_text}")


def test_with_inputs(input_json: list, categorizer: RethinkTransactionCategorization = None):
  """
  Convenient method to test the categorizer with custom inputs.
  
  Args:
    input_json: List of dictionaries containing transaction groups.
    categorizer: Optional RethinkTransactionCategorization instance. If None, creates a new one.
    
  Returns:
    List of dictionaries containing categorized transactions with reasoning
  """
  if categorizer is None:
    categorizer = RethinkTransactionCategorization()
  
  return categorizer.generate_response(json.dumps(input_json, indent=2))


_MAX_CASES_PER_OPTIMIZER_BATCH = 1

TEST_CASES = [
  {
    "batch": 1,
    "name": "batch_1_sidegig_shell_text_mismatch_rent_nomatch",
    "input": """[
  {
    "establishment_name": "Upwork",
    "establishment_description": "Freelance marketplace paying contractors for completed project work.",
    "transactions": [
      {
        "transaction_id": 9011001,
        "transaction_text": "Upwork Escrow Release",
        "amount": -875.00
      }
    ],
    "category_options": [
      "income_sidegig",
      "income_salary",
      "income_business",
      "income_interest",
      "transfer"
    ]
  },
  {
    "establishment_name": "Whole Foods Market",
    "establishment_description": "Organic grocery supermarket selling food for home use.",
    "transactions": [
      {
        "transaction_id": 9011002,
        "transaction_text": "Shell Oil 44221 Houston Tx",
        "amount": 41.20
      }
    ],
    "category_options": [
      "transport_car_fuel",
      "food_groceries",
      "food_dining_out",
      "bills_service_fees",
      "shopping_gadgets"
    ]
  },
  {
    "establishment_name": "Greystar Property Mgmt",
    "establishment_description": "Property manager collecting monthly apartment rent from tenants.",
    "transactions": [
      {
        "transaction_id": 9011003,
        "transaction_text": "Greystar Rent Pmt Apt 4B",
        "amount": 1850.00
      }
    ],
    "category_options": [
      "bills_insurance",
      "bills_service_fees",
      "bills_tax",
      "income_business",
      "leisure_entertainment"
    ]
  }
]""",
    "output": [
      {
        "transaction_id": 9011001,
        "reasoning": "The inflow is a freelance contractor payment from a project marketplace.",
        "category": "income_sidegig",
        "confidence": "high",
      },
      {
        "transaction_id": 9011002,
        "reasoning": "The transaction text shows a gas-station fuel purchase, which overrides the grocery establishment label.",
        "category": "transport_car_fuel",
        "confidence": "high",
      },
      {
        "transaction_id": 9011003,
        "reasoning": "The payment is clearly monthly apartment rent, but none of the listed categories cover housing rent. Closest option is business income, possibly for an office space.",
        "category": "income_business",
        "confidence": "low",
      },
    ],
  },
  {
    "batch": 2,
    "name": "batch_2_own_card_venmo_text_mismatch_interbank",
    "input": """[
  {
    "establishment_name": "Chase Credit Card",
    "establishment_description": "Payment to the user's own credit card account.",
    "transactions": [
      {
        "transaction_id": 9012001,
        "transaction_text": "Online Payment To Chase Card 9912",
        "amount": 425.00
      }
    ],
    "category_options": [
      "transfer",
      "bills_service_fees",
      "shelter_home",
      "income_interest",
      "donations_gifts"
    ]
  },
  {
    "establishment_name": "Venmo Payment To Sarah: Food",
    "establishment_description": "Payment to a friend for food.",
    "transactions": [
      {
        "transaction_id": 9012002,
        "transaction_text": "Venmo Payment To Sarah",
        "amount": 35.00
      }
    ],
    "category_options": [
      "leisure_entertainment",
      "donations_gifts",
      "food_dining_out",
      "transfer",
      "bills_service_fees"
    ]
  },
  {
    "establishment_name": "Internet Transfer from CK ***2974",
    "establishment_description": "An interbank transfer from a checking account.",
    "transactions": [
      {
        "transaction_id": 9012003,
        "transaction_text": "Internet Transfer from Xx2974 CK -",
        "amount": -10.00
      }
    ],
    "category_options": [
      "transfer",
      "income_interest",
      "income_salary",
      "bills_service_fees",
      "donations_gifts"
    ]
  }
]""",
    "output": [
      {
        "transaction_id": 9012001,
        "reasoning": "The establishment description states this is a payment to the user's own credit card account.",
        "category": "transfer",
        "confidence": "high",
      },
      {
        "transaction_id": 9012002,
        "reasoning": "The transaction text is a P2P payment to another person with no stated purpose, not for food.",
        "category": "unknown",
        "confidence": "high",
      },
      {
        "transaction_id": 9012003,
        "reasoning": "The input describes an interbank transfer without explicit same-person ownership or a clear expense purpose.",
        "category": "unknown",
        "confidence": "high",
      },
    ],
  },
  {
    "batch": 3,
    "name": "batch_3_chevron_text_mismatch_gym_vet_nomatch",
    "input": """[
  {
    "establishment_name": "Geico",
    "establishment_description": "Auto insurance company collecting policy premium payments.",
    "transactions": [
      {
        "transaction_id": 9013001,
        "transaction_text": "Chevron 204418 Denver Co",
        "amount": 48.90
      }
    ],
    "category_options": [
      "transport_car_fuel",
      "bills_insurance",
      "bills_service_fees",
      "food_groceries",
      "shopping_gadgets"
    ]
  },
  {
    "establishment_name": "Planet Fitness",
    "establishment_description": "Gym chain offering monthly fitness memberships.",
    "transactions": [
      {
        "transaction_id": 9013002,
        "transaction_text": "Planet Fitness Club Fee",
        "amount": 24.99
      }
    ],
    "category_options": [
      "health_gym_wellness",
      "health_personal_care",
      "leisure_entertainment",
      "bills_service_fees",
      "shopping_clothing"
    ]
  },
  {
    "establishment_name": "BluePearl Pet Hospital",
    "establishment_description": "Veterinary emergency clinic providing animal medical treatment.",
    "transactions": [
      {
        "transaction_id": 9013003,
        "transaction_text": "BluePearl Vet Emergency",
        "amount": 312.00
      }
    ],
    "category_options": [
      "bills_service_fees",
      "donations_gifts",
      "leisure_entertainment",
      "food_groceries",
      "bills_insurance"
    ]
  }
]""",
    "output": [
      {
        "transaction_id": 9013001,
        "reasoning": "The transaction text shows a Chevron gas-station charge, not an insurance premium.",
        "category": "transport_car_fuel",
        "confidence": "high",
      },
      {
        "transaction_id": 9013002,
        "reasoning": "The recurring charge is a monthly gym membership at a fitness facility.",
        "category": "health_gym_wellness",
        "confidence": "high",
      },
      {
        "transaction_id": 9013003,
        "reasoning": "The payment is clearly for veterinary medical treatment, but no medical or pharmacy category is listed. Closest is service fees, possibly for the vet's professional fees.",
        "category": "bills_service_fees",
        "confidence": "high",
      },
    ],
  },
  {
    "batch": 4,
    "name": "batch_4_fee_amount_mismatch_tutoring_netflix_text",
    "input": """[
  {
    "establishment_name": "Ally Bank Savings",
    "establishment_description": "Interest earned on the user's savings account balance.",
    "transactions": [
      {
        "transaction_id": 9014001,
        "transaction_text": "Monthly Maintenance Fee",
        "amount": 12.00
      }
    ],
    "category_options": [
      "bills_service_fees",
      "income_interest",
      "income_salary",
      "transfer",
      "donations_gifts"
    ]
  },
  {
    "establishment_name": "Sylvan Learning Center",
    "establishment_description": "After-school tutoring and test-prep programs for school-age children.",
    "transactions": [
      {
        "transaction_id": 9014002,
        "transaction_text": "Sylvan Learning Monthly",
        "amount": 220.00
      }
    ],
    "category_options": [
      "leisure_entertainment",
      "food_dining_out",
      "bills_connectivity",
      "transport_public",
      "bills_service_fees"
    ]
  },
  {
    "establishment_name": "Best Buy",
    "establishment_description": "Consumer electronics and appliance retail store.",
    "transactions": [
      {
        "transaction_id": 9014003,
        "transaction_text": "Netflix.Com Bill Pay",
        "amount": 15.99
      }
    ],
    "category_options": [
      "leisure_entertainment",
      "shopping_gadgets",
      "bills_connectivity",
      "food_dining_out",
      "bills_service_fees"
    ]
  }
]""",
    "output": [
      {
        "transaction_id": 9014001,
        "reasoning": "The outflow and transaction text show a bank maintenance fee, not interest income despite the establishment description.",
        "category": "bills_service_fees",
        "confidence": "high",
      },
      {
        "transaction_id": 9014002,
        "reasoning": "The charge is clearly for children's tutoring, but no education category appears in the options. Closest option is service fees, possibly for the teachers' professional fees.",
        "category": "bills_service_fees",
        "confidence": "low",
      },
      {
        "transaction_id": 9014003,
        "reasoning": "The transaction text shows a Netflix streaming bill, not an electronics retail purchase.",
        "category": "leisure_entertainment",
        "confidence": "high",
      },
    ],
  },
]

_optimizer_batch_sizes = {}
for _tc in TEST_CASES:
  _bn = int(_tc.get("batch", 1))
  _optimizer_batch_sizes[_bn] = _optimizer_batch_sizes.get(_bn, 0) + 1
assert sorted(_optimizer_batch_sizes.keys()) == [1, 2, 3, 4], _optimizer_batch_sizes
assert all(_count == _MAX_CASES_PER_OPTIMIZER_BATCH for _count in _optimizer_batch_sizes.values()), _optimizer_batch_sizes


def _ideal_outcome(tc: dict):
  ideal = tc.get("output")
  if ideal is None:
    return None
  if isinstance(ideal, str):
    return json.loads(ideal)
  return ideal


def get_test_case(test_name_or_index):
  if isinstance(test_name_or_index, int):
    if 0 <= test_name_or_index < len(TEST_CASES):
      return TEST_CASES[test_name_or_index]
    return None
  if isinstance(test_name_or_index, str):
    for test_case in TEST_CASES:
      if test_case["name"] == test_name_or_index:
        return test_case
  return None


def _compare_expected_categories(actual_items, ideal_items):
  actual_map = {str(item.get("transaction_id")): item.get("category") for item in actual_items}
  ideal_map = {str(item.get("transaction_id")): item.get("category") for item in ideal_items}
  if actual_map != ideal_map:
    return False, f"category mismatch model={actual_map} ideal={ideal_map}"
  return True, "transaction_id to category mapping matches reference"


def _run_test_with_logging(tc: dict, categorizer: RethinkTransactionCategorization = None):
  if categorizer is None:
    categorizer = RethinkTransactionCategorization()

  print("\n" + "=" * 80)
  print(f"Running categorization test: {tc['name']}")
  print("=" * 80)
  print("\nLLM INPUT:")
  print(tc["input"])

  result = categorizer.generate_response(tc["input"])

  print("\nLLM OUTPUT:")
  print(json.dumps(result, indent=2))

  ideal = _ideal_outcome(tc)
  if ideal is not None:
    print("\nEXPECTED OUTPUT (compact):")
    print(json.dumps(ideal, indent=2))
    ok, detail = _compare_expected_categories(result, ideal)
    print("\nCategory mapping match:", "PASS" if ok else "FAIL")
    print(detail)

  print("\n" + "=" * 80 + "\n")
  return result


def run_test(test_name_or_index_or_dict, categorizer: RethinkTransactionCategorization = None):
  if isinstance(test_name_or_index_or_dict, dict):
    tc = test_name_or_index_or_dict
  else:
    tc = get_test_case(test_name_or_index_or_dict)

  if tc is None:
    print(f"Test case '{test_name_or_index_or_dict}' not found.")
    return None

  return _run_test_with_logging(tc, categorizer)


def run_all_tests_batch(categorizer: RethinkTransactionCategorization = None, batch_num: int = 1):
  if categorizer is None:
    categorizer = RethinkTransactionCategorization()
  for tc in [x for x in TEST_CASES if x.get("batch") == batch_num]:
    run_test(tc, categorizer)


def main(test: str = None, run_batch: bool = False, batch_num: int = 1, model: str = None):
  kw = {}
  if model is not None:
    kw["model_name"] = model
  categorizer = RethinkTransactionCategorization(**kw)

  if run_batch:
    run_all_tests_batch(categorizer, batch_num=batch_num)
    return
  if test is not None:
    if test.strip().lower() == "all":
      for batch in sorted({int(tc.get("batch", 1)) for tc in TEST_CASES}):
        run_all_tests_batch(categorizer, batch_num=batch)
      return
    run_test(int(test) if test.isdigit() else test, categorizer)
    return

  print("Available test cases:")
  for i, tc in enumerate(TEST_CASES):
    print(f"  {i}: {tc['name']} (batch {tc.get('batch')})")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Rethink transaction categorization optimizer harness.")
  parser.add_argument("--test", type=str, default=None)
  parser.add_argument("--batch", type=int, nargs="?", const=1, default=None, metavar="N", help="Run the combined multi-id harness case for batch N (1-4; default 1).")
  parser.add_argument("--model", type=str, default=None)
  args = parser.parse_args()
  main(test=args.test, run_batch=args.batch is not None, batch_num=(1 if args.batch is None else args.batch), model=args.model)

