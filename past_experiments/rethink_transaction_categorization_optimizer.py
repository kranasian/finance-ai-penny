from google import genai
from google.genai import types
import os
import json
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are a financial transaction categorization expert. Analyze each transaction and categorize it using the category list in this prompt.

## Input Format
JSON array of transaction groups. Each entry contains:
- `establishment_name`: Merchant/establishment name
- `establishment_description`: What the establishment provides
- `transactions`: Array with `transaction_id`, `transaction_text` (raw bank statement text), and `amount`

## Rational Basis for Category
Choose the category only from evidence in: establishment_name, establishment_description, transaction_text, and/or amount. Do not guess. Assume the subcategory list in this prompt is comprehensive enough that every transaction should map to at least one listed subcategory unless an explicit `unknown` condition applies. Income categories (e.g. income_business, income_salary, income_side_gig) are only for inflows that represent earnings. A purchase or payment (positive amount, outflow) must never be categorized as any income category—use an expense or transfer category instead.

## Inflows (negative amount)
Negative amounts are inflows. Not all inflows are income. Refunds, returns, or reversals of a prior expense should be categorized as the same (or appropriate) expense subcategory when identifiable; use income categories only when the inflow clearly represents earnings (e.g. salary, interest, business revenue).

## Parent Category and Subcategory Meanings
- **income**:
  - `income_salary`: regular paycheck income.
  - `income_interest`: bank or stock interest earned.
  - `income_sidegig`: income from freelance or secondary work.
  - `income_business`: income generated from a business entity.
- **meals**:
  - `food_groceries`: food purchased for home cooking.
  - `food_dining_out`: restaurant meals.
  - `food_delivered_food`: food ordered via delivery.
- **leisure**:
  - `leisure_entertainment`: movies, concerts, events, recreation.
  - `leisure_travel_vacations`: costs related to trips and vacations.
- **bills**:
  - `bills_connectivity`: internet, cable, phone bills.
  - `bills_insurance`: premiums for health, auto, home insurance.
  - `bills_tax`: payments made for local, state, or federal taxes.
  - `bills_service_fees`: bank fees or administrative charges.
- **shelter**:
  - `shelter_home`: mortgage or rent payments.
  - `shelter_utilities`: electricity, gas, water, trash services.
  - `shelter_upkeep`: home repairs and maintenance.
- **education**:
  - `education_kids_activities`: costs for children's classes or sports.
  - `education_tuition`: costs for personal or dependent education.
- **shopping**:
  - `shopping_clothing`: apparel purchases.
  - `shopping_gadgets`: electronics and technology purchases.
  - `shopping_kids`: general shopping for children.
  - `shopping_pets`: supplies and services for pets.
- **transportation**:
  - `transport_public`: bus, train, subway fares.
  - `transport_car_fuel`: gas, maintenance, parking, or car payments.
- **health**:
  - `health_medical_pharmacy`: doctor visits, prescriptions.
  - `health_gym_wellness`: gym memberships, supplements.
  - `health_personal_care`: haircuts, toiletries.
- **donations_gifts**: charitable giving or gifts for others.
- **transfer**: Strictly (1) movement of money between the same person's own accounts (e.g. checking to savings, ACH between own accounts), or (2) payment toward that person's own debts, mortgages, or other liabilities (e.g. credit card payment to own card, loan payment to own loan, mortgage payment to own mortgage). Net worth unchanged. Peer-to-peer to/from another person is NOT transfer unless it's to the same person's account.

## Analysis Process
1. **Transaction Text** (primary): Extract keywords and patterns indicating purpose
2. **Establishment Context**: Use name and description to understand business type
3. **Amount Context**: Large amounts may indicate services/procedures; small amounts may indicate supplies/incidentals. Negative amounts represent inflows (e.g., income, refunds), while positive amounts represent outflows (e.g., purchases, expenses).
4. **Consistency**: Similar transactions should typically share the same category, unless amounts suggest otherwise.
5. **Reasoning**: When writing reasoning, ONLY state positive evidence for the chosen category. DO NOT mention what categories are available/unavailable or why others don't fit.
6. **Deterministic Decision Order**:
   - First decide whether transfer-same-person evidence is explicit.
   - If not explicit, classify by purpose-specific subcategory when purpose is clear.
   - Use `unknown` only for the allowed unknown conditions below.

## Confidence Levels
- **high**: Strong evidence from transaction text, establishment, and amount align clearly
- **medium**: Good evidence with some ambiguity or conflicting signals
- **low**: Weak evidence, multiple plausible categories, or significant uncertainty

## Critical Rules
- **Subcategory only**: Your output `category` MUST NEVER be a parent category (e.g., `income`, `meals`, `leisure`, `bills`, `shelter`, `education`, `shopping`, `transportation`, `health`). You MUST always select a specific subcategory (e.g., `shopping_clothing` is allowed, but `shopping` is forbidden). This applies even if the parent category is the only one that seems to fit—in that case, use `unknown`.
- **Unknown Category**: Use `unknown` ONLY in these cases:
    1. The transaction purpose is too vague to understand what it is for.
    2. The establishment's scope is too broad to map confidently to just one subcategory.
- **Match Best Subcategory**: Choose the subcategory from this prompt that most specifically describes the transaction. Prefer the most specific child category over broader labels.
- **CRITICAL ID Matching**: The `transaction_id` in your output response MUST be an EXACT, character-for-character copy of the `transaction_id` from the input. Your output MUST ONLY contain: `transaction_id`, `reasoning`, `category`, and `confidence`.
- **CRITICAL Transfer Rule**: `transfer` ONLY when (a) money moves between the same person's own accounts (e.g. Transfer To Checking, Transfer From Savings, ACH between own accounts), or (b) payment is toward that person's own debt, mortgage, or other liability (e.g. credit card payment to own card, loan payment to own loan, mortgage payment to own mortgage).
- **CRITICAL Transfer Evidence Rule**: Use `transfer` only when same-person ownership is explicit or strongly implied by direct evidence in transaction text and merchant context (e.g. "to own credit card", "from my savings", payment to card servicer for user's liability). If same-person ownership is not explicit/inferable, use `unknown`.
- **Do Not Infer Ownership from Generic Transfer Wording**: Phrases like "transfer from savings", "transfer to checking", masked account numbers, or generic "internal transfer" descriptions are NOT sufficient by themselves to prove same-person ownership. Without explicit ownership evidence, output `unknown`.
- **CRITICAL Bank/Person-to-Person (P2P) Rule**: For transfers or P2P payments (Zelle, Venmo, PayPal, Cash App, etc.):
    1. If the transfer is to an account belonging to the **SAME person**, use `transfer`.
    2. If the transfer is to **ANOTHER person**:
        - If the purpose is **specified** (e.g., "for dinner", "rent"), select the subcategory matching that purpose.
        - If the purpose is **unspecified**, use `unknown`.
    3. If you are **unsure** whether it is to the same person or another person, use `unknown`.
    4. If sender/recipient identity relation (same person vs another person) is not explicitly inferable, use `unknown`.
    5. Example handling:
        - "Transfer from SV ***2131" -> `unknown` (could be another person's account; same-person ownership not established).
        - "Credit Card Payment" -> `transfer` (typically payment toward own liability unless evidence indicates otherwise).
        - "Transfer From Savings 0012" without explicit ownership -> `unknown`.
- **Transfer Ambiguity Precedence**: When transfer ownership evidence conflicts (or is weak) across fields, prioritize ambiguity handling and output `unknown` rather than `transfer`.
- **Category Selection**: The `category` value in your output MUST be an exact character-for-character copy of one subcategory listed in this prompt, or the literal `unknown`. Do not alter spelling, casing, or wording. Parent categories are forbidden. For general-purpose marketplaces (e.g. Amazon, Shopee, Walmart) when the specific purchase is unknown, use `unknown`. When a transaction is clearly general shopping (e.g. a retail store where the specific product is not stated), choose the most plausible shopping subcategory based on establishment type, description, or amount—and state that basis briefly in reasoning (e.g. "General retail; clothing store description suggests shopping_clothing."). Do not use a parent shopping category when a subcategory is available. **Outflows (positive amount)**: Never use income_side_gig, income_business, or income_salary for a payment or purchase—only for money received as earnings. Use an expense or transfer category for outflows.
- **Evidence Priority**: Prefer direct evidence from `transaction_text`, then establishment name/description, then amount heuristics. Do not let broad or generic establishment descriptions override direct transfer-identity ambiguity in the transaction text.
- **CRITICAL Reasoning Rule**: Reasoning must be concise. State ONLY positive evidence that supports the chosen category. Do not mention available options, subcategory lists, or other categories. For `unknown`, state briefly why purpose cannot be determined.
- **Consistency**: Similar transactions should typically share the same category, unless amounts suggest otherwise.
- When uncertain but a category is plausible, use "medium" or "low" confidence and briefly explain the uncertainty
- **Specific Examples**:
    - **AAA**: Roadside assistance or membership dues should be `bills_insurance`; if evidence indicates direct vehicle operational costs instead, use `transport_car_fuel`.
    - **Software/Digital Services**: Subscriptions for design tools, AI, or apps should be `bills_service_fees` or `shopping_gadgets` depending on the description. If neither fits well, use `unknown`.
"""

RESPONSE_SCHEMA = {
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "transaction_id": {
        "anyOf": [
          {"type": "string"},
          {"type": "integer"},
        ]
      },
      "reasoning": {"type": "string"},
      "category": {"type": "string"},
      "confidence": {"type": "string"},
    },
    "required": ["transaction_id", "reasoning", "category", "confidence"],
  },
}

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
      response_schema=RESPONSE_SCHEMA,
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
      # Without a response schema, the model may wrap the array in prose; try the outermost [...] slice.
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
    "name": "batch_1_salary_fuel_utility_mix",
    "input": """[
  {
    "establishment_name": "ACME Corp Payroll",
    "establishment_description": "Biweekly employee payroll deposit from employer.",
    "transactions": [
      {
        "transaction_id": 7001001,
        "transaction_text": "Acme Corp Payroll Direct Dep",
        "amount": -2480.55
      }
    ]
  },
  {
    "establishment_name": "Shell Gas",
    "establishment_description": "Fuel station for gasoline purchases.",
    "transactions": [
      {
        "transaction_id": 7001002,
        "transaction_text": "Shell Oil 574421 Boston Ma",
        "amount": 52.34
      }
    ]
  },
  {
    "establishment_name": "City Water Utility",
    "establishment_description": "Municipal water and sewer utility billing.",
    "transactions": [
      {
        "transaction_id": 7001003,
        "transaction_text": "City Water Autopay",
        "amount": 78.11
      }
    ]
  }
]""",
    "output": """[
  {
    "transaction_id": 7001001,
    "reasoning": "Direct payroll deposit from an employer indicates regular paycheck income.",
    "category": "income_salary",
    "confidence": "high"
  },
  {
    "transaction_id": 7001002,
    "reasoning": "The merchant is a gas station and the charge is for fuel.",
    "category": "transport_car_fuel",
    "confidence": "high"
  },
  {
    "transaction_id": 7001003,
    "reasoning": "The payment is to a city water utility for household utility service.",
    "category": "shelter_utilities",
    "confidence": "high"
  }
]""",
  },
  {
    "batch": 2,
    "name": "batch_2_transfer_and_unknown_p2p",
    "input": """[
  {
    "establishment_name": "Chase Credit Card",
    "establishment_description": "Payment to user's own credit card account.",
    "transactions": [
      {
        "transaction_id": 7002001,
        "transaction_text": "Online Payment To Chase Card 4432",
        "amount": 500.00
      }
    ]
  },
  {
    "establishment_name": "Venmo",
    "establishment_description": "Peer-to-peer payment app.",
    "transactions": [
      {
        "transaction_id": 7002002,
        "transaction_text": "Venmo Payment To Alex",
        "amount": 45.00
      }
    ]
  },
  {
    "establishment_name": "Transfer From SV ***0012",
    "establishment_description": "Internal transfer from a savings account.",
    "transactions": [
      {
        "transaction_id": 7002003,
        "transaction_text": "Transfer From Savings 0012",
        "amount": -300.00
      }
    ]
  }
]""",
    "output": """[
  {
    "transaction_id": 7002001,
    "reasoning": "The transaction text shows a payment to the user's own credit card account.",
    "category": "transfer",
    "confidence": "high"
  },
  {
    "transaction_id": 7002002,
    "reasoning": "This is a P2P payment to another person with no stated purpose.",
    "category": "unknown",
    "confidence": "high"
  },
  {
    "transaction_id": 7002003,
    "reasoning": "Unclear what the transaction is for.",
    "category": "unknown",
    "confidence": "high"
  }
]""",
  },
  {
    "batch": 3,
    "name": "batch_3_delivery_refund_tax",
    "input": """[
  {
    "establishment_name": "Uber Eats",
    "establishment_description": "Food delivery platform.",
    "transactions": [
      {
        "transaction_id": 7003001,
        "transaction_text": "Uber Eats Help.Uber.Com",
        "amount": 27.89
      }
    ]
  },
  {
    "establishment_name": "Uber Eats",
    "establishment_description": "Food delivery platform.",
    "transactions": [
      {
        "transaction_id": 7003002,
        "transaction_text": "Uber Eats Refund",
        "amount": -27.89
      }
    ]
  },
  {
    "establishment_name": "State Tax Board",
    "establishment_description": "State government tax payment portal.",
    "transactions": [
      {
        "transaction_id": 7003003,
        "transaction_text": "State Tax Web Pmt",
        "amount": 420.00
      }
    ]
  }
]""",
    "output": """[
  {
    "transaction_id": 7003001,
    "reasoning": "The charge is from a food delivery platform for delivered meals.",
    "category": "food_delivered_food",
    "confidence": "high"
  },
  {
    "transaction_id": 7003002,
    "reasoning": "The transaction is a refund from the same food delivery merchant.",
    "category": "food_delivered_food",
    "confidence": "high"
  },
  {
    "transaction_id": 7003003,
    "reasoning": "The payment is made to a state tax authority for taxes.",
    "category": "bills_tax",
    "confidence": "high"
  }
]""",
  },
  {
    "batch": 4,
    "name": "batch_4_marketplace_gym_tuition",
    "input": """[
  {
    "establishment_name": "Amazon Marketplace",
    "establishment_description": "General online marketplace with mixed product types.",
    "transactions": [
      {
        "transaction_id": 7004001,
        "transaction_text": "Amazon Mktplace Pmts Amzn.Com/Bill",
        "amount": 63.42
      }
    ]
  },
  {
    "establishment_name": "Planet Fitness",
    "establishment_description": "Gym and wellness membership facility.",
    "transactions": [
      {
        "transaction_id": 7004002,
        "transaction_text": "Planet Fitness Club Monthly",
        "amount": 10.00
      }
    ]
  },
  {
    "establishment_name": "State University",
    "establishment_description": "University tuition and academic fees.",
    "transactions": [
      {
        "transaction_id": 7004003,
        "transaction_text": "State University Tuition Payment",
        "amount": 1850.00
      }
    ]
  }
]""",
    "output": """[
  {
    "transaction_id": 7004001,
    "reasoning": "The merchant is a general marketplace and the specific purchase type is not identified.",
    "category": "unknown",
    "confidence": "high"
  },
  {
    "transaction_id": 7004002,
    "reasoning": "The recurring membership charge is for a gym facility.",
    "category": "health_gym_wellness",
    "confidence": "high"
  },
  {
    "transaction_id": 7004003,
    "reasoning": "The payment is explicitly for university tuition.",
    "category": "education_tuition",
    "confidence": "high"
  }
]""",
  },
]

_optimizer_batch_sizes = {}
for _tc in TEST_CASES:
  _bn = int(_tc.get("batch", 1))
  _optimizer_batch_sizes[_bn] = _optimizer_batch_sizes.get(_bn, 0) + 1
assert sorted(_optimizer_batch_sizes.keys()) == [1, 2, 3, 4], _optimizer_batch_sizes
assert all(_count == _MAX_CASES_PER_OPTIMIZER_BATCH for _count in _optimizer_batch_sizes.values()), _optimizer_batch_sizes


def _compare_expected_categories(actual_items, ideal_items):
  actual_map = {str(item.get("transaction_id")): item.get("category") for item in actual_items}
  ideal_map = {str(item.get("transaction_id")): item.get("category") for item in ideal_items}
  if actual_map != ideal_map:
    return False, f"category mismatch model={actual_map} ideal={ideal_map}"
  return True, "transaction_id to category mapping matches reference"


def run_test(test_name_or_index_or_dict, categorizer: RethinkTransactionCategorization = None):
  if isinstance(test_name_or_index_or_dict, dict):
    tc = test_name_or_index_or_dict
  elif isinstance(test_name_or_index_or_dict, int):
    idx = test_name_or_index_or_dict
    if not (0 <= idx < len(TEST_CASES)):
      return None
    tc = TEST_CASES[idx]
  else:
    tc = next((t for t in TEST_CASES if t["name"] == test_name_or_index_or_dict), None)
    if not tc:
      return None

  print(f"# Test: **{tc['name']}**")
  if categorizer is None:
    categorizer = RethinkTransactionCategorization()

  llm_input = tc["input"]
  result = categorizer.generate_response(llm_input)
  print("## LLM Output:")
  print(json.dumps(result, indent=2))
  if tc.get("output"):
    print("## Ideal output (reference):")
    print(tc["output"])
    try:
      expected = json.loads(tc["output"])
      ok, detail = _compare_expected_categories(result, expected)
      print("## Category mapping match:", "PASS" if ok else "FAIL")
      print(detail)
    except json.JSONDecodeError as exc:
      print(f"Failed to parse reference output: {exc}")
  return result


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

