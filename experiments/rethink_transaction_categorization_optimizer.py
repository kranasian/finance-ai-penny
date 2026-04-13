from google import genai
from google.genai import types
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are a financial transaction categorization expert. Analyze each transaction and categorize it using the provided category options.

## Input Format
JSON array of transaction groups. Each entry contains:
- `establishment_name`: Merchant/establishment name
- `establishment_description`: What the establishment provides
- `transactions`: Array with `transaction_id`, `transaction_text` (raw bank statement text), and `amount`
- `category_options`: Array of valid category strings. Your output `category` MUST be a character-for-character exact copy of one of these strings—no rephrasing, no spelling changes, no singular/plural variants. Copy the string exactly as it appears in `category_options`. When both a parent and a subcategory from the same hierarchy appear, output the subcategory string exactly as listed.

## Rational Basis for Category
Choose the category only from evidence in: establishment_name, establishment_description, transaction_text, and/or amount. Do not guess. If there is no clear evidence tying the transaction to a specific category in category_options, use `unknown`. `unknown` is always allowed even if not listed in category_options. Income categories (e.g. income_business, income_salary, income_side_gig) are only for inflows that represent earnings. A purchase or payment (positive amount, outflow) must never be categorized as any income category—use an expense or transfer category instead.

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

## Confidence Levels
- **high**: Strong evidence from transaction text, establishment, and amount align clearly
- **medium**: Good evidence with some ambiguity or conflicting signals
- **low**: Weak evidence, multiple plausible categories, or significant uncertainty

## Critical Rules
- **Subcategory only**: Your output `category` MUST NEVER be a parent category (e.g., `income`, `meals`, `leisure`, `bills`, `shelter`, `education`, `shopping`, `transportation`, `health`). You MUST always select a specific subcategory (e.g., `shopping_clothing` is allowed, but `shopping` is forbidden). This applies even if the parent category is the only one that seems to fit—in that case, use `unknown`.
- **Unknown Category**: Use `unknown` ONLY if none of the specific subcategories in `category_options` match the transaction. If you are unsure or if only a parent category matches, use `unknown`.
- **Match Best Subcategory**: When multiple subcategories are available, choose the one that most specifically describes the transaction. For example, if both `bills` and `bills_connectivity` are in `category_options` for an internet bill, you MUST choose `bills_connectivity`. If only `bills` is available, you MUST use `unknown` because `bills` is a parent category.
- **CRITICAL ID Matching**: The `transaction_id` in your output response MUST be an EXACT, character-for-character copy of the `transaction_id` from the input. Your output MUST ONLY contain: `transaction_id`, `reasoning`, `category`, and `confidence`.
- **CRITICAL Transfer Rule**: `transfer` ONLY when (a) money moves between the same person's own accounts (e.g. Transfer To Checking, Transfer From Savings, ACH between own accounts), or (b) payment is toward that person's own debt, mortgage, or other liability (e.g. credit card payment to own card, loan payment to own loan, mortgage payment to own mortgage).
- **CRITICAL Bank/Person-to-Person (P2P) Rule**: For transfers or P2P payments (Zelle, Venmo, PayPal, Cash App, etc.):
    1. If the transfer is to an account belonging to the **SAME person**, use `transfer`.
    2. If the transfer is to **ANOTHER person**:
        - If the purpose is **specified** (e.g., "for dinner", "rent"), select the subcategory matching that purpose.
        - If the purpose is **unspecified**, use `unknown`.
    3. If you are **unsure** whether it is to the same person or another person, use `unknown`.
- **Category Selection**: The `category` value in your output MUST be an exact character-for-character copy of one string from `category_options`, or the literal `unknown`. Do not alter spelling, casing, or wording. For general-purpose marketplaces (e.g. Amazon, Shopee, Walmart) when the specific purchase is unknown, use `unknown`. When a transaction is clearly general shopping (e.g. a retail store where the specific product is not stated), choose the most plausible shopping subcategory from `category_options` based on establishment type, description, or amount—and state that basis briefly in reasoning (e.g. "General retail; clothing store description suggests shopping_clothing."). Do not use a parent shopping category when a subcategory is available. **Outflows (positive amount)**: Never use income_side_gig, income_business, or income_salary for a payment or purchase—only for money received as earnings. Use an expense or transfer category for outflows.
- **CRITICAL Reasoning Rule**: Reasoning must be concise. State ONLY positive evidence that supports the chosen category. Do not mention category_options, subcategory, "options include", or other categories. For `unknown`, state briefly why purpose cannot be determined.
- **Consistency**: Similar transactions should typically share the same category, unless amounts suggest otherwise.
- When uncertain but a category is plausible, use "medium" or "low" confidence and briefly explain the uncertainty
- **Specific Examples**:
    - **AAA**: Roadside assistance or membership dues should be `bills_insurance` if available, or `transport_car_fuel` if it's the only car-related option.
    - **Software/Digital Services**: Subscriptions for design tools, AI, or apps should be `bills_service_fees` or `shopping_gadgets` depending on the description. If neither fits well, use `unknown`.
"""

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


# Test data dictionary keyed by fixture id (internal lookup only; not sent as model input).
TEST_DATA = {
  8809: {
    "establishment_name": "Macho's",
    "establishment_description": "sells Mexican food such as tacos, burritos, and quesadillas",
    "transactions": [
      {
        "transaction_id": 1,
        "transaction_text": "PY *MACHO SE REF# 509700022869 972-525-0686,TX Card ending in 3458",
        "amount": 40.00
      },
      {
        "transaction_id": 2,
        "transaction_text": "PY *MACHO SE REF# 506600014887 972-525-0686,TX Card ending in 3458",
        "amount": 40.00
      }
    ],
    "category_options": [
      "food_dining_out",
      "food_delivered_food",
      "donations_gifts",
      "food",
      "income_business",
      "food_groceries",
      "leisure_travel_vacations",
      "leisure"
    ]
  },
  12723: {
    "establishment_name": "Marco's Pizza",
    "establishment_description": "pizza restaurant that sells a variety of pizzas, subs, wings, sides, and desserts",
    "transactions": [
      {
        "transaction_id": 1,
        "transaction_text": "Marco's Pizza",
        "amount": 16.23
      }
    ],
    "category_options": [
      "food_dining_out",
      "food_delivered_food",
      "donations_gifts",
      "leisure_travel_vacations",
      "food_groceries",
      "food",
      "bills_service_fees",
      "income_business"
    ]
  },
  8965: {
    "establishment_name": "Marcos Pizza",
    "establishment_description": "A payment for food and beverages at a pizza restaurant.",
    "transactions": [
      {
        "transaction_id": 1,
        "transaction_text": "MARCOS PIZZA REF# 505100025146 WAXAHACHIE,TX Card ending in 3458",
        "amount": 31.37
      }
    ],
    "category_options": [
      "food_dining_out",
      "food_delivered_food",
      "donations_gifts",
      "leisure_travel_vacations",
      "food_groceries",
      "food",
      "income_business"
    ]
  },
  11711: {
    "establishment_name": "Seamless",
    "establishment_description": "sells food delivery services from various restaurants",
    "transactions": [
      {
        "transaction_id": 1,
        "transaction_text": "Seamless",
        "amount": 84.54
      }
    ],
    "category_options": [
      "food_delivered_food",
      "food_dining_out",
      "donations_gifts",
      "leisure_travel_vacations",
      "income_business"
    ]
  },
  4966: {
    "establishment_name": "Cos",
    "establishment_description": "sells clothing, accessories, and beauty products for women",
    "transactions": [
      {
        "transaction_id": 1,
        "transaction_text": "COS WEB",
        "amount": 108.17
      }
    ],
    "category_options": [
      "shopping_clothing",
      "donations_gifts",
      "shopping"
    ]
  },
  12671: {
    "establishment_name": "Cleveland Marriott",
    "establishment_description": "sells hotel accommodations, meeting rooms, and other services",
    "transactions": [
      {
        "transaction_id": 1,
        "transaction_text": "MOBILE PURCHASE 0501 CLE EMBERS 681115 CLEVELAND OH XXXXX8751XXXXXXXXXX5621",
        "amount": 23.67
      },
      {
        "transaction_id": 2,
        "transaction_text": "MOBILE PURCHASE CLE EMBERS 681115 CLEVELAND OH ON 05/01",
        "amount": 5.40
      },
      {
        "transaction_id": 3,
        "transaction_text": "MOBILE PURCHASE CLE SPORTS ST1576 CLEVELAND OH ON 05/01",
        "amount": 46.04
      },
      {
        "transaction_id": 4,
        "transaction_text": "MOBILE PURCHASE CLEVELAND MARRIOT CLEVELAND OH ON 04/30",
        "amount": 43.88
      }
    ],
    "category_options": [
      "leisure_travel_vacations",
      "food_dining_out",
      "bills_service_fees",
      "income_business",
      "leisure_entertainment",
      "donations_gifts",
      "shelter_home",
      "leisure"
    ]
  },
  11741: {
    "establishment_name": "Marriott",
    "establishment_description": "sells hotel accommodations, resort stays, and other travel-related services",
    "transactions": [
      {
        "transaction_id": 1,
        "transaction_text": "Marriott",
        "amount": 23.47
      }
    ],
    "category_options": [
      "leisure_travel_vacations",
      "bills_service_fees",
      "income_business",
      "donations_gifts",
      "transport_car_fuel",
      "shelter_home",
      "food_dining_out",
      "leisure_entertainment"
    ]
  },
  4613: {
    "establishment_name": "Salvatore's",
    "establishment_description": "sells Italian food such as pizza, pasta, sandwiches, and salads",
    "transactions": [
      {
        "transaction_id": 1,
        "transaction_text": "SALVATORE P BRUNO",
        "amount": 70.00
      }
    ],
    "category_options": [
      "food_dining_out",
      "food_delivered_food",
      "donations_gifts",
      "leisure_travel_vacations",
      "food",
      "food_groceries",
      "income_business"
    ]
  },
  12712: {
    "establishment_name": "Farm Carol San Pedro",
    "establishment_description": "sells groceries, including fresh produce, meat, dairy, packaged goods, and household items",
    "transactions": [
      {
        "transaction_id": 1,
        "transaction_text": "Pos Debit 3151 Farm Carol San Ped San Pedro DO",
        "amount": 23.69
      }
    ],
    "category_options": [
      "food_groceries",
      "food_dining_out",
      "donations_gifts",
      "leisure_travel_vacations",
      "transport_car_fuel",
      "food",
      "income_business",
      "food_delivered_food"
    ]
  },
  12633: {
    "establishment_name": "Zeko's Grill",
    "establishment_description": "sells various types of grilled meats, sandwiches, and other food items",
    "transactions": [
      {
        "transaction_id": 1,
        "transaction_text": "ZEKO'S GRILL RIVERVIRIVERVIEW",
        "amount": 12.77
      }
    ],
    "category_options": [
      "food_dining_out",
      "food_delivered_food",
      "donations_gifts",
      "leisure_travel_vacations",
      "food",
      "food_groceries",
      "income_business"
    ]
  },
  3578: {
    "establishment_name": "AT&T Bill Payment",
    "establishment_description": "bill payment for AT&T services like internet, phone, and television",
    "transactions": [
      {
        "transaction_id": 1,
        "transaction_text": "ATT*BILL PAYMENT US",
        "amount": 247.15
      }
    ],
    "category_options": [
      "bills_connectivity",
      "shopping_gadgets",
      "bills_service_fees",
      "bills"
    ]
  },
  8657: {
    "establishment_name": "Transfer to Checking",
    "establishment_description": "transfer of funds from one checking account to another",
    "transactions": [
      {
        "transaction_id": 1,
        "transaction_text": "Transfer To Checking -6534",
        "amount": 1000.00
      },
      {
        "transaction_id": 2,
        "transaction_text": "Transfer To Checking -2833",
        "amount": 667.00
      },
      {
        "transaction_id": 3,
        "transaction_text": "Transfer To Checking -1369",
        "amount": 25.00
      },
      {
        "transaction_id": 4,
        "transaction_text": "Transfer To Checking -7101",
        "amount": 14.00
      }
    ],
    "category_options": [
      "transfer",
      "shelter_home",
      "income_business",
      "bills",
      "bills_service_fees",
      "transport_car_fuel",
      "income_side_gig",
      "income_salary"
    ]
  },
  8377: {
    "establishment_name": "F & S Metro News",
    "establishment_description": "sells newspapers, magazines, and other periodicals",
    "transactions": [
      {
        "transaction_id": 1,
        "transaction_text": "MOBILE PURCHASE F & S METRO NEWS NEW HYDE PARK NY ON 03/07",
        "amount": 17.64
      },
      {
        "transaction_id": 2,
        "transaction_text": "MOBILE PURCHASE 0307 F & S METRO NEWS NEW HYDE PARKNY XXXXX5350XXXXXXXXXX7572",
        "amount": 11.30
      }
    ],
    "category_options": [
      "leisure_entertainment",
      "education_kids_activities",
      "bills_service_fees",
      "donations_gifts",
      "bills_connectivity",
      "income_business",
      "bills",
      "leisure"
    ]
  },
  12586: {
    "establishment_name": "Placeit",
    "establishment_description": "sells website templates, mockups, and other design assets",
    "transactions": [
      {
        "transaction_id": 1,
        "transaction_text": "PLACEIT EMPOWERKIT",
        "amount": 14.95
      }
    ],
    "category_options": [
      "income_business",
      "bills_service_fees",
      "income_side_gig",
      "leisure_entertainment",
      "education_kids_activities",
      "education_tuition",
      "shopping_gadgets",
      "donations_gifts"
    ]
  },
  12583: {
    "establishment_name": "Cleo Express Fee",
    "establishment_description": "A recurring fee for a financial assistant application.",
    "transactions": [
      {
        "transaction_id": 1,
        "transaction_text": "Cleo* Advance+Express",
        "amount": 19.98
      },
      {
        "transaction_id": 2,
        "transaction_text": "Cleo* Advance+Express Meetcleo.com Deus",
        "amount": 29.98
      }
    ],
    "category_options": [
      "bills_service_fees",
      "transfer",
      "bills",
      "bills_insurance"
    ]
  },
  4417: {
    "establishment_name": "OpenAI",
    "establishment_description": "sells access to its large language model API, including text generation, translation, and code completion",
    "transactions": [
      {
        "transaction_id": 1,
        "transaction_text": "OPENAI",
        "amount": 6.00
      },
      {
        "transaction_id": 2,
        "transaction_text": "OPENAI +14158799686 USA",
        "amount": 5.32
      }
    ],
    "category_options": [
      "shopping_gadgets",
      "donations_gifts",
      "bills_service_fees",
      "food_dining_out",
      "shelter_upkeep",
      "shelter_home",
      "shopping",
      "shopping_clothing"
    ]
  },
  12562: {
    "establishment_name": "OpenAI Chatgpt",
    "establishment_description": "sells subscriptions to use the AI chatbot for various tasks such as writing, coding, and problem-solving",
    "transactions": [
      {
        "transaction_id": 1,
        "transaction_text": "Openai Chatgpt Subscr",
        "amount": 21.28
      }
    ],
    "category_options": [
      "bills_service_fees",
      "income_business",
      "education_tuition",
      "leisure_entertainment",
      "bills"
    ]
  },
  651: {
    "establishment_name": "SFO Parking",
    "establishment_description": "sells parking services at San Francisco International Airport, with options for short-term and long-term parking",
    "transactions": [
      {
        "transaction_id": 1,
        "transaction_text": "60776 - Sfo Parkingcentrasan Franciscoca",
        "amount": 4.00
      }
    ],
    "category_options": [
      "transport_car_fuel",
      "transport",
      "bills_service_fees"
    ]
  },
  4485: {
    "establishment_name": "Equinox",
    "establishment_description": "sells fitness classes, personal training, and other wellness services",
    "transactions": [
      {
        "transaction_id": 1,
        "transaction_text": "CHECKCARD EQUINOX MOTO #142 XXXXX26549 NY ON 10/25",
        "amount": 200.00
      },
      {
        "transaction_id": 2,
        "transaction_text": "PURCHASE Equinox 105 New York NY ON 01/06",
        "amount": 30.00
      },
      {
        "transaction_id": 3,
        "transaction_text": "CHECKCARD 11/23 EQUINOX 105",
        "amount": 200.00
      },
      {
        "transaction_id": 4,
        "transaction_text": "CHECKCARD 0523 EQUINOX MOTO #142 XXXXX26549 NY XXXXX2041XXXXXXXXXX4351 RECURRING",
        "amount": 200.00
      }
    ],
    "category_options": [
      "health_gym_wellness",
      "health_personal_care",
      "leisure_entertainment",
      "bills_service_fees",
      "donations_gifts"
    ]
  },
  8611: {
    "establishment_name": "Hand and Stone",
    "establishment_description": "offers professional massage, facial, and hair removal services",
    "transactions": [
      {
        "transaction_id": 1,
        "transaction_text": "Hand and Stone 180-37260364",
        "amount": 40.00
      },
      {
        "transaction_id": 2,
        "transaction_text": "Hand and Stone",
        "amount": 14.00
      }
    ],
    "category_options": [
      "health_gym_wellness",
      "health_personal_care",
      "health",
      "leisure",
      "donations_gifts"
    ]
  },
  8460: {
    "establishment_name": "Ace Hardware",
    "establishment_description": "sells a wide variety of hardware, tools, paint, plumbing supplies, electrical supplies, and lawn and garden products",
    "transactions": [
      {
        "transaction_id": 1,
        "transaction_text": "Ace Hardware",
        "amount": 20.24
      }
    ],
    "category_options": [
      "shelter_upkeep",
      "shopping_clothing",
      "shopping_kids",
      "food_groceries",
      "donations_gifts"
    ]
  },
  6458: {
    "establishment_name": "Bob's Burgers & Brew",
    "establishment_description": "sells burgers, sandwiches, and other pub fare",
    "transactions": [
      {
        "transaction_id": 1,
        "transaction_text": "TST* BOBS BURGER & BYAKIMA",
        "amount": 30.78
      }
    ],
    "category_options": [
      "shopping_gadgets",
      "donations_gifts",
      "bills_service_fees",
      "food_dining_out",
      "shelter_upkeep",
      "shelter_home",
      "shopping",
      "shopping_clothing"
    ]
  },
  8103: {
    "establishment_name": "AAA",
    "establishment_description": "membership for roadside assistance, travel services, and insurance",
    "transactions": [
      {
        "transaction_id": 1,
        "transaction_text": "PURCHASE 02/19 AAA MEMBERSHIP DUES XXXXX82000",
        "amount": 41.98
      },
      {
        "transaction_id": 2,
        "transaction_text": "AAA TX MEMBE REF# 436000029821 800-765-0766,CA Card ending in 3458",
        "amount": 58.66
      }
    ],
    "category_options": [
      "transport_car_fuel",
      "bills_insurance",
      "shelter_home",
      "income_business",
      "bills_service_fees",
      "bills",
      "shelter",
      "donations_gifts"
    ]
  },
  4402: {
    "establishment_name": "Lazada",
    "establishment_description": "sells a wide variety of products online, including electronics, fashion, home goods, beauty products, and more",
    "transactions": [
      {
        "transaction_id": 1,
        "transaction_text": "LAZADA PH MAKATI PHL",
        "amount": 45.43
      }
    ],
    "category_options": [
      "transport_car_fuel",
      "bills_insurance",
      "bills_service_fees",
      "bills",
      "income_business"
    ]
  },
  12361: {
    "establishment_name": "DoorDash: Flamers",
    "establishment_description": "payment to DoorDash for food delivery",
    "transactions": [
      {
        "transaction_id": 1,
        "transaction_text": "DD DOORDASH FLAMEANDS",
        "amount": 46.35
      }
    ],
    "category_options": [
      "income_side_gig",
      "shopping_clothing",
      "donations_gifts",
      "shopping_kids"
    ]
  },
  4481: {
    "establishment_name": "Empower",
    "establishment_description": "Payment to a financial technology company offering cash advances and budgeting tools.",
    "transactions": [
      {
        "transaction_id": 1,
        "transaction_text": "Empower Inc DES:XXXXX51692 ID:EMPSUBSCR08/24 INDN:MikeKehoe CO ID:XXXXX79144 PPD",
        "amount": 8.00
      },
      {
        "transaction_id": 2,
        "transaction_text": "EMPOWER EMPOWER",
        "amount": 10.00
      },
      {
        "transaction_id": 3,
        "transaction_text": "Empower Inc DES:EmpowerFin ID:35f4a96037ff47 INDN:MikeKehoe CO ID:XXXXX79144 PPD",
        "amount": 8.00
      },
      {
        "transaction_id": 4,
        "transaction_text": "Empower Inc DES:XXXXX95818 ID:EMPSUBSCR03/25 INDN:MikeKehoe CO ID:XXXXX79144 PPD",
        "amount": 10.00
      }
    ],
    "category_options": [
      "education_tuition",
      "education_kids_activities",
      "donations_gifts",
      "shelter_home",
      "bills_service_fees",
      "education",
      "leisure_entertainment",
      "income_business"
    ]
  },
  562: {
    "establishment_name": "Lululemon",
    "establishment_description": "Purchase of athletic apparel and accessories from a retail store.",
    "transactions": [
      {
        "transaction_id": 1,
        "transaction_text": "LULULEMONUS",
        "amount": 75.00
      },
      {
        "transaction_id": 2,
        "transaction_text": "LULULEMON ORLANDO OUTL",
        "amount": 98.21
      },
      {
        "transaction_id": 3,
        "transaction_text": "Lululemon 10160-Ca Burlin,Burlingame,Ca,940100000,840",
        "amount": 40.66
      }
    ],
    "category_options": [
      "food_dining_out",
      "food_groceries",
      "food_delivered_food",
      "donations_gifts",
      "leisure_travel_vacations",
      "food",
      "income_business",
      "education_kids_activities"
    ]
  },
  12374: {
    "establishment_name": "Sam's Restaurant",
    "establishment_description": "sells a variety of dishes, including seafood, pasta, and cocktails",
    "transactions": [
      {
        "transaction_id": 1,
        "transaction_text": "SAMS RESTAURANT",
        "amount": 140.89
      }
    ],
    "category_options": [
      "health_medical_pharmacy",
      "health",
      "donations_gifts",
      "bills_service_fees",
      "health_personal_care"
    ]
  },
  10001: {
    "establishment_name": "asdfqwerlkjhasd",
    "establishment_description": "No description available.",
    "transactions": [
      {
        "transaction_id": 1,
        "transaction_text": "DEBIT CARD PURCHASE - THE COFFEE BEAN 123 MAIN ST",
        "amount": 4.50
      }
    ],
    "category_options": [
      "food_dining_out",
      "shopping_groceries",
      "bills"
    ]
  },
  10002: {
    "establishment_name": "Shopee",
    "establishment_description": "sells a wide variety of products online, including electronics, fashion, home goods, beauty products, and more",
    "transactions": [
      {
        "transaction_id": 1,
        "transaction_text": "SHOPEE PH*PURCHASE",
        "amount": 25.99
      }
    ],
    "category_options": [
      "shopping",
      "bills_service_fees",
      "donations_gifts",
      "shopping_electronics"
    ]
  },
  10003: {
    "establishment_name": "McDonald's",
    "establishment_description": "A fast-food restaurant known for burgers and fries.",
    "transactions": [
      {
        "transaction_id": 1,
        "transaction_text": "MCDONALD'S F2345 ANYTOWN USA",
        "amount": 12.50
      }
    ],
    "category_options": [
      "meals",
      "meals_dining_out",
      "meals_delivered_food"
    ]
  },
  10004: {
    "establishment_name": "Zelle",
    "establishment_description": "peer-to-peer payment service for sending and receiving money between individuals",
    "transactions": [
      {
        "transaction_id": 1,
        "transaction_text": "ZELLE TO JOHN DOE",
        "amount": 50.00
      }
    ],
    "category_options": [
      "transfer",
      "donations_gifts",
      "bills_service_fees",
      "income_business",
      "income_side_gig"
    ]
  },
  3336: {
    "establishment_name": "Ebay",
    "establishment_description": "Purchases from an online marketplace for new and used goods.",
    "transactions": [
      {
        "transaction_id": 13572,
        "transaction_text": "eBay ComGZULXXXD PAYMENTS",
        "amount": -18.53
      }
    ],
    "category_options": [
      "income_side-gig",
      "income_business",
      "income"
    ]
  },
  "1:4739": {
    "establishment_name": "ABCDESFASD2312",
    "establishment_description": "general purchase from an unknown establishment.",
    "transactions": [
      {
        "transaction_id": 37637,
        "transaction_text": "PURCHASE 12/10 ABCDESFASD2312 +XXXXX517908 000",
        "amount": 60
      }
    ],
    "category_options": [
      "meals_dining_out",
      "shelter_home",
      "shelter_upkeep",
      "shopping_clothing",
      "shopping_gadgets",
      "donations_gifts",
      "transfer"
    ]
  },
  "1:8123193": {
    "establishment_name": "Thea's Chicken",
    "establishment_description": "This is a payment to a restaurant for a meal.",
    "transactions": [
      {
        "transaction_id": 10032875,
        "transaction_text": "Thea's Chicken",
        "amount": 15
      }
    ],
    "category_options": [
      "meals_dining_out",
      "meals",
      "meals_delivered food"
    ]
  },
  "SFO_Parking_Example": {
    "establishment_name": "SFO Parking",
    "establishment_description": "sells parking services at San Francisco International Airport, with options for short-term and long-term parking",
    "transactions": [
      {
        "transaction_id": 1,
        "transaction_text": "60776 - Sfo Parkingcentrasan Franciscoca",
        "amount": 4.0
      }
    ],
    "category_options": [
      "transport_car_fuel",
      "transport",
      "bills_service_fees"
    ]
  }
}


def run_test_machos(categorizer: RethinkTransactionCategorization = None):
  """
  Run the test case for Macho's transactions.
  """
  return test_with_inputs([TEST_DATA[8809]], categorizer)


def run_test_marcos_pizza(categorizer: RethinkTransactionCategorization = None):
  """
  Run the test case for Marco's Pizza transactions.
  """
  return test_with_inputs([TEST_DATA[12723]], categorizer)


def run_test_marcos_pizza_alt(categorizer: RethinkTransactionCategorization = None):
  """
  Run the test case for Marcos Pizza (alternative spelling) transactions.
  """
  return test_with_inputs([TEST_DATA[8965]], categorizer)


def run_test_seamless(categorizer: RethinkTransactionCategorization = None):
  """
  Run the test case for Seamless transactions.
  """
  return test_with_inputs([TEST_DATA[11711]], categorizer)


def run_test_cos(categorizer: RethinkTransactionCategorization = None):
  """
  Run the test case for Cos transactions.
  """
  return test_with_inputs([TEST_DATA[4966]], categorizer)


def run_test_cleveland_marriott(categorizer: RethinkTransactionCategorization = None):
  """
  Run the test case for Cleveland Marriott transactions.
  """
  return test_with_inputs([TEST_DATA[12671]], categorizer)


def run_test_marriott(categorizer: RethinkTransactionCategorization = None):
  """
  Run the test case for Marriott transactions.
  """
  return test_with_inputs([TEST_DATA[11741]], categorizer)


def run_test_salvatores(categorizer: RethinkTransactionCategorization = None):
  """
  Run the test case for Salvatore's transactions.
  """
  return test_with_inputs([TEST_DATA[4613]], categorizer)


def run_test_farm_carol_san_pedro(categorizer: RethinkTransactionCategorization = None):
  """
  Run the test case for Farm Carol San Pedro transactions.
  """
  return test_with_inputs([TEST_DATA[12712]], categorizer)


def run_test_zekos_grill(categorizer: RethinkTransactionCategorization = None):
  """
  Run the test case for Zeko's Grill transactions.
  """
  return test_with_inputs([TEST_DATA[12633]], categorizer)


def run_test_multiple_groups(categorizer: RethinkTransactionCategorization = None):
  """
  Run a test case with five different transaction groups in a single input.
  This tests the categorizer's ability to handle multiple transaction groups at once.
  Uses fixtures with the most transactions.
  """
  # Select the 5 groups with the most transactions
  # 12671: 4 transactions, 8809: 2 transactions, 8377: 2 transactions, 11711: 1 transaction, 4966: 1 transaction
  fixture_keys = [12671, 8809, 8377, 11711, 4966]
  return test_with_inputs([TEST_DATA[key] for key in fixture_keys], categorizer)


def run_test_att_bill_payment(categorizer: RethinkTransactionCategorization = None):
  """
  Run the test case for AT&T Bill Payment transactions.
  """
  return test_with_inputs([TEST_DATA[3578]], categorizer)


def run_test_transfer_to_checking(categorizer: RethinkTransactionCategorization = None):
  """
  Run the test case for Transfer to Checking transactions.
  """
  return test_with_inputs([TEST_DATA[8657]], categorizer)


def run_test_fs_metro_news(categorizer: RethinkTransactionCategorization = None):
  """
  Run the test case for F & S Metro News transactions.
  """
  return test_with_inputs([TEST_DATA[8377]], categorizer)


def run_test_placeit(categorizer: RethinkTransactionCategorization = None):
  """
  Run the test case for Placeit transactions.
  """
  return test_with_inputs([TEST_DATA[12586]], categorizer)


def run_test_cleo_express_fee(categorizer: RethinkTransactionCategorization = None):
  """
  Run the test case for Cleo Express Fee transactions.
  """
  return test_with_inputs([TEST_DATA[12583]], categorizer)


def run_test_openai(categorizer: RethinkTransactionCategorization = None):
  """
  Run the test case for OpenAI transactions.
  """
  return test_with_inputs([TEST_DATA[4417]], categorizer)


def run_test_openai_chatgpt(categorizer: RethinkTransactionCategorization = None):
  """
  Run the test case for OpenAI Chatgpt transactions.
  """
  return test_with_inputs([TEST_DATA[12562]], categorizer)


def run_test_sfo_parking(categorizer: RethinkTransactionCategorization = None):
  """
  Run the test case for SFO Parking transactions.
  """
  return test_with_inputs([TEST_DATA[651]], categorizer)


def run_test_equinox(categorizer: RethinkTransactionCategorization = None):
  """
  Run the test case for Equinox transactions.
  """
  return test_with_inputs([TEST_DATA[4485]], categorizer)


def run_test_hand_and_stone(categorizer: RethinkTransactionCategorization = None):
  """
  Run the test case for Hand and Stone transactions.
  """
  return test_with_inputs([TEST_DATA[8611]], categorizer)


def run_test_ace_hardware(categorizer: RethinkTransactionCategorization = None):
  """
  Run the test case for Ace Hardware transactions.
  """
  return test_with_inputs([TEST_DATA[8460]], categorizer)


def run_test_bobs_burgers_brew(categorizer: RethinkTransactionCategorization = None):
  """
  Run the test case for Bob's Burgers & Brew transactions.
  """
  return test_with_inputs([TEST_DATA[6458]], categorizer)


def run_test_aaa(categorizer: RethinkTransactionCategorization = None):
  """
  Run the test case for AAA transactions.
  """
  return test_with_inputs([TEST_DATA[8103]], categorizer)


def run_test_lazada(categorizer: RethinkTransactionCategorization = None):
  """
  Run the test case for Lazada transactions.
  """
  return test_with_inputs([TEST_DATA[4402]], categorizer)


def run_test_doordash_flamers(categorizer: RethinkTransactionCategorization = None):
  """
  Run the test case for DoorDash: Flamers transactions.
  """
  return test_with_inputs([TEST_DATA[12361]], categorizer)


def run_test_empower(categorizer: RethinkTransactionCategorization = None):
  """
  Run the test case for Empower transactions.
  """
  return test_with_inputs([TEST_DATA[4481]], categorizer)


def run_test_lululemon(categorizer: RethinkTransactionCategorization = None):
  """
  Run the test case for Lululemon transactions.
  """
  return test_with_inputs([TEST_DATA[562]], categorizer)


def run_test_sams_restaurant(categorizer: RethinkTransactionCategorization = None):
  """
  Run the test case for Sam's Restaurant transactions.
  """
  return test_with_inputs([TEST_DATA[12374]], categorizer)


def run_test_random_establishment(categorizer: RethinkTransactionCategorization = None):
  """
  Run the test case for a random establishment name.
  """
  return test_with_inputs([TEST_DATA[10001]], categorizer)


def run_test_shopee(categorizer: RethinkTransactionCategorization = None):
  """
  Run the test case for Shopee transactions.
  """
  return test_with_inputs([TEST_DATA[10002]], categorizer)


def run_test_meals_categories(categorizer: RethinkTransactionCategorization = None):
  """
  Run the test case for meals categories.
  """
  return test_with_inputs([TEST_DATA[10003]], categorizer)


def run_test_zelle(categorizer: RethinkTransactionCategorization = None):
  """
  Run the test case for Zelle transactions.
  """
  return test_with_inputs([TEST_DATA[10004]], categorizer)


def run_test_ebay(categorizer: RethinkTransactionCategorization = None):
  """
  Run the test case for Ebay transactions.
  """
  return test_with_inputs([TEST_DATA[3336]], categorizer)


def run_test_unknown_establishment(categorizer: RethinkTransactionCategorization = None):
  """
  Run the test case for an unknown establishment.
  """
  return test_with_inputs([TEST_DATA["1:4739"]], categorizer)


def run_test_theas_chicken(categorizer: RethinkTransactionCategorization = None):
  """
  Run the test case for Thea's Chicken transactions.
  """
  return test_with_inputs([TEST_DATA["1:8123193"]], categorizer)



def _run_batch(batch_num: int, categorizer: RethinkTransactionCategorization):
  """
  Run a specific batch of tests.
  
  Args:
    batch_num: Batch number (1-6) to determine which tests to run
    categorizer: RethinkTransactionCategorization instance
  """
  if batch_num == 1:
    # Batch 1: 5 tests
    print("Test 1: Macho's")
    print("-" * 80)
    run_test_machos(categorizer)
    print("\n")
    
    print("Test 2: Marco's Pizza")
    print("-" * 80)
    run_test_marcos_pizza(categorizer)
    print("\n")
    
    print("Test 3: Marcos Pizza (Alt)")
    print("-" * 80)
    run_test_marcos_pizza_alt(categorizer)
    print("\n")
    
    print("Test 4: Seamless")
    print("-" * 80)
    run_test_seamless(categorizer)
    print("\n")
    
    print("Test 5: Salvatore's")
    print("-" * 80)
    run_test_salvatores(categorizer)
    print("\n")
    
  elif batch_num == 2:
    # Batch 2: 5 tests
    print("Test 1: Cos")
    print("-" * 80)
    run_test_cos(categorizer)
    print("\n")
    
    print("Test 2: Cleveland Marriott")
    print("-" * 80)
    run_test_cleveland_marriott(categorizer)
    print("\n")
    
    print("Test 3: Marriott")
    print("-" * 80)
    run_test_marriott(categorizer)
    print("\n")
    
    print("Test 4: Farm Carol San Pedro")
    print("-" * 80)
    run_test_farm_carol_san_pedro(categorizer)
    print("\n")
    
    print("Test 5: Zeko's Grill")
    print("-" * 80)
    run_test_zekos_grill(categorizer)
    print("\n")
    
  elif batch_num == 3:
    # Batch 3: 5 tests
    print("Test 1: AT&T Bill Payment")
    print("-" * 80)
    run_test_att_bill_payment(categorizer)
    print("\n")
    
    print("Test 2: Transfer to Checking")
    print("-" * 80)
    run_test_transfer_to_checking(categorizer)
    print("\n")
    
    print("Test 3: F & S Metro News")
    print("-" * 80)
    run_test_fs_metro_news(categorizer)
    print("\n")
    
    print("Test 4: Placeit")
    print("-" * 80)
    run_test_placeit(categorizer)
    print("\n")
    
    print("Test 5: AAA")
    print("-" * 80)
    run_test_aaa(categorizer)
    print("\n")
    
  elif batch_num == 4:
    # Batch 4: 5 tests
    print("Test 1: Cleo Express Fee")
    print("-" * 80)
    run_test_cleo_express_fee(categorizer)
    print("\n")
    
    print("Test 2: OpenAI")
    print("-" * 80)
    run_test_openai(categorizer)
    print("\n")
    
    print("Test 3: OpenAI Chatgpt")
    print("-" * 80)
    run_test_openai_chatgpt(categorizer)
    print("\n")
    
    print("Test 4: SFO Parking")
    print("-" * 80)
    run_test_sfo_parking(categorizer)
    print("\n")
    
    print("Test 5: Equinox")
    print("-" * 80)
    run_test_equinox(categorizer)
    print("\n")
    
  elif batch_num == 5:
    # Batch 5: 5 tests
    print("Test 1: Hand and Stone")
    print("-" * 80)
    run_test_hand_and_stone(categorizer)
    print("\n")
    
    print("Test 2: Ace Hardware")
    print("-" * 80)
    run_test_ace_hardware(categorizer)
    print("\n")
    
    print("Test 3: Bob's Burgers & Brew")
    print("-" * 80)
    run_test_bobs_burgers_brew(categorizer)
    print("\n")
    
    print("Test 4: Lazada")
    print("-" * 80)
    run_test_lazada(categorizer)
    print("\n")
    
    print("Test 5: Sam's Restaurant")
    print("-" * 80)
    run_test_sams_restaurant(categorizer)
    print("\n")
    
  elif batch_num == 6:
    # Batch 6: 5 tests
    print("Test 1: DoorDash: Flamers")
    print("-" * 80)
    run_test_doordash_flamers(categorizer)
    print("\n")
    
    print("Test 2: Empower")
    print("-" * 80)
    run_test_empower(categorizer)
    print("\n")
    
    print("Test 3: Lululemon")
    print("-" * 80)
    run_test_lululemon(categorizer)
    print("\n")
    
    print("Test 4: Multiple Groups (5 fixtures)")
    print("-" * 80)
    run_test_multiple_groups(categorizer)
    print("\n")
    
    print("Test 5: AAA")
    print("-" * 80)
    run_test_aaa(categorizer)
    print("\n")
    
  elif batch_num == 7:
    # Batch 7: 3 new tests
    print("Test 1: Random Establishment Name")
    print("-" * 80)
    run_test_random_establishment(categorizer)
    print("\n")
    
    print("Test 2: Shopee")
    print("-" * 80)
    run_test_shopee(categorizer)
    print("\n")
    
    print("Test 3: Meals Categories")
    print("-" * 80)
    run_test_meals_categories(categorizer)
    print("\n")
    
  elif batch_num == 8:
    # Batch 8: 1 test
    print("Test 1: Ebay")
    print("-" * 80)
    run_test_ebay(categorizer)
    print("\n")

  elif batch_num == 9:
    # Batch 9: 3 tests
    print("Test 1: Unknown Establishment")
    print("-" * 80)
    run_test_unknown_establishment(categorizer)
    print("\n")
    
    print("Test 2: Thea's Chicken")
    print("-" * 80)
    run_test_theas_chicken(categorizer)
    print("\n")
    
    print("Test 3: Zelle")
    print("-" * 80)
    run_test_zelle(categorizer)
    print("\n")
    
  else:
    raise ValueError(f"batch must be between 1 and 9, got {batch_num}")


def main(batches=None):
  """
  Main function to test the RethinkTransactionCategorization categorizer
  
  Args:
    batches: List of batch numbers to run (e.g., [1, 2, 3]). If None, defaults to [1].
            To run all batches, pass [1, 2, 3, 4, 5, 6, 7, 8, 9]
  """
  if batches is None:
    batches = [1]
  
  print("Testing RethinkTransactionCategorization\n")
  categorizer = RethinkTransactionCategorization()
  
  for batch_num in batches:
    if batch_num < 1 or batch_num > 9:
      raise ValueError(f"Batch number must be between 1 and 9, got {batch_num}")
    
    print(f"\n{'='*80}")
    print(f"Running Batch {batch_num}")
    print("="*80)
    _run_batch(batch_num, categorizer)
  
  print("\n" + "="*80)
  print("All tests completed!")
  print("="*80)


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Run transaction categorization tests in batches')
  parser.add_argument('--batch', type=str, default='1',
                      help='Batch numbers to run, comma-separated (e.g., "1,2,3" or "1-9" for all). Valid batches: 1-9')
  args = parser.parse_args()
  
  # Parse batch argument
  batches = []
  batch_str = args.batch.strip()
  
  # Handle range notation (e.g., "1-6")
  if '-' in batch_str:
    parts = batch_str.split('-')
    if len(parts) == 2:
      try:
        start = int(parts[0].strip())
        end = int(parts[1].strip())
        batches = list(range(start, end + 1))
      except ValueError:
        raise ValueError(f"Invalid batch range: {batch_str}")
    else:
      raise ValueError(f"Invalid batch range format: {batch_str}")
  else:
    # Handle comma-separated list
    try:
      batches = [int(b.strip()) for b in batch_str.split(',')]
    except ValueError:
      raise ValueError(f"Invalid batch numbers: {batch_str}")
  
  # Validate batch numbers
  for batch_num in batches:
    if batch_num < 1 or batch_num > 9:
      raise ValueError(f"Batch number must be between 1 and 9, got {batch_num}")
  
  main(batches=batches)

