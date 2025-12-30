from google import genai
from google.genai import types
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """#### 1. Role & Goal
You are a financial transaction categorization expert. Your goal is to deeply analyze and categorize transactions based on their transaction text, establishment information, and amount, selecting the most appropriate category from the provided options.

#### 2. Core Task
Your task is to analyze each transaction within a group and determine:
- The most appropriate category from the provided `category_options`
- Your confidence level (high, medium, or low)
- Clear reasoning for your categorization decision

#### 3. Input Data
You will be provided with a JSON array containing transaction groups. Each group contains:
- `group_id`: Unique identifier for the transaction group
- `establishment_name`: The name of the merchant or establishment
- `establishment_description`: Description of what the establishment provides
- `transactions`: Array of transactions with:
  - `transaction_id`: Unique identifier for the transaction
  - `transaction_text`: The raw transaction text from the bank statement
  - `amount`: The transaction amount
- `category_options`: Array of possible categories to choose from

#### 4. Output Requirements
- **Format:** Return a JSON array where each transaction gets its own object
- **Structure:** Each output object must contain:
  - `group_id`: The group ID from the input
  - `transaction_id`: The transaction ID from the input
  - `category`: One of the provided category options (must match exactly)
  - `confidence`: One of "high", "medium", or "low"
  - `reasoning`: A clear, concise explanation of why this category was chosen

#### 5. Critical Constraints
- **Use Only Provided Categories:** The `category` field must exactly match one of the values in `category_options`
- **Analyze Transaction Text:** Pay close attention to the transaction text, as it often contains the most specific information
- **Consider Amount:** Use the transaction amount as context (e.g., small amounts might indicate different purposes than large ones)
- **Consider Establishment:** Use the establishment name and description to understand the context
- **Be Consistent:** Transactions from the same group with similar characteristics should typically receive the same category
- **Output Valid JSON:** Your response must be valid JSON that can be parsed

#### 6. Analysis Guidelines
- Look for keywords in the transaction text that indicate the purpose
- Consider the establishment type and what services/products it typically provides
- Use amount as a signal (e.g., $500+ at a vet might be a procedure, $20 might be supplies)
- If uncertain, choose "medium" or "low" confidence and explain the uncertainty

input: [
  {
    "group_id": 2112,
    "establishment_name": "Best Friends Veterinary",
    "establishment_description": "A purchase for veterinary services or products.",
    "transactions": [
      {
        "transaction_id": 328202,
        "transaction_text": "MOBILE PURCHASE 1222 BEST FRIENDS VETE NESCONSET NY XXXXX1344XXXXXXXXXX3323",
        "amount": 509.22
      },
      {
        "transaction_id": 272828,
        "transaction_text": "MOBILE PURCHASE 1031 BEST FRIENDS VETE NESCONSET NY XXXXX3276XXXXXXXXXX3712",
        "amount": 12.20
      }
    ],
    "category_options": [
      "shopping_pets",
      "education_kids_activities",
      "bills_service_fees",
      "donations_gifts",
      "bills",
      "shelter"
    ]
  }
]
output: [
  {
    "group_id": 2112,
    "transaction_id": 328202,
    "category": "shopping_pets",
    "confidence": "high",
    "reasoning": "The transaction is at a veterinary clinic (Best Friends Veterinary) with a substantial amount of $509.22, which is typical for veterinary services like checkups, procedures, or treatments for pets. The establishment description confirms this is for veterinary services or products."
  },
  {
    "group_id": 2112,
    "transaction_id": 272828,
    "category": "shopping_pets",
    "confidence": "high",
    "reasoning": "This is also a transaction at Best Friends Veterinary, but with a smaller amount of $12.20, which likely represents pet supplies, medications, or a minor service. Still clearly pet-related spending at a veterinary establishment."
  }
]

input: [
  {
    "group_id": 3456,
    "establishment_name": "Starbucks Coffee",
    "establishment_description": "Coffee shop and cafe.",
    "transactions": [
      {
        "transaction_id": 123456,
        "transaction_text": "POS DEBIT STARBUCKS STORE #1234 SEATTLE WA",
        "amount": 5.75
      }
    ],
    "category_options": [
      "food_dining_out",
      "shopping_groceries",
      "bills",
      "transportation"
    ]
  }
]
output: [
  {
    "group_id": 3456,
    "transaction_id": 123456,
    "category": "food_dining_out",
    "confidence": "high",
    "reasoning": "Starbucks is a coffee shop/cafe, and the transaction amount of $5.75 is typical for a coffee purchase. This represents dining out/consuming food and beverages away from home."
  }
]
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
      thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
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
        # Remove ```json or ``` at start and end
        lines = output_text_clean.split("\n")
        if lines[0].startswith("```"):
          lines = lines[1:]
        if lines[-1].strip() == "```":
          lines = lines[:-1]
        output_text_clean = "\n".join(lines)
      
      result = json.loads(output_text_clean)
      if not isinstance(result, list):
        raise ValueError(f"Expected a JSON array, but got: {type(result)}")
      
      return result
    except json.JSONDecodeError as e:
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


def run_test_veterinary_transactions(categorizer: RethinkTransactionCategorization = None):
  """
  Run the test case for veterinary transactions.
  """
  return test_with_inputs([
    {
      "group_id": 2112,
      "establishment_name": "Best Friends Veterinary",
      "establishment_description": "A purchase for veterinary services or products.",
      "transactions": [
        {
          "transaction_id": 328202,
          "transaction_text": "MOBILE PURCHASE 1222 BEST FRIENDS VETE NESCONSET NY XXXXX1344XXXXXXXXXX3323",
          "amount": 509.22
        },
        {
          "transaction_id": 272828,
          "transaction_text": "MOBILE PURCHASE 1031 BEST FRIENDS VETE NESCONSET NY XXXXX3276XXXXXXXXXX3712",
          "amount": 12.20
        }
      ],
      "category_options": [
        "shopping_pets",
        "education_kids_activities",
        "bills_service_fees",
        "donations_gifts",
        "bills",
        "shelter"
      ]
    }
  ], categorizer)


def main():
  """Main function to test the RethinkTransactionCategorization categorizer"""
  categorizer = RethinkTransactionCategorization()
  result = run_test_veterinary_transactions(categorizer)
  
  print(f"\n{'='*80}")
  print("FINAL RESULT:")
  print(json.dumps(result, indent=2))
  print("="*80)


if __name__ == "__main__":
  main()

