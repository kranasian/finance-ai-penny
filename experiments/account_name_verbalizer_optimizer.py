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
            "id": types.Schema(type=types.Type.INTEGER, description="Same as the id in the input"),
            "purpose_name": types.Schema(type=types.Type.STRING, description="Name with the purpose details only"),
            "crisp_name": types.Schema(type=types.Type.STRING, description="purpose_name + product name"),
            "bank_added_name": types.Schema(type=types.Type.STRING, description="purpose_name + bank"),
        },
        required=["id", "purpose_name", "crisp_name", "bank_added_name"]
    )
)

SYSTEM_PROMPT = r"""#### 1. Goal
Take a list of bank account details and simplify/rename each account based on its details. The output should provide three variations of the name: purpose-focused, a crisp product-focused name, and a bank-branded name.

#### 2. Naming Logic
- **purpose_name**: Deduce the core purpose of the account based on its name and details. It should be concise and descriptive.
  - Examples: Checking, Savings, Credit Card, Mortgage, Brokerage, Auto Loan, Student Loan, Home Equity, Line of Credit, Money Market.
  - If the purpose is not explicitly mentioned (e.g., "Chase Secure Banking"), deduce it from the product context (e.g., "Checking").
- **crisp_name**: Combine a specific product identifier or sub-brand with the purpose (e.g., "360 Checking", "Venture Rewards Credit Card", "Secure Checking"). Avoid repeating the bank name here if it's already in the bank_added_name.
- **bank_added_name**: Combine the bank name with the purpose (e.g., "Capital One Checking", "Chase Credit Card", "Chase Checking").

#### 3. Rules
- Always maintain the original `id`.
- If `bank_name` is null, try to infer it from `account_name` or `long_account_name`.
- For Credit Cards, ensure "Credit Card" is the `purpose_name` if not explicitly stated but implied by product names (e.g., "Venture", "Sapphire", "Platinum").
- If the account name is generic like "Secure Banking" or "First Banking", deduce the purpose (usually "Checking" for these Chase products).
- Keep names professional and easy to read.
- If multiple accounts have the same purpose, use the product details to differentiate them in `crisp_name`.

#### 4. Output Format
Return a JSON array of objects, each containing `id`, `purpose_name`, `crisp_name`, and `bank_added_name`.
"""

class AccountNameVerbalizerOptimizer:
    """Handles Gemini API interactions for simplifying and renaming bank account names."""

    def __init__(self, model_name="gemini-flash-lite-latest"):
        """Initialize the Gemini client with API configuration."""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        self.client = genai.Client(api_key=api_key)
        
        self.model_name = model_name
        self.temperature = 0.0
        self.top_p = 0.95
        self.max_output_tokens = 2048
        
        self.safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
        ]
        
        self.system_prompt = SYSTEM_PROMPT
        self.output_schema = SCHEMA

    def generate_response(self, accounts: list) -> list:
        """
        Generate simplified names for a list of bank accounts.
        """
        user_input = json.dumps(accounts, indent=2)
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_input)])]
        
        generate_content_config = types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_output_tokens,
            safety_settings=self.safety_settings,
            system_instruction=[types.Part.from_text(text=self.system_prompt)],
            response_schema=self.output_schema,
            response_mime_type="application/json"
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=generate_content_config,
        )
        
        if not response.text:
            raise ValueError("Empty response from model.")
            
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:-3].strip()
            elif text.startswith("```"):
                text = text[3:-3].strip()
            return json.loads(text)

def test_optimizer():
    optimizer = AccountNameVerbalizerOptimizer()
    
    test_cases = [
        # Case 1: Provided example (Homogeneous bank, different products)
        [
            {"id": 9512, "account_name": "Capital One 360 Checking", "bank_name": None, "long_account_name": "Capital One 360 Checking"},
            {"id": 9514, "account_name": "Capital One Savings", "bank_name": None, "long_account_name": "Capital One 360 Savings"},
            {"id": 9513, "account_name": "Capital One Venture Rewards", "bank_name": "Capital One", "long_account_name": "Capital One Venture Rewards"}
        ],
        # Case 2: Heterogeneous banks and account types
        [
            {"id": 101, "account_name": "CHASE SAPPHIRE PREFERRED", "bank_name": "JPMorgan Chase", "long_account_name": "CHASE SAPPHIRE PREFERRED VISA SIGNATURE"},
            {"id": 102, "account_name": "High Yield Savings", "bank_name": "Marcus by Goldman Sachs", "long_account_name": "Marcus High Yield Savings Account"},
            {"id": 103, "account_name": "Chase Secure Banking", "bank_name": "Chase", "long_account_name": "Chase Secure Banking"},
            {"id": 104, "account_name": "Brokerage Account", "bank_name": "Charles Schwab", "long_account_name": "Schwab Individual Brokerage Account"}
        ],
        # Case 3: Minimal detail / Generic names
        [
            {"id": 201, "account_name": "Chase First Banking", "bank_name": "Chase", "long_account_name": "Chase First Banking"},
            {"id": 202, "account_name": "Savings", "bank_name": "Bank of America", "long_account_name": "Bank of America Advantage Savings"},
            {"id": 203, "account_name": "Credit Card", "bank_name": "Citibank", "long_account_name": "Citi Double Cash Card"}
        ],
        # Case 4: Loans and specialized accounts
        [
            {"id": 301, "account_name": "Mortgage", "bank_name": "Rocket Mortgage", "long_account_name": "Rocket Mortgage Conventional Fixed"},
            {"id": 302, "account_name": "Auto Loan", "bank_name": "Ally Bank", "long_account_name": "Ally Auto Finance Loan"},
            {"id": 303, "account_name": "Student Loan", "bank_name": "Navient", "long_account_name": "Navient Federal Student Loan"}
        ]
    ]
    
    for i, accounts in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        print("Input:")
        print(json.dumps(accounts, indent=2))
        try:
            result = optimizer.generate_response(accounts)
            print("\nOutput:")
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_optimizer()
