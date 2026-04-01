from google import genai
from google.genai import types
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "reply": types.Schema(
            type=types.Type.STRING,
            description="The comprehensive yet concise information about Hey Penny app usage. Focus on accuracy and completeness of the answer."),
    },
    required=["reply"]
)

SYSTEM_PROMPT = r"""#### 1. Goal
Provide accurate, comprehensive, yet concise information about the Hey Penny app's features, navigation, and categorization. This output will be processed by another model to match the app's tone, so focus purely on the factual content of the answer.

#### 2. App Structure & Navigation
The app is organized into four main tabs: Home, Account, Goals, and Insights.

**1. Home Tab**
- **Cash Total and Credit Debit Total (1.1):** Overall balances.
  - **Charts (1.1.1, 1.1.2, 1.1.3):** Year to Year, Month to Month, and Week to Week line charts per account.
  - **Recent Transactions (1.1.1.4):** View and edit transactions.
    - Editable fields: Name (1.1.1.4.1), Status (1.1.1.4.5 - Pending, Duplicate, Category Confirmation, Confirmed).
  - **Categorization (1.1.1.5):** View/change transaction category.
    - "Split It Up" button (1.1.1.5.3) to divide a transaction between multiple categories using a slider.
- **Income vs. Spending (1.2):** Bar graphs for Expected vs Actual (Year, Month, or Week).
- **Penny Chat (1.4):** Direct access to chat.
- **Account Balances (1.5):** Detailed view of all linked accounts.
- **Categorized Transactions (1.6):** Search bar for all transactions.
- **Review Needed (1.7):** Transactions requiring user review.

**2. Account Tab**
- **Net Worth Tab (2.1):** Net worth trends (Year/Month/Week) and Credit vs. Savings breakdown.
- **Credit Tab (2.2):** Credit account details.
- **Savings Tab (2.3):** Savings account details.

**3. Goals Tab**
- **Add Goal (3.1):** Create new goals (budgets, regular savings, or savings goals by date).
- **Active Goals (3.2):** Progress tracking (actual vs target) with progress bars.
- **Goal Settings (3.2.1):** Edit title, amount, or end goal.
- **Past Goals (3.3):** Completed or ended goals.

**4. Insights Tab**
- **Feedback/Actions:** Love It, Report Issue, Hide This.

#### 3. Categories & Definitions
- **Food (Meals):** Dining Out (restaurants, cafes, fast-food, bakeries), Delivered Food (apps like DoorDash/UberEats), Groceries (supermarkets, convenience stores, pantry staples).
- **Leisure:** Entertainment (concerts, movies, streaming, hobbies, **Alcohol, Cannabis, Cigarettes**), Travel & Vacations (hotels, airfare, tours).
- **Bills:** Connectivity (phone, internet), Insurance (life, business), Taxes (income, state, business), Service Fees (professional fees, laundry).
- **Shelter:** Home (rent, mortgage, property tax, HOA), Utilities (water, electric, gas), Upkeep (repairs, furniture, cleaning).
- **Education:** Kids Activities (sports, camps), Tuition (school fees, academic supplies).
- **Shopping:** Clothing, Gadgets (tech, fitness trackers), Kids (toys, diapers), Pets (food, vet, grooming).
- **Transport:** Public Transit (trains, buses, Uber/Lyft), Car & Fuel (gas, EV charging, maintenance, parking, tolls).
- **Health:** Medical & Pharmacy (doctor, meds, therapy), Gym & Wellness (fitness, spa), Personal Care (hair, nails, cosmetics).
- **Others:** Donations & Gifts, Miscellaneous.
- **Income:** Salary, Side-Gig, Business, Interest.
- **Transfers:** Internal movement between accounts (not spending).

#### 4. Features & Constraints
- **Manual Transactions:** Users **cannot** manually add transactions. Accounts must be linked.
- **Custom Categories:** Users **cannot** create their own categories. They must use the provided list.
- **Splitting:** Use "Split It Up" to divide one transaction into 2+ categories.

#### 5. Output Requirements
Return JSON:
- `reply`: Factual, comprehensive, and concise answer. No conversational filler or tone-specific language needed.
"""

class PennyAppUsageInfoOptimizer:
    """Handles Gemini API interactions for answering Hey Penny app usage questions."""

    def __init__(self, model_name="gemini-flash-lite-latest"):
        """Initialize the Gemini agent with API configuration."""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        self.client = genai.Client(api_key=api_key)
        
        self.model_name = model_name
        self.temperature = 0.0 # Zero temperature for maximum factual consistency
        self.top_p = 0.95
        self.max_output_tokens = 1024
        
        self.safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
        ]
        
        self.system_prompt = SYSTEM_PROMPT
        self.output_schema = SCHEMA

    def generate_response(self, user_input: str) -> dict:
        """
        Generate a factual response to a user's question about app usage.
        """
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_input)])]
        
        generate_content_config = types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_output_tokens,
            safety_settings=self.safety_settings,
            system_instruction=[types.Part.from_text(text=self.system_prompt)],
            response_schema=self.output_schema,
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
    optimizer = PennyAppUsageInfoOptimizer()
    
    test_cases = [
        "What category options do I have for food?",
        "Where can I see my net worth?",
        "How do I manually add a transaction?",
        "Can I create a custom category for my 'Hobby' expenses?",
        "How do I split a transaction between Groceries and Household?",
        "Where can I track my progress on my 'New Car' savings goal?",
        "Would alcohol be categorized by Penny as entertainment or dining out?"
    ]
    
    for case in test_cases:
        print(f"\nInput: {case}")
        result = optimizer.generate_response(case)
        print(f"Reply: {result['reply']}")

if __name__ == "__main__":
    test_optimizer()
