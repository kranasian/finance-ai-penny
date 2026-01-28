from google import genai
from google.genai import types
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()


# Output Schema - array of result objects
SCHEMA = types.Schema(
    type=types.Type.ARRAY,
    items=types.Schema(
        type=types.Type.OBJECT,
        required=[
            "id",
            "is_bills",
            "is_salary",
            "is_sidegig"
        ],
        properties={
            "id": types.Schema(
                type=types.Type.INTEGER,
                description=""
            ),
            "is_bills": types.Schema(
                type=types.Type.STRING,
                description=""
            ),
            "is_salary": types.Schema(
                type=types.Type.STRING,
                description=""
            ),
            "is_sidegig": types.Schema(
                type=types.Type.STRING,
                description=""
            )
        }
    )
)


SYSTEM_PROMPT = """## Persona

You are a personal finance advisor whose role includes identifying possible bills, salaries, and sidegigs to predict a user's cash flow and notify them if the inflow/outflow has not been recorded as expected.

## Objective

Identify recurring transactions, then the likelihood for them to be bills, salaries, and sidegigs.

## Input

- `id`: unique numbers matched to each establishment
- `name`: establishment the transaction is with (to drop in output)
- `description` (to drop in output)
   - **For transactions with businesses:** what the business is, including the specific products and services being purchased/sold
   - **For person to person transfers, interbank transfers, and credit card payments:** what the transaction is and its likely purpose

## Output

- `id`: `id` from input
- `is_bills`: likelihood of transaction with `name` to be a bill
- `is_salary`: likelihood of transaction with `name` to be a salary
- `is_sidegig`: likelihood of transaction with `name` to be a sidegig

### Categories
- `Bill`: outflow that is a consistent payment for a loan, goods, or services
- `Salary`: inflow for work performed as a permanent employee
- `Sidegig`: Recurring income from freelance work, contract jobs, or other non-employer sources. It is **always an inflow** of money.

### Likelihood Options
- `LIKELY`: high probability of being in the category
- `IMPOSSIBLE`: low to no probability of being in the category
- `UNLIKELY`: 50-50 between being and not being in the category

### Processing Rules
1. **Recurring Check (CRITICAL FIRST STEP)**: Based on the `name` and `description`, determine if a transaction is likely to be recurring.
    - **Consider the nature of the business:** A subscription service is almost guaranteed to be recurring (`LIKELY`). A bank may have recurring fees, but not all bank transactions are recurring, so it's less certain (`UNLIKELY`). A grocery store is rarely a recurring bill (`IMPOSSIBLE`).
    - Subscriptions (e.g., streaming services, software, meal kits), utilities, rent, insurance, or payroll are typically recurring.
    - One-time purchases from retailers, ride-sharing, or P2P transfers are typically not.
   - **If the transaction does not appear to be recurring, you MUST tag `is_bills`, `is_salary`, and `is_sidegig` as `IMPOSSIBLE`.**
2. **Directionality Inference**: From the `description`, infer if the transaction is an **inflow** (income) or an **outflow** (expense).
   - **If the direction is ambiguous**: Assume the transaction type that is more common for the establishment. For example, "Netflix" is typically an outflow (payment), while a transaction from a known payroll company is an inflow.
3. **Categorization Logic (only if recurring)**:
   - **Outflows (Payments/Expenses)**: If the transaction is an outflow, it is most likely a `bill`. `is_salary` and `is_sidegig` **MUST** be `IMPOSSIBLE`.
   - **Inflows (Income)**: If the transaction is an inflow, it can be a `salary` or `sidegig`. `is_bills` MUST be `IMPOSSIBLE`. Differentiate between a stable salary and more variable side gig income.
4. **Final Review**: Briefly review your classifications for common sense. For example, a university (`NYU`) is not a monthly bill, but it does have recurring tuition payments, so `is_bills` should be `UNLIKELY`, not `IMPOSSIBLE`."""


class SummarizedNamesRecurringLikelihood:
    """Handles all Gemini API interactions for identifying recurring likelihood of transactions"""

    def __init__(self, model_name="gemini-flash-lite-latest"):
        """Initialize the Gemini agent with API configuration"""
        # API Configuration
        self.client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

        # Model Configuration
        self.thinking_budget = 0
        self.model_name = model_name

        # Generation Configuration Constants
        self.temperature = 0.2
        self.top_p = 0.95
        self.top_k = 40
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

    def identify_recurring_likelihood(self, transactions: list) -> list:
        """
        Identify recurring likelihood for transactions using Gemini API.

        Args:
          transactions: A list of dictionaries, each containing:
            - id: Unique identifier
            - name: Establishment name
            - description: Transaction description

        Returns:
          A list of dictionaries, each containing:
            - id: The same id from input
            - is_bills: LIKELY, IMPOSSIBLE, or UNLIKELY
            - is_salary: LIKELY, IMPOSSIBLE, or UNLIKELY
            - is_sidegig: LIKELY, IMPOSSIBLE, or UNLIKELY
        """
        import json

        # Convert input to JSON string
        input_json = json.dumps(transactions, indent=2)

        # Create request text
        request_text = types.Part.from_text(text=f"""input:
{input_json}

output:""")

        # Create content and configuration
        contents = [types.Content(role="user", parts=[request_text])]

        generate_content_config = types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_output_tokens=self.max_output_tokens,
            safety_settings=self.safety_settings,
            system_instruction=[types.Part.from_text(text=self.system_prompt)],
            thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
            response_mime_type="application/json",
            response_schema=SCHEMA,
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

            result = json.loads(output_text)
            return result
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}\nResponse was: {output_text}")

    def get_available_models(self):
        """
        Get list of available Gemini models.

        Returns:
          List of available model names
        """
        try:
            models = genai.list_models()
            return [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        except Exception as e:
            raise Exception(f"Failed to get models: {str(e)}")


def _run_test_with_logging(transactions: list, classifier: SummarizedNamesRecurringLikelihood = None):
    """
    Internal helper function that runs a test with consistent logging.

    Args:
      transactions: List of transactions to analyze
      classifier: Optional SummarizedNamesRecurringLikelihood instance. If None, creates a new one.

    Returns:
      The results as a list
    """
    import json

    if classifier is None:
        classifier = SummarizedNamesRecurringLikelihood()

    # Print the input
    print("=" * 80)
    print("INPUT:")
    print("=" * 80)
    print(json.dumps(transactions, indent=2))
    print("=" * 80)
    print()

    try:
        result = classifier.identify_recurring_likelihood(transactions)

        # Print the output
        print("=" * 80)
        print("OUTPUT:")
        print("=" * 80)
        print(json.dumps(result, indent=2))
        print("=" * 80)
        print()

        return result
    except Exception as e:
        print(f"**Error**: {str(e)}")
        import traceback
        print(traceback.format_exc())
        print("=" * 80)
        return None


def test_provided_examples(classifier: SummarizedNamesRecurringLikelihood = None):
    """
    Test with the examples provided in the prompt
    """
    transactions = [
        {
            "id": 8630,
            "name": "Municipal Water and Sewer",
            "description": "Public utility for water and wastewater management."
        },
        {
            "id": 1657,
            "name": "Lyft",
            "description": "A technology company that facilitates peer-to-peer transportation."
        },
        {
            "id": 12010,
            "name": "Pret A Manger",
            "description": "A chain of sandwich shops."
        },
        {
            "id": 953,
            "name": "Travelers Insurance",
            "description": "An insurance company providing a range of coverage options."
        },
        {
            "id": 22891,
            "name": "Visa Provisioning Service",
            "description": "Service for managing and provisioning Visa card credentials for digital wallets."
        }
    ]
    return _run_test_with_logging(transactions, classifier)


def test_batch_1(classifier: SummarizedNamesRecurringLikelihood = None):
    """Batch 1: Outflows and Inflows mix"""
    transactions = [
        {"id": 1, "name": "Netflix", "description": "An online streaming service for movies and TV shows."},
        {"id": 2, "name": "Starbucks", "description": "A global chain of coffeehouses."},
        {"id": 3, "name": "Direct Deposit", "description": "An electronic transfer of payment into a checking or savings account."},
        {"id": 4, "name": "Stripe", "description": "An online payment processing platform for businesses."},
        {"id": 5, "name": "John Smith", "description": "A person-to-person transfer."}
    ]
    return _run_test_with_logging(transactions, classifier)


def test_batch_2(classifier: SummarizedNamesRecurringLikelihood = None):
    """Batch 2: Ambiguous and Non-recurring"""
    transactions = [
        {"id": 6, "name": "Home Depot", "description": "A retail company that sells home improvement and construction products."},
        {"id": 7, "name": "State Farm", "description": "An insurance and financial services company."},
        {"id": 8, "name": "Uber", "description": "A technology company that offers ride-hailing and food delivery services."},
        {"id": 9, "name": "ClassPass", "description": "A subscription service providing access to a network of fitness studios and gyms."},
        {"id": 10, "name": "Amazon", "description": "An e-commerce and cloud computing company."}
    ]
    return _run_test_with_logging(transactions, classifier)


def test_batch_3(classifier: SummarizedNamesRecurringLikelihood = None):
    """Batch 3: Potential Side-gig expenses vs Salary"""
    transactions = [
        {"id": 11, "name": "Adobe Creative Cloud", "description": "A suite of software and services for creative professionals."},
        {"id": 12, "name": "H&M", "description": "A multinational retail company for fashion."},
        {"id": 13, "name": "Verizon", "description": "A telecommunications company offering wireless and wireline services."},
        {"id": 14, "name": "Patreon", "description": "A membership platform for creators to run a subscription content service."},
        {"id": 15, "name": "Chevron", "description": "An energy corporation involved in petroleum and natural gas."}
    ]
    return _run_test_with_logging(transactions, classifier)


def test_batch_4(classifier: SummarizedNamesRecurringLikelihood = None):
    """Batch 4: Bills vs One-offs"""
    transactions = [
        {"id": 16, "name": "ConEd", "description": "An energy company that provides electric, gas, and steam service."},
        {"id": 17, "name": "Affirm", "description": "A financial technology company offering installment loans for consumers at the point of sale."},
        {"id": 18, "name": "Zelle", "description": "A digital payments network."},
        {"id": 19, "name": "Blue Apron", "description": "A meal kit delivery service."},
        {"id": 20, "name": "Apple Services", "description": "Digital services from Apple, such as App Store, iCloud, and Apple Music."}
    ]
    return _run_test_with_logging(transactions, classifier)


def test_batch_5(classifier: SummarizedNamesRecurringLikelihood = None):
    """Batch 5: Financial, Government, and Healthcare"""
    transactions = [
        {"id": 21, "name": "Chase Bank", "description": "Financial institution offering banking and investment services."},
        {"id": 22, "name": "IRS", "description": "United States Internal Revenue Service, responsible for collecting federal taxes."},
        {"id": 23, "name": "CVS Pharmacy", "description": "Retail pharmacy and healthcare company."},
        {"id": 24, "name": "Kaiser Permanente", "description": "Integrated managed care consortium, combining health care services with health plan coverage."},
        {"id": 25, "name": "Department of Motor Vehicles", "description": "Government agency that administers vehicle registration and driver licensing."}
    ]
    return _run_test_with_logging(transactions, classifier)


def test_batch_6(classifier: SummarizedNamesRecurringLikelihood = None):
    """Batch 6: Education, Travel, and Entertainment"""
    transactions = [
        {"id": 26, "name": "NYU", "description": "A private research university in New York City."},
        {"id": 27, "name": "Delta Airlines", "description": "A major American airline."},
        {"id": 28, "name": "Marriott Hotels", "description": "A multinational hospitality company."},
        {"id": 29, "name": "Ticketmaster", "description": "A ticket sales and distribution company."},
        {"id": 30, "name": "American Red Cross", "description": "A humanitarian organization providing emergency assistance and disaster relief."}
    ]
    return _run_test_with_logging(transactions, classifier)




def main():
    """
    Main function to test the classifier
    """
    print("Testing SummarizedNamesRecurringLikelihood\n")
    classifier = SummarizedNamesRecurringLikelihood()

    batches = [test_batch_1, test_batch_2, test_batch_3, test_batch_4, test_batch_5, test_batch_6]
    
    for i, batch in enumerate(batches):
        print(f"Running Batch {i+1}")
        print("-" * 80)
        batch(classifier)
        print("\n")

    print("All tests completed!")



if __name__ == "__main__":
    main()
