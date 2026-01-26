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
- `Sidegig`: inflow for work performed as a supplemental, contract, or freelance worker

### Likelihood Options
- `LIKELY`: high probability of being in the category
- `IMPOSSIBLE`: low to no probability of being in the category
- `UNLIKELY`: 50-50 between being and not being in the category

### Processing Rules
1. **Recurring Check**: Based on `description`, determine if a transaction is recurring (e.g., paid/received weekly, monthly, yearly).
   - **If NOT Recurring**: Tag `is_bills`, `is_salary`, and `is_sidegig` as `IMPOSSIBLE`.
2. **Directionality Check**:
   - **Outflows (Payments/Expenses)**: If the transaction is an outflow (e.g., "monthly subscription", "monthly rent", "monthly membership fee", "monthly cell phone bill"), `is_salary` MUST be `IMPOSSIBLE`. However, it may be a side-gig expense (categorized under `is_sidegig` if it supports freelance work) or a bill.
   - **Inflows (Income)**: If the transaction is an inflow (e.g., "salary payment", "payment for logo design gig"), it could be `is_salary` or `is_sidegig`.
3. **Categorization**: Determine the likelihood of the transaction to be categorized as a bill, salary, and/or sidegig following the options above."""


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
            "description": "water utility company that collects water service fees"
        },
        {
            "id": 1657,
            "name": "Lyft",
            "description": "ride-sharing service where customers pay for transportation"
        },
        {
            "id": 12010,
            "name": "Pret A Manger",
            "description": "sells sandwiches, salads, and other food items"
        },
        {
            "id": 953,
            "name": "Travelers Insurance",
            "description": "sells various types of insurance, including auto, home, renters, and business insurance"
        },
        {
            "id": 22891,
            "name": "Visa Provisioning Service",
            "description": "a debit card authorization for a credit from a Visa provisioning service"
        }
    ]
    return _run_test_with_logging(transactions, classifier)


def test_batch_1(classifier: SummarizedNamesRecurringLikelihood = None):
    """Batch 1: Outflows and Inflows mix"""
    transactions = [
        {"id": 1, "name": "Netflix", "description": "monthly streaming subscription service"},
        {"id": 2, "name": "Starbucks", "description": "coffee shop for daily caffeine fix"},
        {"id": 3, "name": "Company Payroll", "description": "bi-weekly salary payment from employer"},
        {"id": 4, "name": "Upwork", "description": "freelance platform payment for software project"},
        {"id": 5, "name": "Landlord", "description": "monthly rent payment for apartment"}
    ]
    return _run_test_with_logging(transactions, classifier)


def test_batch_2(classifier: SummarizedNamesRecurringLikelihood = None):
    """Batch 2: Ambiguous and Non-recurring"""
    transactions = [
        {"id": 6, "name": "Home Depot", "description": "one-time purchase of garden tools"},
        {"id": 7, "name": "State Farm", "description": "monthly auto insurance premium"},
        {"id": 8, "name": "Uber", "description": "occasional ride-sharing for weekend outings"},
        {"id": 9, "name": "Local Gym", "description": "monthly membership fee for fitness center"},
        {"id": 10, "name": "Amazon", "description": "various online shopping purchases"}
    ]
    return _run_test_with_logging(transactions, classifier)


def test_batch_3(classifier: SummarizedNamesRecurringLikelihood = None):
    """Batch 3: Potential Side-gig expenses vs Salary"""
    transactions = [
        {"id": 11, "name": "Adobe Creative Cloud", "description": "monthly subscription for design software used in freelance work"},
        {"id": 12, "name": "H&M", "description": "clothing store purchase for personal wardrobe"},
        {"id": 13, "name": "T-Mobile", "description": "monthly cell phone bill"},
        {"id": 14, "name": "Fiverr", "description": "payment for logo design gig"},
        {"id": 15, "name": "Chevron", "description": "gas station for car fuel"}
    ]
    return _run_test_with_logging(transactions, classifier)


def test_batch_4(classifier: SummarizedNamesRecurringLikelihood = None):
    """Batch 4: Bills vs One-offs"""
    transactions = [
        {"id": 16, "name": "ConEd", "description": "monthly electricity and gas utility bill"},
        {"id": 17, "name": "Best Buy", "description": "purchase of a new laptop"},
        {"id": 18, "name": "Zelle Transfer", "description": "received money from a friend for dinner"},
        {"id": 19, "name": "Blue Apron", "description": "weekly meal kit subscription service"},
        {"id": 20, "name": "Apple Services", "description": "monthly iCloud storage fee"}
    ]
    return _run_test_with_logging(transactions, classifier)


def main():
    """
    Main function to test the classifier
    """
    print("Testing SummarizedNamesRecurringLikelihood\n")
    classifier = SummarizedNamesRecurringLikelihood()

    batches = [test_batch_1, test_batch_2, test_batch_3, test_batch_4]
    
    for i, batch in enumerate(batches):
        print(f"Running Batch {i+1}")
        print("-" * 80)
        batch(classifier)
        print("\n")

    print("All tests completed!")


if __name__ == "__main__":
    main()
