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
        "is_correct": types.Schema(type=types.Type.BOOLEAN, description="True if all constraints are met, False otherwise"),
        "reasoning": types.Schema(type=types.Type.STRING, description="Detailed explanation of why the output is correct or incorrect"),
        "corrections": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "id": types.Schema(type=types.Type.INTEGER),
                    "field": types.Schema(type=types.Type.STRING),
                    "suggested_value": types.Schema(type=types.Type.STRING),
                    "issue": types.Schema(type=types.Type.STRING)
                }
            ),
            description="List of suggested corrections if is_correct is False"
        )
    },
    required=["is_correct", "reasoning"]
)

SYSTEM_PROMPT = r"""#### 1. Goal
Evaluate the output of an account name verbalizer against a set of strict constraints and logic rules. Determine if the output is correct and provide detailed reasoning and corrections if necessary.

#### 2. Rules to Check
- **purpose_name**: Must be a concise and descriptive deduction of the core purpose (e.g., Checking, Savings, Credit Card, Brokerage, Auto Loan).
- **crisp_name**: 
  - MUST include the `purpose_name` word for word.
  - MUST NOT include the bank name unless it's a specific sub-brand (e.g., "360", "Venture").
  - If product name == purpose_name, `crisp_name` should be identical to `purpose_name` (no "Standard").
  - MUST NOT start with the bank name if the bank name is in `bank_added_name`.
- **bank_added_name**: 
  - MUST be exactly `[Bank Name] [purpose_name]`.
  - MUST NOT include any product names (e.g., "360", "Venture", "Sapphire", "Secure").
  - MUST include the `purpose_name` word for word.
- **General Constraints**:
  - The word "Account" is STRICTLY FORBIDDEN in any part of the output.
  - Masked account numbers (e.g., "...1234") must be ignored/removed.
  - Original `id` must be maintained.
  - Generic names like "Secure Banking" or "First Banking" should be deduced as "Checking".

#### 3. Output Format
Return a JSON object with `is_correct`, `reasoning`, and an optional `corrections` array.
"""

class AccountNameVerbalizerChecker:
    """Evaluates the output of the AccountNameVerbalizerOptimizer."""

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
        self.thinking_budget = 0

    def check_output(self, original_input: list, verbalizer_output: list) -> dict:
        """
        Check the verbalizer output against the original input and constraints.
        """
        combined_input = {
            "original_input": original_input,
            "verbalizer_output": verbalizer_output
        }
        user_input = json.dumps(combined_input, indent=2)
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_input)])]
        
        generate_content_config = types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_output_tokens,
            safety_settings=self.safety_settings,
            system_instruction=[types.Part.from_text(text=self.system_prompt)],
            thinking_config=types.ThinkingConfig(
                thinking_budget=self.thinking_budget,
                include_thoughts=True
            ),
            response_schema=self.output_schema,
            response_mime_type="application/json"
        )

        # Generate response using streaming to extract thoughts
        output_text = ""
        thought_summary = ""
        
        for chunk in self.client.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=generate_content_config,
        ):
            # Extract text content (non-thought parts)
            if chunk.text is not None:
                output_text += chunk.text
            
            # Extract thought summary from chunk
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
        
        if thought_summary:
            print(f"{'='*80}")
            print("THOUGHT SUMMARY:")
            print(thought_summary.strip())
            print("="*80)
            
        if not output_text:
            raise ValueError("Empty response from model.")
            
        try:
            return json.loads(output_text)
        except json.JSONDecodeError:
            text = output_text.strip()
            if text.startswith("```json"):
                text = text[7:-3].strip()
            elif text.startswith("```"):
                text = text[3:-3].strip()
            return json.loads(text)

def test_checker():
    checker = AccountNameVerbalizerChecker()
    
    test_cases = [
        # Case 1: Correct Output
        {
            "input": [
                {"id": 9512, "account_name": "Capital One 360 Checking", "bank_name": None, "long_account_name": "Capital One 360 Checking"}
            ],
            "output": [
                {"id": 9512, "purpose_name": "Checking", "crisp_name": "360 Checking", "bank_added_name": "Capital One Checking"}
            ]
        },
        # Case 2: Incorrect Output (Includes "Account", includes product in bank_added_name)
        {
            "input": [
                {"id": 104, "account_name": "Brokerage Account", "bank_name": "Charles Schwab", "long_account_name": "Schwab Individual Brokerage Account"}
            ],
            "output": [
                {"id": 104, "purpose_name": "Brokerage Account", "crisp_name": "Schwab Brokerage", "bank_added_name": "Charles Schwab Individual Brokerage"}
            ]
        },
        # Case 3: Incorrect Output (Missing purpose_name in crisp_name, starts with bank name)
        {
            "input": [
                {"id": 103, "account_name": "Chase Secure Banking", "bank_name": "Chase", "long_account_name": "Chase Secure Banking"}
            ],
            "output": [
                {"id": 103, "purpose_name": "Checking", "crisp_name": "Chase Secure", "bank_added_name": "Chase Checking"}
            ]
        }
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        try:
            result = checker.check_output(case["input"], case["output"])
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_checker()
