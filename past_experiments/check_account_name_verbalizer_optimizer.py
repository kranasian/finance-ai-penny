from google import genai
from google.genai import types
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = r"""#### 1. Goal
Evaluate verbalizer JSON against `ORIGINAL_INPUT`. Output `{"good_copy": boolean, "info_correct": boolean, "eval_text": string}`.

Expected processing: **identify** purpose (if any), product name, and bank from inputs → **build** `purpose_name` (only if customized), `crisp_name` (purpose + product), `bank_added_name` (purpose + bank) → **Title Case** and concise cleanup. Each verbalizer row object should list keys in order **`id` → `purpose_name` (if present) → `crisp_name` → `bank_added_name`**.

Inputs are **not standardized**: purpose, product, and bank signals may appear in **any** of `account_name`, `long_account_name`, or `bank_name`. Judge from the **row as a whole**.

#### 2. purpose_name (optional key — step identify + build)
- No `custom_account_name` in input. Judge from the **combined** `account_name`, `long_account_name`, and `bank_name`.
- Key MUST be **absent** when the row, taken as a whole, is plain FI product naming (uncustomized). Present only when a user-specific use/goal/nickname applies.
- `purpose_name` should be the **shortest faithful purpose phrase** supported by the row text; do not invent purpose details.
- MUST NOT be a generic product type alone (reject solo 'Checking', 'Savings', 'Credit Card', 'Mortgage', 'Brokerage', 'Loan', etc.).

#### 3. crisp_name (purpose + product)
- **Uncustomized:** product only (e.g. `Total Checking`); minimal tokens; sub-brands '360', 'Venture' when in-product.
- **Customized:** every `purpose_name` word verbatim once + minimal product tokens — e.g. `Gabby's Total Checking`. Shortest clear string.

#### 4. bank_added_name (purpose + bank)
- **Uncustomized:** institution short name ONLY (infer if needed)—no product descriptors.
- **Customized:** purpose + bank, tight compounds (`Gabby's Citi`, `Citi Business` over `Business at Citi`). **Fail** `Citi Gabby's` / `Chase Gabby's` for possessive purposes.

#### 5. Cleanup (Title Case / general)
- Proper title structure; conventional brand casing (`360`, `Sapphire Preferred`, etc.).
- Forbidden substring `Account` in any value. **Do not** treat Checking, Savings, Rewards, Loan, Mortgage, Brokerage as forbidden by that rule.
- **Concision:** as short as possible without losing required content.
- `id` unchanged. Ignore masked tails when judging.

#### 6. eval_text
- `""` if both flags true.
- Else single string, `\n` bullets, single quotes inside, start with FIX: `- ID <id>: ... (Expected: ...)`

#### 7. Decision policy
Pass unless a **clear** violation of identify → build → cleanup. **Pass** concise purpose+product / purpose+bank strings with Title Case. If you pass, `eval_text` must be exactly `""`. Never set `good_copy` false after writing that the output is correct.
"""

CHECK_RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "good_copy": types.Schema(type=types.Type.BOOLEAN),
        "info_correct": types.Schema(type=types.Type.BOOLEAN),
        "eval_text": types.Schema(type=types.Type.STRING),
    },
    required=["good_copy", "info_correct", "eval_text"],
)


class AccountNameVerbalizerChecker:
    """Evaluates the output of the AccountNameVerbalizerOptimizer."""

    def __init__(self, model_name="gemini-flash-lite-latest"):
        """Initialize the Gemini client with API configuration."""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        self.client = genai.Client(api_key=api_key)
        
        self.model_name = model_name
        self.thinking_budget = 0
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
        self.response_schema = CHECK_RESPONSE_SCHEMA

    def check_output(self, original_input: list, verbalizer_output: list) -> dict:
        """
        Check the verbalizer output against the original input and constraints.
        """
        request_text_str = f"""<ORIGINAL_INPUT>

{json.dumps(original_input, indent=2)}

</ORIGINAL_INPUT>

<VERBALIZER_OUTPUT>

{json.dumps(verbalizer_output, indent=2)}

</VERBALIZER_OUTPUT>

Output:"""
        
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=request_text_str)])]
        
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
            response_schema=self.response_schema,
            response_mime_type="application/json",
        )

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

        if not output_text:
            raise ValueError("Empty response from model.")

        try:
            json_start = output_text.find("{")
            json_end = output_text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                return json.loads(output_text[json_start:json_end])
            return json.loads(output_text.strip())
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}\nResponse text: {output_text}") from e

def test_checker():
    checker = AccountNameVerbalizerChecker()
    
    test_cases = [
        # Case 1: Correct Output
        {
            "name": "Correct Output",
            "input": [
                {"id": 9512, "account_name": "Capital One 360 Checking", "bank_name": None, "long_account_name": "Capital One 360 Checking"}
            ],
            "output": [
                {"id": 9512, "purpose_name": "Checking", "crisp_name": "360 Checking", "bank_added_name": "Capital One Checking"}
            ]
        },
        # Case 2: Incorrect Output (Includes "Account", includes product in bank_added_name)
        {
            "name": "Forbidden Word & Formatting Error",
            "input": [
                {"id": 104, "account_name": "Brokerage Account", "bank_name": "Charles Schwab", "long_account_name": "Schwab Individual Brokerage Account"}
            ],
            "output": [
                {"id": 104, "purpose_name": "Brokerage Account", "crisp_name": "Schwab Brokerage", "bank_added_name": "Charles Schwab Individual Brokerage"}
            ]
        },
        # Case 3: Incorrect Output (Missing purpose_name in crisp_name, starts with bank name)
        {
            "name": "Missing Purpose in Crisp & Prefix Error",
            "input": [
                {"id": 103, "account_name": "Chase Secure Banking", "bank_name": "Chase", "long_account_name": "Chase Secure Banking"}
            ],
            "output": [
                {"id": 103, "purpose_name": "Checking", "crisp_name": "Chase Secure", "bank_added_name": "Chase Checking"}
            ]
        }
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}: {case['name']} ---")
        try:
            result = checker.check_output(case["input"], case["output"])
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_checker()
