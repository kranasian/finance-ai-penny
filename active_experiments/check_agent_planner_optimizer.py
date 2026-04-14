from google import genai
from google.genai import types
import os
import json
from typing import Optional, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = r"""#### 1. Goal
Evaluate the output of a Planner Agent that generates a Python function `execute_plan` to address user requests. Determine if the tool selection and the request strings are correct.

#### 2. Core Task
- Evaluate `REVIEW_NEEDED` against `EVAL_INPUT`.
- Use `PAST_REVIEW_OUTCOMES` as reference for consistency.
- Output a JSON object: `{"good_copy": boolean, "info_correct": boolean, "eval_text": string}`.

#### 3. Rules to Check
- **info_correct (Tool Selection)**:
  - `app_usage_info`: Used ONLY for navigation, UI, and general app questions (no user data).
  - `lookup_user_accounts_transactions_income_and_spending_patterns`: Used for ANY user data (balances, transactions, spending, income). MUST be called first if data is needed.
  - `research_and_strategize_financial_outcomes`: Used for complex analysis, long-term planning, or "what-if" scenarios. DO NOT use if `lookup` alone suffices.
  - `create_budget_or_goal`: Used ONLY when the user explicitly asks to *create* or *set up* a budget/goal.
  - `update_transaction_category_or_create_category_rules`: Used for recategorization or rule creation.
  - **Efficiency**: No unnecessary chaining (e.g., `lookup` + `research` when `lookup` alone works for "how's my accounts doing?").

- **good_copy (Request Accuracy)**:
  - The natural language strings passed into tool parameters (e.g., `lookup_request`, `strategize_request`) must be accurate and comprehensive.
  - They MUST incorporate relevant context from the `Previous Conversation`.
  - They MUST NOT be redundant or generic if specific details are available.

#### 4. Output Schema
- `good_copy`: `true` if the request strings are accurate and context-aware. `false` otherwise.
- `info_correct`: `true` if the correct tools are selected and chained efficiently. `false` otherwise.
- `eval_text`: Required if `good_copy` or `info_correct` is `false`.
  - **Must be an empty string ("")** if both are `true`.
  - **Must be a single string** (use `\n` for newlines).
  - Concise bullet points explaining the violation.
  - Start with the required fix.
"""

class CheckAgentPlannerOptimizer:
    """Evaluates the output of the Planner Agent."""

    def __init__(self, model_name: str = "gemini-flash-lite-latest"):
        """Initialize the Gemini client with API configuration."""
        api_key = os.getenv("GEMINI_API_KEY")
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

    def evaluate(self, eval_input: str, review_needed: str, past_review_outcomes: str = "") -> Dict:
        """
        Evaluate the planner output against the input and rules.
        """
        request_text_str = f"""### EVAL_INPUT
{eval_input}

### REVIEW_NEEDED
{review_needed}

### PAST_REVIEW_OUTCOMES
{past_review_outcomes}

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
                include_thoughts=True
            ),
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
            # Extract JSON object from the response
            json_start = output_text.find('{')
            json_end = output_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = output_text[json_start:json_end]
                return json.loads(json_str)
            else:
                return json.loads(output_text.strip())
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}\nResponse text: {output_text}")

def test_checker():
    evaluator = CheckAgentPlannerOptimizer()
    
    test_cases = [
        {
            "name": "Correct Output (Lookup Only)",
            "eval_input": "**Last User Request**: how's my accounts doing?\n**Previous Conversation**: User: I spent a lot in Europe.\nAssistant: Yes, $2000 total.",
            "review_needed": "def execute_plan():\n    return lookup_user_accounts_transactions_income_and_spending_patterns(\"How are my accounts doing?\")",
            "past_review_outcomes": ""
        },
        {
            "name": "Incorrect Tool (Navigation using Lookup)",
            "eval_input": "**Last User Request**: Where can I see my net worth?\n**Previous Conversation**: None.",
            "review_needed": "def execute_plan():\n    return lookup_user_accounts_transactions_income_and_spending_patterns(\"Where is my net worth in the app?\")",
            "past_review_outcomes": ""
        },
        {
            "name": "Unnecessary Chaining",
            "eval_input": "**Last User Request**: how's my accounts doing?",
            "review_needed": "def execute_plan():\n    success, lookup_result = lookup_user_accounts_transactions_income_and_spending_patterns(\"How are my accounts doing?\")\n    return research_and_strategize_financial_outcomes(\"Analyze my accounts.\", input_info=lookup_result)",
            "past_review_outcomes": ""
        }
    ]
    
    for case in test_cases:
        print(f"\n--- Running Test: {case['name']} ---")
        try:
            result = evaluator.evaluate(case["eval_input"], case["review_needed"], case["past_review_outcomes"])
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_checker()
