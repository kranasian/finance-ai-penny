from google import genai
from google.genai import types
import os
import sys
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the tools
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are Penny, a financial analyst. Deliver concise, well-rationalized, and easy-to-follow plans.

## Process
1.  **Find the Goal**: Pinpoint the user's primary financial aim.
2.  **List Key Facts**: Take income, expenses, savings, capacity from **Input Information**, and research on market data necessary to answer the **User request**.
3.  **Analyze Savings Opportunities**: If the user's goal requires increased savings, you must analyze all spending data from the "Key Facts" and provide a prioritized, actionable list of recommendations for reducing spending or increasing income. These recommendations must be integrated directly into the strategy.
4.  **Create a Strategy**: Design a complete, self-contained strategy of no more than 3 steps. There must be no open-ended tasks for the user. You must provide the concrete insights an expert would. Specify exact financial vehicles and researched targets. User should have liquidity (an emergency fund) at any point of time.

## Example Output Format (~120 words)

**Summary:** [A simple 1-2 sentence summary of the plan.]

**Key Facts:**
- [Main result] (Input: "[quote relevant part]")
- [Monthly/timeline] (Input: "[quote]")
- [Feasibility/growth] (Input: "[quote]")

**Strategy:**

1. [Step from strategy with calculation and Input reference]
2. [Step from strategy with calculation and Input reference]
3. [Step from strategy with calculation and Input reference]

**Risks:** [A single, brief bullet point on a potential risk or consideration.]"""

class ResearchStrategizeOptimizer:
  """Handles all Gemini API interactions for financial research and strategizing"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for financial strategizing"""
    # API Configuration
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file or environment.")
    self.client = genai.Client(api_key=api_key)
    
    # Model Configuration
    self.model_name = model_name
    
    # Generation Configuration
    self.gen_config = {
      "top_k": 40,
      "top_p": 0.95,
      "temperature": 0.6,
      "thinking_budget": 4096,
      "max_output_tokens": 4096
    }
    
    # Safety Settings
    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
    ]
    
    # System Prompt
    self.system_prompt = SYSTEM_PROMPT

  def generate_response(self, user_request: str, input_info: str) -> str:
    """
    Generate a response using Gemini API for financial strategizing.
    
    Args:
      user_request: The user's request or goal
      input_info: The input information (financial data, etc.)
      
    Returns:
      Generated strategy as a string
    """
    # Create request text
    request_text = types.Part.from_text(text=f"""**User request**: {user_request}

**Input Information**:
{input_info}

output:""")
    
    # Create content and configuration
    contents = [types.Content(role="user", parts=[request_text])]
    
    generate_content_config = types.GenerateContentConfig(
      temperature=self.gen_config["temperature"],
      top_p=self.gen_config["top_p"],
      top_k=self.gen_config["top_k"],
      max_output_tokens=self.gen_config["max_output_tokens"],
      safety_settings=self.safety_settings,
      system_instruction=[types.Part.from_text(text=self.system_prompt)],
      thinking_config=types.ThinkingConfig(thinking_budget=self.gen_config["thinking_budget"]),
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
    
    return output_text

# Test cases list
TEST_CASES = [
  {
    "name": "save_for_house_down_payment",
    "user_request": "I want to save $10,000 for a down payment on a house in 12 months. What's the best plan?",
    "input_info": "Monthly Income: $5,000. Monthly Expenses: $3,500 (Rent: $1,500, Utilities: $200, Groceries: $400, Dining Out: $600, Subscriptions: $100, Misc: $700). Current Savings: $2,000. Emergency Fund: $5,000 (separate from savings)."
  },
  {
    "name": "vacation_to_japan_feasibility",
    "user_request": "Is it feasible for me to take a 2-week vacation to Japan next year? Research the costs and tell me if I can afford it.",
    "input_info": "Monthly Income: $4,500. Monthly Expenses: $3,200. Current Savings: $3,000. Emergency Fund: $6,000. Goal: Save $4,000 for Japan in 10 months."
  },
  {
    "name": "reduce_debt_and_save",
    "user_request": "I have $3,000 in credit card debt at 20% APR. I also want to save $2,000 for a car repair. What should I prioritize?",
    "input_info": "Monthly Income: $3,800. Monthly Expenses: $3,000. Current Savings: $500. Emergency Fund: $1,000."
  }
]

def _run_test_with_logging(user_request: str, input_info: str, optimizer: ResearchStrategizeOptimizer = None):
  """
  Internal helper function that runs a test with consistent logging.
  """
  if optimizer is None:
    optimizer = ResearchStrategizeOptimizer()
  
  print("=" * 80)
  print("LLM INPUT:")
  print("=" * 80)
  print(f"**User request**: {user_request}")
  print(f"\n**Input Information**:\n{input_info}")
  print("=" * 80)
  print()
  
  result = optimizer.generate_response(user_request, input_info)
  
  print("=" * 80)
  print("LLM OUTPUT:")
  print("=" * 80)
  print(result)
  print("=" * 80)
  print()
  
  return result

def run_test(test_name_or_index_or_dict, optimizer: ResearchStrategizeOptimizer = None):
  """
  Run a single test by name, index, or by passing test data directly.
  """
  if isinstance(test_name_or_index_or_dict, dict):
    test_data = test_name_or_index_or_dict
    test_name = test_data.get("name", "custom_test")
  else:
    # Find test case by name or index
    test_case = None
    if isinstance(test_name_or_index_or_dict, int):
      if 0 <= test_name_or_index_or_dict < len(TEST_CASES):
        test_case = TEST_CASES[test_name_or_index_or_dict]
    else:
      for tc in TEST_CASES:
        if tc["name"] == test_name_or_index_or_dict:
          test_case = tc
          break
    
    if test_case is None:
      print(f"Test case '{test_name_or_index_or_dict}' not found.")
      return None
    test_data = test_case
    test_name = test_data["name"]

  print(f"\n{'='*80}")
  print(f"Running test: {test_name}")
  print(f"{'='*80}\n")
  
  return _run_test_with_logging(
    test_data["user_request"],
    test_data["input_info"],
    optimizer
  )

def main():
  """Main function to test the research strategize optimizer"""
  # Run the first test case
  run_test(0)

if __name__ == "__main__":
  main()
