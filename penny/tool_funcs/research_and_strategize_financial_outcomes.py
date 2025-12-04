from google import genai
from google.genai import types
import os
from typing import Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """YOU MUST FOLLOW THESE INSTRUCTIONS EXACTLY.

1.  **FILL IN THE BLANKS:** Your only job is to fill in the bracketed sections `[...]` in the template below.
2.  **DO NOT DEVIATE FROM THE TEMPLATE:** Do not change the headers, the structure, or the wording.
3.  **PRIORITIZE DEBT:** If the user has high liabilities, the first option MUST be about paying down debt.

---

**HERE IS THE TEMPLATE YOU MUST USE:**

**Summary:** "Here are two great paths to help you save for [User's Goal], while also building a strong financial foundation!"

**Penny's Quick Research:** "A typical down payment for a condo is about [Calculated Down Payment], which is a fantastic goal to work towards!"

**Here are two paths to your condo investment:**

**Option 1: The Debt-First Foundation**
*   **What to do:** "Before we start saving for a condo, let's focus on your [Total Liabilities]. A great first step would be to allocate [Percentage of Savings, e.g., 70%] of your [Monthly Savings Amount] savings to your highest-interest debt."
*   **Timeline:** "You could be debt-free in about [Calculated Debt-Free Timeline], which would be an amazing achievement!"
*   **Why it's a great path:** "Clearing your debts first will make it much easier and safer to invest in a condo down the road."

**Option 2: The Balanced Growth Plan**
*   **What to do:** "Let's balance paying down your [Total Liabilities] and saving for your condo. You could put [Percentage of Savings, e.g., 50%] of your savings toward debt, and the other [Percentage of Savings, e.g., 50%] into a high-yield savings account for your down payment."
*   **Timeline:** "This balanced approach could see you ready for a down payment in about [Calculated Balanced Timeline], while also reducing your debt."
*   **Why it's a great path:** "This path helps you make steady progress on both your big goals at the same time!"

**Friendly reminder:** "Don't forget to set aside a little money for an emergency fund! Having 3-6 months of living expenses saved up is a great way to protect your financial future."

---

**HERE IS A BAD EXAMPLE (DO NOT DO THIS):**

**Summary:** Consistently save the $2359 monthly allocation...
**Key Calculations:** ...
**Plan:** ...
**Risks:** ...
"""


class ResearchAndStrategizeOptimizer:
  """Handles all Gemini API interactions for financial planning and optimization"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for financial planning"""
    # API Configuration
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file or environment.")
    self.client = genai.Client(api_key=api_key)
    
    # Model Configuration
    self.thinking_budget = 4096
    self.model_name = model_name
    
    # Generation Configuration Constants
    self.temperature = 0.6
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

  
  def generate_response(self, user_request: str, input_information: str) -> str:
    """
    Generate a response using Gemini API for financial planning.
    
    Args:
      user_request: The last user request as a string
      input_information: The previous conversation as a string
      
    Returns:
      Generated code as a string
    """
    # Create request text with Last User Request and Previous Conversation
    request_text = types.Part.from_text(text=f"""User: {user_request}
**Input Information**:
{input_information}
output:""")
    
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
    
    return output_text


def research_and_strategize_financial_outcomes(
    strategize_request: str, 
    input_info: str = None
) -> Tuple[bool, str]:
    """
    Research and strategize financial outcomes based on a natural language query.
    
    This function uses the ResearchAndStrategizeOptimizer to research financial situations,
    simulate financial outcomes, and strategize multiple potential scenarios.
    
    Args:
        strategize_request: What needs to be thought out, planned or strategized. 
                           It can contain research information like "average dining out for a couple in Chicago, Illinois" 
                           and factoring in information from input_info.
        input_info: Optional input from another skill function
        
    Returns:
        Tuple[bool, str]: (success, output_info)
    """
    try:
        optimizer = ResearchAndStrategizeOptimizer()
        result = optimizer.generate_response(strategize_request, input_info or "")
        return True, result
    except Exception as e:
        error_msg = f"Error in research and strategize: {str(e)}"
        return False, error_msg
