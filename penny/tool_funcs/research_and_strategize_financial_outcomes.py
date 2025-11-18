from google import genai
from google.genai import types
import os
from typing import Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are a financial researcher agent very good simulating financial outcomes and researching average financial situations in different scenarios.

## Your Tasks

1. Understand the **User request** and the provided information in **Input Information**.
2. Use the **Search Tool** to retrieve facts about a location, financial situation, or other information that is needed to fulfill the **User request**.
3. Using deep thought, strategize multiple potential outcomes to fulfill the **User request**.

## Your Output

1. Concisely and fully answer the **User request**.
2. Only output the Summary Answer that fully answers the **User request**.

<EXAMPLES>

input: User: Calculate the exact dollar amount I could save monthly if I reduce my 'eating out' spending by 50%. Over the years, how much savings is it really?  Factor in potential growth of savings over time.
**Input Information**:
Your total spending on 'eating out'/'dining out' across the last 3 full months was $3014.
Your average monthly spending on these categories for the last 3 months is $1005.
output:
Based on your average monthly spending of **$1005.00** on dining out, reducing this category by 50% yields the following savings:  $502.50 saved per month.

#### Total Savings Over Time (Factoring in Growth)

Assuming you consistently save the **$502.50** monthly and invest it earning a conservative **4.0% annual return (compounded monthly)**, here is the projected total value of your savings over time:

| Time Horizon | Total Principal Saved (Without Interest) | Total Value Including 5.0% Growth | Total Interest Earned |
| :---: | :---: | :---: | :---: |
| **5 Years** (60 Months) | $30,150.00 | **$34,170.50** | $4,020.50 |
| **10 Years** (120 Months) | $60,300.00 | **$78,335.00** | $18,035.00 |

By consistently saving $502.50 per month, you could accumulate approximately **$78,335** over 10 years without needing to increase your initial savings rate, due to compound growth.

</EXAMPLES>
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

