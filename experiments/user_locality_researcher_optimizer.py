from google import genai
from google.genai import types
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define the output schema for structured JSON responses
# Iteration 2: Fully flattened Category-first schema for maximum consistency
SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "rent_monthly_usd": types.Schema(
            type=types.Type.OBJECT,
            description="Monthly rent/housing ranges in USD",
            properties={
                "single_household": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.NUMBER), description="[min, max] for 1 person (Studio or 1-Bedroom)"),
                "couple_household": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.NUMBER), description="[min, max] for 2 people (1-Bedroom or 2-Bedroom)"),
                "family_of_four_household": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.NUMBER), description="[min, max] for 4 people (3-Bedroom or House)")
            },
            required=["single_household", "couple_household", "family_of_four_household"]
        ),
        "groceries_weekly_usd": types.Schema(
            type=types.Type.OBJECT,
            description="Weekly grocery spending ranges in USD",
            properties={
                "single_household": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.NUMBER), description="[min, max] for 1 person"),
                "couple_household": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.NUMBER), description="[min, max] for 2 people"),
                "family_of_four_household": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.NUMBER), description="[min, max] for 4 people")
            },
            required=["single_household", "couple_household", "family_of_four_household"]
        ),
        "utilities_monthly_usd": types.Schema(
            type=types.Type.OBJECT,
            description="Monthly utilities (electricity, water, heating, internet) ranges in USD",
            properties={
                "single_household": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.NUMBER), description="[min, max] for 1 person"),
                "couple_household": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.NUMBER), description="[min, max] for 2 people"),
                "family_of_four_household": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.NUMBER), description="[min, max] for 4 people")
            },
            required=["single_household", "couple_household", "family_of_four_household"]
        ),
        "fast_food_per_meal_usd": types.Schema(
            type=types.Type.OBJECT,
            description="Fast food meal cost per instance/visit ranges in USD",
            properties={
                "single_household": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.NUMBER), description="[min, max] cost for 1 person's meal"),
                "couple_household": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.NUMBER), description="[min, max] cost for 2 people's meals"),
                "family_of_four_household": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.NUMBER), description="[min, max] cost for 4 people's meals")
            },
            required=["single_household", "couple_household", "family_of_four_household"]
        ),
        "restaurant_per_meal_usd": types.Schema(
            type=types.Type.OBJECT,
            description="Sit-down restaurant meal cost per instance/visit ranges in USD",
            properties={
                "single_household": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.NUMBER), description="[min, max] cost for 1 person's meal"),
                "couple_household": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.NUMBER), description="[min, max] cost for 2 people's meals"),
                "family_of_four_household": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.NUMBER), description="[min, max] cost for 4 people's meals")
            },
            required=["single_household", "couple_household", "family_of_four_household"]
        )
    },
    required=[
        "rent_monthly_usd", 
        "groceries_weekly_usd", 
        "utilities_monthly_usd", 
        "fast_food_per_meal_usd", 
        "restaurant_per_meal_usd"
    ]
)

SYSTEM_PROMPT = """You are a locality cost researcher.
Task: Provide 2024-2025 cost estimates for a given `locality`.

Protocol:
1.  **Research**: Locate recent pricing for the following categories:
    *   **Rent**: Studio/1BR (Single), 1BR/2BR (Couple), 3BR+ (Family).
    *   **Groceries**: Weekly basket for 1, 2, and 4 people.
    *   **Utilities**: Monthly basics + Internet for 1, 2, and 4 people.
    *   **Dining**: Per-visit cost (Fast Food & Restaurant) for the *total* group size (1, 2, 4).
2.  **Estimate**: Calculate realistic [min, max] USD ranges based on findings.
3.  **Output**: Return strictly valid JSON matching the schema.

Data Rules:
*   Year: 2024-2025 data only.
*   Scope: Specific to `locality` (avoid national averages).
*   Format: JSON only, no markdown, no chatter.
"""

class UserLocalityResearcherOptimizer:
  """Handles all Gemini API interactions for researching locality spending patterns"""
  
  def __init__(self, model_name="gemini-3-flash-preview"):
    """Initialize the Gemini agent with API configuration for locality research"""
    # API Configuration
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file or environment.")
    self.client = genai.Client(api_key=api_key)
    
    # Model Configuration
    self.thinking_budget = 4096
    self.model_name = model_name
    
    # Generation Configuration Constants
    self.temperature = 0.3  # Lower temperature for more consistent, factual outputs
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
    
    # Output Schema
    self.output_schema = SCHEMA

  
  def generate_response(self, locality: str) -> dict:
    """
    Generate spending estimates for a locality using Gemini API.
    The model will use its search capabilities to research the locality.
    
    Args:
      locality: The locality string (e.g., "Millbrae, California, US")
      
    Returns:
      Dictionary containing spending estimates in the required format
    """
    # Create request text
    input_json = {
      "locality": locality
    }
    
    # Display input in easy-to-read format
    print(f"\n{'='*80}")
    print("INPUT:")
    print(f"  Locality: {locality}")
    print("="*80)
    
    request_text_str = f"""input: {json.dumps(input_json, indent=2)}
output: """
    
    request_text = types.Part.from_text(text=request_text_str)
    
    # Create content and configuration
    contents = [types.Content(role="user", parts=[request_text])]
    
    # Enable Google Search tool for research capabilities
    # Based on Gemini API docs, create Tool with google_search
    # Note: The exact API may vary - this handles potential variations
    tools_list = []

    config_kwargs = {
      "temperature": self.temperature,
      "top_p": self.top_p,
      "max_output_tokens": self.max_output_tokens,
      "safety_settings": self.safety_settings,
      "system_instruction": [types.Part.from_text(text=self.system_prompt)],
      "thinking_config": types.ThinkingConfig(
        thinking_budget=self.thinking_budget,
        include_thoughts=True
      ),
      "response_schema": self.output_schema,
      "response_mime_type": "application/json"
    }
    tool_instance = types.Tool(google_search={})
    tools_list = [tool_instance]
    # Only add tools if we successfully created the tool
    if tools_list:
      config_kwargs["tools"] = tools_list
    
    generate_content_config = types.GenerateContentConfig(**config_kwargs)

    # Generate response
    output_text = ""
    thought_summary = ""
    web_search_queries = []
    
    # According to Gemini API docs: iterate through chunks and check part.thought boolean
    for chunk in self.client.models.generate_content_stream(
      model=self.model_name,
      contents=contents,
      config=generate_content_config,
    ):
      # Extract text content (non-thought parts)
      if chunk.text is not None:
        output_text += chunk.text
      
      # Extract thought summary and web search queries from chunk
      if hasattr(chunk, 'candidates') and chunk.candidates:
        for candidate in chunk.candidates:
          # Extract web search queries from groundingMetadata
          if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
            if hasattr(candidate.grounding_metadata, 'web_search_queries') and candidate.grounding_metadata.web_search_queries:
              for query in candidate.grounding_metadata.web_search_queries:
                if query and query not in web_search_queries:
                  web_search_queries.append(query)
          # Also try alternative attribute names (snake_case vs camelCase)
          if hasattr(candidate, 'groundingMetadata') and candidate.groundingMetadata:
            if hasattr(candidate.groundingMetadata, 'webSearchQueries') and candidate.groundingMetadata.webSearchQueries:
              for query in candidate.groundingMetadata.webSearchQueries:
                if query and query not in web_search_queries:
                  web_search_queries.append(query)
            elif hasattr(candidate.groundingMetadata, 'web_search_queries') and candidate.groundingMetadata.web_search_queries:
              for query in candidate.groundingMetadata.web_search_queries:
                if query and query not in web_search_queries:
                  web_search_queries.append(query)
          
          # Extract thought summary from parts (per Gemini API docs)
          # Check part.thought boolean to identify thought parts
          if hasattr(candidate, 'content') and candidate.content:
            if hasattr(candidate.content, 'parts') and candidate.content.parts:
              for part in candidate.content.parts:
                # Check if this part is a thought summary (per documentation)
                if hasattr(part, 'thought') and part.thought:
                  if hasattr(part, 'text') and part.text:
                    # Accumulate thought summary text (for streaming, it may come in chunks)
                    if thought_summary:
                      thought_summary += part.text
                    else:
                      thought_summary = part.text
    
    # Check if response is empty
    if not output_text or not output_text.strip():
      raise ValueError(f"Empty response from model. Check API key and model availability.")
    
    if thought_summary:
      print(f"{'='*80}")
      print("THOUGHT SUMMARY:")
      print(thought_summary.strip())
      print("="*80)
    
    if web_search_queries:
      print(f"{'='*80}")
      print("WEB SEARCH QUERIES:")
      for i, query in enumerate(web_search_queries, 1):
        print(f"  {i}. {query}")
      print("="*80)
      
    print(f"{'='*80}")
    print("RESPONSE OUTPUT:")
    print(output_text.strip())
    print("="*80)
    
    # Parse JSON response
    try:
      # Try to extract JSON from the response (in case there's extra text)
      output_text_clean = output_text.strip()
      # Remove markdown code blocks if present
      if output_text_clean.startswith("```json"):
        output_text_clean = output_text_clean[7:]
      if output_text_clean.startswith("```"):
        output_text_clean = output_text_clean[3:]
      if output_text_clean.endswith("```"):
        output_text_clean = output_text_clean[:-3]
      output_text_clean = output_text_clean.strip()
      
      result = json.loads(output_text_clean)
      return result
    except json.JSONDecodeError as e:
      raise ValueError(f"Failed to parse JSON response: {e}\nResponse was: {output_text}")


def test_with_inputs(locality: str, researcher: UserLocalityResearcherOptimizer = None):
  """
  Convenient method to test the locality researcher with custom inputs.
  
  Args:
    locality: The locality string to research
    researcher: Optional UserLocalityResearcherOptimizer instance. If None, creates a new one.
    
  Returns:
    Dictionary containing spending estimates
  """
  if researcher is None:
    researcher = UserLocalityResearcherOptimizer()
  
  return researcher.generate_response(locality)


def run_test_millbrae(researcher: UserLocalityResearcherOptimizer = None):
  """
  Run the test case for Millbrae, California.
  """
  return test_with_inputs("Millbrae, California, US", researcher)


def run_test_austin(researcher: UserLocalityResearcherOptimizer = None):
  """
  Run the test case for Austin, Texas.
  """
  return test_with_inputs("Austin, Texas, US", researcher)


def run_test_salt_lake_city(researcher: UserLocalityResearcherOptimizer = None):
  """
  Run the test case for Salt Lake City, Utah.
  """
  return test_with_inputs("Salt Lake City, Utah, US", researcher)


def run_test_memphis(researcher: UserLocalityResearcherOptimizer = None):
  """
  Run the test case for Memphis, Tennessee (cheap city).
  """
  return test_with_inputs("Memphis, Tennessee, US", researcher)


def run_test_wichita(researcher: UserLocalityResearcherOptimizer = None):
  """
  Run the test case for Wichita, Kansas (cheap city).
  """
  return test_with_inputs("Wichita, Kansas, US", researcher)


def run_test_toledo(researcher: UserLocalityResearcherOptimizer = None):
  """
  Run the test case for Toledo, Ohio (cheap city).
  """
  return test_with_inputs("Toledo, Ohio, US", researcher)


def run_test_fort_wayne(researcher: UserLocalityResearcherOptimizer = None):
  """
  Run the test case for Fort Wayne, Indiana (cheap city).
  """
  return test_with_inputs("Fort Wayne, Indiana, US", researcher)


def run_test_lubbock(researcher: UserLocalityResearcherOptimizer = None):
  """
  Run the test case for Lubbock, Texas (cheap city).
  """
  return test_with_inputs("Lubbock, Texas, US", researcher)


def run_test_mobile(researcher: UserLocalityResearcherOptimizer = None):
  """
  Run the test case for Mobile, Alabama (cheap city).
  """
  return test_with_inputs("Mobile, Alabama, US", researcher)


def run_test_jackson(researcher: UserLocalityResearcherOptimizer = None):
  """
  Run the test case for Jackson, Mississippi (cheap city).
  """
  return test_with_inputs("Jackson, Mississippi, US", researcher)


def main(batch: int = 1):
  """
  Main function to test the UserLocalityResearcherOptimizer
  
  Args:
    batch: Batch number (1 or 2) to determine which tests to run
  """
  print("Testing UserLocalityResearcherOptimizer\n")
  
  if batch == 1:
    # Original test cities
    researcher = UserLocalityResearcherOptimizer()
    print("Test 1: Millbrae, California")
    print("-" * 80)
    run_test_millbrae(researcher)
    print("\n")

    print("Test 2: Wichita, Kansas (cheap city)")
    print("-" * 80)
    run_test_wichita(researcher)
    print("\n")
        
    # print("Test 3: Salt Lake City, Utah")
    # print("-" * 80)
    # run_test_salt_lake_city(researcher)
    # print("\n")

  elif batch == 2:
    # Cheaper US cities batch
    researcher = UserLocalityResearcherOptimizer()
    print("Test 1: Memphis, Tennessee (cheap city)")
    print("-" * 80)
    run_test_memphis(researcher)
    print("\n")
    
    print("Test 2: Austin, Texas")
    print("-" * 80)
    run_test_austin(researcher)
    print("\n")
    
    # print("Test 3: Toledo, Ohio (cheap city)")
    # print("-" * 80)
    # run_test_toledo(researcher)
    # print("\n")
    
    # print("Test 4: Fort Wayne, Indiana (cheap city)")
    # print("-" * 80)
    # run_test_fort_wayne(researcher)
    # print("\n")
    
    # print("Test 5: Lubbock, Texas (cheap city)")
    # print("-" * 80)
    # run_test_lubbock(researcher)
    # print("\n")
    
    # print("Test 6: Mobile, Alabama (cheap city)")
    # print("-" * 80)
    # run_test_mobile(researcher)
    # print("\n")
    
    # print("Test 7: Jackson, Mississippi (cheap city)")
    # print("-" * 80)
    # run_test_jackson(researcher)
    # print("\n")
  else:
    raise ValueError("batch must be 1 or 2")
  
  print("All tests completed!")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Run locality research tests in batches')
  parser.add_argument('--batch', type=int, default=1, choices=[1, 2],
                      help='Batch number to run (1 for original cities, 2 for cheap cities)')
  args = parser.parse_args()
  main(batch=args.batch)
