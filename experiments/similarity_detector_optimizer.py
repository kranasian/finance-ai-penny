from google import genai
from google.genai import types
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()


SYSTEM_PROMPT = """You detect if two names represent the same entity or establishment.

## Task

Analyze a JSON array of establishment pairs. For each pair, determine if the names are from the same or different entities/establishments, assign confidence, and provide brief reasoning.

## Input Format

JSON array where each element contains:
- `match_id`: Unique identifier
- `left` and `right`: Each with:
  - `short_name`: Shortened/cleaned name
  - `raw_names`: Array of original raw names
  - `description`: Establishment description
  - `amounts`: Array of amounts

## Output Format

JSON array where each element contains:
- `match_id`: Same as input
- `result`: "same" or "different"
- `confidence`: "high", "medium", or "low"
- `reasoning`: 1-2 sentence explanation focusing on decisive factors

## Analysis

Compare names, descriptions, and amounts to determine entity identity:

**Names** (primary): Compare `short_name` and all `raw_names`. Look for:
- Exact matches (case-insensitive)
- Partial matches (one contains the other)
- Semantic similarity (e.g., "AT&T Bill" vs "AT&T Bill Charge")
- Common variations (abbreviations, punctuation)

**Description** (key indicator): 
- Similar/matching descriptions → same entity
- Different descriptions → different entities
- Use to disambiguate when names are the same

**Amounts** (supporting):
- Similar amounts can support same entity (e.g., recurring bills)
- Different amounts don't necessarily mean different entities
- Secondary to names/descriptions

**Confidence**:
- **high**: Strong evidence (exact name + matching description, or completely different names/descriptions)
- **medium**: Good evidence with some ambiguity (similar names/descriptions, slight variations)
- **low**: Weak/conflicting evidence (similar names but different descriptions, ambiguous matches)

## Rules

- Process all pairs, maintain input order
- "same" = same entity/establishment, "different" = different entities/establishments
- Focus on entity identity, not just name similarity
- Be conservative with "high" confidence
- Reasoning: 1-2 sentences, focus on decisive factors only
"""





class SimilarityDetectorOptimizer:
  """Handles all Gemini API interactions for detecting if names are from the same entity or establishment"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for similarity detection"""
    # API Configuration
    self.client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    
    # Model Configuration
    self.thinking_budget = 4096
    self.model_name = model_name
    
    # Generation Configuration Constants
    self.temperature = 0.6
    self.top_p = 0.95
    self.max_output_tokens = 2048
    
    # Safety Settings
    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
    ]
    
    # System Prompt
    self.system_prompt = SYSTEM_PROMPT
    
    # Output Schema - array of result objects
    result_item_schema = types.Schema(
      type=types.Type.OBJECT,
      properties={
        "match_id": types.Schema(type=types.Type.INTEGER),
        "result": types.Schema(
          type=types.Type.STRING,
          enum=["same", "different"]
        ),
        "confidence": types.Schema(
          type=types.Type.STRING,
          enum=["high", "medium", "low"]
        ),
        "reasoning": types.Schema(type=types.Type.STRING)
      },
      required=["match_id", "result", "confidence", "reasoning"]
    )
    
    self.output_schema = types.Schema(
      type=types.Type.ARRAY,
      items=result_item_schema
    )

  def detect_similarity(self, transaction_history_pairs: list) -> list:
    """
    Detect if establishment pairs are from the same entity or establishment using Gemini API.
    
    Args:
      transaction_history_pairs: A list of dictionaries, each containing:
        - match_id: Unique identifier for the pair
        - left: Name object with short_name, raw_names (list), description, amounts
        - right: Name object with short_name, raw_names (list), description, amounts
      
    Returns:
      A list of dictionaries, each containing:
        - match_id: The same match_id from input
        - result: "same" (same entity/establishment) or "different" (different entities/establishments)
        - confidence: "high", "medium", or "low"
        - reasoning: Explanation string
    """
    import json
    
    # Convert input to JSON string
    input_json = json.dumps(transaction_history_pairs, indent=2)
    
    # Create request text
    request_text = types.Part.from_text(text=f"""input:
{input_json}

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
      response_mime_type="application/json",
      response_schema=self.output_schema,
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


def _run_test_with_logging(transaction_history_pairs: list, detector: SimilarityDetectorOptimizer = None):
  """
  Internal helper function that runs a test with consistent logging.
  
  Args:
    transaction_history_pairs: List of establishment pairs to analyze
    detector: Optional SimilarityDetectorOptimizer instance. If None, creates a new one.
    
  Returns:
    The detection results as a list
  """
  import json
  
  if detector is None:
    detector = SimilarityDetectorOptimizer()
  
  # Print the input
  print("=" * 80)
  print("INPUT:")
  print("=" * 80)
  print(json.dumps(transaction_history_pairs, indent=2))
  print("=" * 80)
  print()
  
  try:
    result = detector.detect_similarity(transaction_history_pairs)
    
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


def test_multiple_pairs(detector: SimilarityDetectorOptimizer = None):
  """
  Test method for multiple pairs of entities/establishments.
  
  Args:
    detector: Optional SimilarityDetectorOptimizer instance. If None, creates a new one.
    
  Returns:
    The detection results
  """
  transaction_history_pairs = [
    {
      "match_id": 2112,
      "left": {
        "short_name": "OpenAI",
        "raw_names": [
          "OPENAI",
          "OPENAI +14158799686 USA"
        ],
        "description": "sells access to its large language model API, including text generation, translation, and code completion",
        "amounts": [6.00, 5.32]
      },
      "right": {
        "short_name": "OpenAI Chatgpt",
        "raw_names": ["Openai Chatgpt Subscr"],
        "description": "sells subscriptions to use the AI chatbot for various tasks such as writing, coding, and problem-solving",
        "amounts": [21.28, 20.00, 21.28, 21.28, 21.28, 21.28, 21.28]
      }
    },
    {
      "match_id": 2113,
      "left": {
        "short_name": "MGM Grand",
        "raw_names": ["MGM Grand"],
        "description": "sells hotel rooms, casino gaming, dining, entertainment, and other resort amenities",
        "amounts": [128.93]
      },
      "right": {
        "short_name": "MGM Grand Detroit",
        "raw_names": ["Mgm grand detroi"],
        "description": "sells various types of casino games, including slots, table games, and poker",
        "amounts": [506.23, 1023.44, 1057.96, 1166.57, 1181.45, 1168.88]
      }
    },
    {
      "match_id": 2114,
      "left": {
        "short_name": "Klarna: Revolve",
        "raw_names": ["Klarna* Revolve, +"],
        "description": "sells clothing, accessories, and home goods",
        "amounts": [12.87]
      },
      "right": {
        "short_name": "Revolve",
        "raw_names": ["REVOLVE"],
        "description": "sells clothing, shoes, accessories, and other fashion items, known for their trendy and stylish designs",
        "amounts": [85.52, 32.48, 106.19]
      }
    },
    {
      "match_id": 2115,
      "left": {
        "short_name": "Texas Roadhouse",
        "raw_names": [
          "Texas Roadhouse",
          "Logans Roadhouse"
        ],
        "description": "steak restaurant",
        "amounts": [23.38, 81.17, 69.20, 74.86, 79.13]
      },
      "right": {
        "short_name": "Texas SOS",
        "raw_names": ["TEXAS S.O.S. REF# 510600029749 512-463-9308,MD Card ending in 3466"],
        "description": "provides services related to vehicle registration and titling in Texas",
        "amounts": [0.41]
      }
    },
    {
      "match_id": 2116,
      "left": {
        "short_name": "ACH Deposit: Flyte",
        "raw_names": ["ACH DEPOSIT: Flyte"],
        "description": "processes payments for airline tickets and travel services",
        "amounts": [6027.77, 4790.02, 3865.66, 3865.66, 3865.66, 608.41]
      },
      "right": {
        "short_name": "ACH Deposit: Tesla",
        "raw_names": ["ACH DEPOSIT: Tesla_Inc XFA15LS6DS9X5VX"],
        "description": "electric vehicle and clean energy company",
        "amounts": [134.14, 134.14, 112.33, 112.33]
      }
    }
  ]
  return _run_test_with_logging(transaction_history_pairs, detector)


def test_similar_names_different_stores(detector: SimilarityDetectorOptimizer = None):
  """
  Test method for similar names but different amounts (should be "different" with medium confidence).
  
  Args:
    detector: Optional SimilarityDetectorOptimizer instance. If None, creates a new one.
    
  Returns:
    The detection results
  """
  transaction_history_pairs = [
    {
      "match_id": 2115,
      "left": {
        "short_name": "Macho's",
        "raw_names": [
          "PY *MACHO SE REF# 509700022869 972-525-0686,TX Card ending in 3458",
          "PY *MACHO SE REF# 506600014887 972-525-0686,TX Card ending in 3458"
        ],
        "description": "sells Mexican food such as tacos, burritos, and quesadillas",
        "amounts": [50.00, 50.00]
      },
      "right": {
        "short_name": "Marco's Pizza",
        "raw_names": ["Marco's Pizza"],
        "description": "pizza restaurant that sells a variety of pizzas, subs, wings, sides, and desserts",
        "amounts": [16.23, 16.23, 16.23, 22.70, 27.48, 23.97]
      }
    }
  ]
  return _run_test_with_logging(transaction_history_pairs, detector)


def test_similar_names_same_stores1(detector: SimilarityDetectorOptimizer = None):
  """
  Test method for similar names but different amounts (should be "different" with medium confidence).
  
  Args:
    detector: Optional SimilarityDetectorOptimizer instance. If None, creates a new one.
    
  Returns:
    The detection results
  """
  transaction_history_pairs = [
    {
      "match_id": 2115,
      "left": {
        "short_name": "Marco's Pizza",
        "raw_names": ["Marco's Pizza"],
        "description": "pizza restaurant that sells a variety of pizzas, subs, wings, sides, and desserts",
        "amounts": [70.92, 31.37]
      },
      "right": {
        "short_name": "Marcos Pizza",
        "raw_names": ["MARCOS PIZZA REF# 505100025146 WAXAHACHIE,TX Card ending in 3458"],
        "description": "A payment for food and beverages at a pizza restaurant",
        "amounts": [16.23, 16.23, 16.23, 22.70, 27.48, 23.97]
      }
    }
  ]
  return _run_test_with_logging(transaction_history_pairs, detector)

def test_similar_names_same_stores2(detector: SimilarityDetectorOptimizer = None):
  """
  Test method for similar names but different amounts (should be "different" with medium confidence).
  
  Args:
    detector: Optional SimilarityDetectorOptimizer instance. If None, creates a new one.
    
  Returns:
    The detection results
  """
  transaction_history_pairs = [
    {
      "match_id": 2115,
      "left": {
        "short_name": "Cleveland Marriott",
        "raw_names": [
          "MOBILE PURCHASE 0501 CLE EMBERS 681115 CLEVELAND OH XXXXX8751XXXXXXXXXX5621",
          "MOBILE PURCHASE CLE EMBERS 681115 CLEVELAND OH ON 05/01",
          "MOBILE PURCHASE CLE SPORTS ST1576 CLEVELAND OH ON 05/01",
          "MOBILE PURCHASE CLEVELAND MARRIOT CLEVELAND OH ON 04/30" 
        ],
        "description": "sells hotel accommodations, meeting rooms, and other services",
        "amounts": [5.40, 46.04, 23.67, 10.72, 43.88]
      },
      "right": {
        "short_name": "Marriott",
        "raw_names": ["Marriott"],
        "description": "sells hotel accommodations, resort stays, and other travel-related services",
        "amounts": [23.47, 6.43, 7.54]
      }
    }
  ]
  return _run_test_with_logging(transaction_history_pairs, detector)

def test_establishment_undetermined(detector: SimilarityDetectorOptimizer = None):
  """
  Test method for establishment undetermined (should be "undetermined" with medium confidence).
  
  Args:
    detector: Optional SimilarityDetectorOptimizer instance. If None, creates a new one.
    
  Returns:
    The detection results
  """
  transaction_history_pairs = [
    {
      "match_id": 2115,
      "left": {
        "short_name": "Sales",
        "raw_names": ["Sales"],
        "description": "Undetermined",
        "amounts": [1581.33, 2366.88, 3487.34, 1582.67, 3181.54, 439.55]
      },
      "right": {
        "short_name": "Seamless",
        "raw_names": ["Seamless"],
        "description": "sells food delivery services from various restaurants",
        "amounts": [84.29, 52.17, 78.39, 46.76]
      }
    }
  ]
  return _run_test_with_logging(transaction_history_pairs, detector)


def test_similar_amounts_description_small_name_variation(detector: SimilarityDetectorOptimizer = None):
  """
  Test method for similar amounts, similar description, and small variation in names (should be "same" with high/medium confidence).
  
  Args:
    detector: Optional SimilarityDetectorOptimizer instance. If None, creates a new one.
    
  Returns:
    The detection results
  """
  transaction_history_pairs = [
    {
      "match_id": 2116,
      "left": {
        "short_name": "Better Proposal",
        "raw_names": [
          "BETTER PROPO REF# 510800012325 LONDON,GB Card ending in 3458",
          "BETTER PROPO REF# 504900028199 LONDON,GB Card ending in 3458"
        ],
        "description": "provides online marketing and advertising services",
        "amounts": [29.00, 29.00, 29.00]
      },
      "right": {
        "short_name": "Better Proposals",
        "raw_names": ["BETTER PROPOSALS LONDON 253"],
        "description": "sells various proposal writing and business consulting services",
        "amounts": [29.00, 29.00, 29.00, 29.00, 29.00, 29.00]
      }
    },
    {
      "match_id": 2117,
      "left": {
        "short_name": "Dollar Tree",
        "raw_names": [
          "DOLLARTREE REF# 504200037537 WAXAHACHIE,TX Card ending in 3466",
          "DOLLARTREE REF# 504500026805 DESOTO,TX Card ending in 3466",
          "DOLLARTREE REF# 505200026122 WAXAHACHIE,TX Card ending in 3466",
          "Dollartree"
        ],
        "description": "A discount retail store selling a variety of household goods, groceries, and seasonal items.",
        "amounts": [4.06, 18.34, 5.34, 11.13]
      },
      "right": {
        "short_name": "Dollar Tree Payment",
        "raw_names": ["DOLLARTREE REF# 502300022154 DESOTO,TX Card ending in 3466"],
        "description": "payment to Dollar Tree with reference number 502300022154",
        "amounts": [4.06]
      }
    }
  ]
  return _run_test_with_logging(transaction_history_pairs, detector)

def main():
  """Main function to test the similarity detector optimizer"""
  detector = SimilarityDetectorOptimizer()
  
  # print("Test 1: Multiple pairs")
  # test_multiple_pairs(detector)
  # print("\n")

  # print("Test 2: Similar names, different stores")
  # test_similar_names_different_stores(detector)
  # print("\n")

  # print("Test 3: Similar names, same stores")
  # test_similar_names_same_stores1(detector)
  # test_similar_names_same_stores2(detector)
  # print("\n")

  # print("Test 4: Establishment undetermined")
  # test_establishment_undetermined(detector)
  # print("\n")
  
  print("Test 5: Similar amounts, description, and small name variation")
  test_similar_amounts_description_small_name_variation(detector)


if __name__ == "__main__":
  main()
