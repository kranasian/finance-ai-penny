from google import genai
from google.genai import types
import json
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

CONFIG = {
  "json": True,
  "sanitize": True,
  "gen_config": {
    "top_k": 40,
    "top_p": 0.95,
    "temperature": 0.6,
    "max_output_tokens": 1024,
    "thinking_budget": 1024,
    "response_mime_type": "application/json",
  },
  "model_name": "gemini-flash-lite-latest",
  "check_template": "Chk:SameSummarizedNameClassifier",
  "replacements": None,
  "output_schema": {
    "type": 5,
    "items": {
      "type": 6,
      "required": ["match_id", "reasoning", "result", "confidence"],
      "properties": {
        "match_id": {
          "type": 1,
          "description": "The unique identifier for the establishment pair",
        },
        "reasoning": {
          "type": 1,
          "description": "Brief explanation of the decision (1-2 sentences)",
        },
        "result": {
          "type": 1,
          "description": "Whether the names are from the same entity/establishment",
          "enum": ["same", "different"],
        },
        "confidence": {
          "type": 1,
          "description": "Confidence level of the result",
          "enum": ["high", "medium", "low"],
        },
      },
    },
    "required": [],
    "properties": {},
  },
}


# Output schema for Gemini (aligned with CONFIG["output_schema"])
SCHEMA = types.Schema(
  type=types.Type.ARRAY,
  items=types.Schema(
    type=types.Type.OBJECT,
    properties={
      "match_id": types.Schema(
        type=types.Type.STRING,
        description="The unique identifier for the establishment pair",
      ),
      "reasoning": types.Schema(
        type=types.Type.STRING,
        description="Brief explanation of the decision (1-2 sentences)",
      ),
      "result": types.Schema(
        type=types.Type.STRING,
        description="Whether the names are from the same entity/establishment",
        enum=["same", "different"],
      ),
      "confidence": types.Schema(
        type=types.Type.STRING,
        description="Confidence level of the result",
        enum=["high", "medium", "low"],
      ),
    },
    required=["match_id", "reasoning", "result", "confidence"],
  ),
)


SYSTEM_PROMPT = """Same **merchant/establishment + same kind of charge** for categorization.

## Steps
1. **`raw_names` primary:** Build identity from `raw_names` first; use `description` to disambiguate. If `raw_names` agree on the merchant but `short_name` disagrees (e.g. both AMAZON.COM vs “Amazon Prime” vs “Amazon.com”), trust **raw_names** → `"same"` when interchangeability still holds.
2. **Physical Locations:** Ignore locations, stores, cities, or branch #s.
3. **Interchangeability:** After that read, evaluate based on the rules below? Both yes → `"same"`; else `"different"`.

## `"different"` if any
- **Generic vs. Specific:** plan/tier/sub-brand/dept ≠ generic parent (e.g. Amazon: Diapers vs Amazon; Ring Annual Plan vs Ring Basic; Venmo to Jose vs Venmo to Jose: Happy Birthday).
- **Marketplace vs. Direct:** DoorDash: McDonald's ≠ McDonald's.
- **Inflow vs. Outflow:** AirBNB Income ≠ AirBNB Payment.
- **Physical vs Web:** brick-and-mortar store vs web storefront (e.g. **Adidas** vs **Adidas.com**) → `"different"`, unless establishment is solely online (e.g. **Amazon** vs **Amazon.com**) → `"same"`.
- **Transfer Labels:** Different **type words** mean different rails: **Bank Transfer** ≠ **Mobile Payment** ≠ **Check Payment**; retry ≠ standard; ACH ≠ standard/plain.
- **Transfer Status:** pending and posted transactions should be different
- **Bank-to-bank / P2P:** Different **\*tails** or beneficiaries → `"different"`; P2P memo/purpose differs → `"different"`; different source accounts → `"different"`.

## `"same"` when
- Only distinction is location, store, city, or branch # (e.g. Jollibee San Francisco = Jollibee = Jollibee Manila).
- **Not** bank-to-bank or P2P: paying a **merchant** through PayPal/Venmo/etc. vs the merchant direct (e.g. PAYPAL *McDonald’s vs McDonald’s) → same merchant; still ignore pure processor noise **unless** it changes **transfer/check/ACH type** above.
- Benign refs, city slugs—ignore for identity unless they encode a **different account** on a **transfer**.

<EXAMPLES>
- Raw `AMAZON.COM` on both sides but `short_name` “Amazon Prime” vs “Amazon.com” → **same** (raw_names win); **Amazon** vs **Amazon.com** → **same** (carve-out); **Adidas** vs **Adidas.com** → **different** (web vs generic physical-channel label).
- **ACH Withdrawal: Apple** vs **ACH Deposit: Apple** vs card **Apple** → **pairwise different**; **Bank Transfer** vs **Online Transfer** vs **Transfer** (same payee) → **different**; **Check Payment** vs **Payment** (non-check) → **different**; **\*1111** vs **\*2222** on B2B/P2P → **different**.
- **PAYPAL *CHIPOTLE** vs **CHIPOTLE** → **same**; **DOORDASH *CHIPOTLE** vs **CHIPOTLE** → **different**; **NETFLIX.COM** vs **NETFLIX** → **same**; **VENMO RETRY** vs **VENMO PAYMENT** → **different**.
</EXAMPLES>

## Output
JSON array, input order: `match_id`, short `reasoning`, `result`, `confidence`. Use `high` rarely.
"""





class SameSummarizedNameClassifierOptimizer:
  """Handles all Gemini API interactions for detecting if names are from the same entity or establishment"""
  
  def __init__(self, model_name=None, thinking_budget=None):
    """Initialize the Gemini agent with API configuration for similarity detection"""
    # API Configuration
    self.client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    gc = CONFIG["gen_config"]

    # Model Configuration
    self.model_name = model_name if model_name is not None else CONFIG["model_name"]
    self.thinking_budget = (
      thinking_budget if thinking_budget is not None else gc["thinking_budget"]
    )

    # Registry / tooling metadata (see CONFIG)
    self.response_json = CONFIG["json"]
    self.sanitize = CONFIG["sanitize"]
    self.check_template = CONFIG["check_template"]
    self.replacements = CONFIG["replacements"]

    # Generation Configuration Constants
    self.temperature = gc["temperature"]
    self.top_p = gc["top_p"]
    self.top_k = gc["top_k"]
    self.max_output_tokens = gc["max_output_tokens"]
    self.response_mime_type = gc["response_mime_type"]
    
    # Safety Settings
    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
    ]
    
    # System Prompt
    self.system_prompt = SYSTEM_PROMPT

  def detect_similarity(self, transaction_history_pairs: list) -> dict:
    """
    Detect if establishment pairs are from the same entity or establishment using Gemini API.

    Returns a dict with ``results`` (parsed JSON array) and ``thought_summary`` when
    the API emits thought parts (same streaming extraction pattern as
    ``PennyAppUsageInfoOptimizer``).

    Args:
      transaction_history_pairs: A list of dictionaries, each containing:
        - match_id: Unique identifier for the pair
        - left: Name object with short_name, raw_names (list), description, amounts
        - right: Name object with short_name, raw_names (list), description, amounts

    Returns:
      ``results``: list of dicts with match_id, result, confidence, reasoning;
      ``thought_summary``: concatenated thought text from the stream (may be empty).
    """
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
      top_k=self.top_k,
      max_output_tokens=self.max_output_tokens,
      safety_settings=self.safety_settings,
      system_instruction=[types.Part.from_text(text=self.system_prompt)],
      thinking_config=types.ThinkingConfig(
        thinking_budget=self.thinking_budget,
        include_thoughts=True,
      ),
      response_mime_type=self.response_mime_type,
      response_schema=SCHEMA,
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
      print("=" * 80)

    if not output_text:
      raise ValueError("Empty response from model.")

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
      return {
        "results": result,
        "thought_summary": thought_summary.strip(),
      }
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


def _run_sandbox_check(transaction_history_pairs: list, result: list, label: str = "Sandbox"):
  """Run checker on classifier output (Sandbox Execution)."""
  if result is None:
    return
  try:
    from check_same_summarized_name_classifier_optimizer import (
      run_test_case,
      CheckSameSummarizedNameClassifier,
    )
    checker = CheckSameSummarizedNameClassifier()
    run_test_case(label, transaction_history_pairs, result, [], checker)
  except Exception as e:
    print(f"Sandbox check failed: {e}")
    import traceback
    print(traceback.format_exc())


def _run_test_with_logging(
  transaction_history_pairs: list,
  detector: SameSummarizedNameClassifierOptimizer = None,
  run_sandbox: bool = True,
):
  """
  Internal helper function that runs a test with consistent logging.

  Args:
    transaction_history_pairs: List of establishment pairs to analyze
    detector: Optional SameSummarizedNameClassifierOptimizer instance. If None, creates a new one.
    run_sandbox: If True, run checker on the classifier output after detection.

  Returns:
    The detection results as a list
  """
  if detector is None:
    detector = SameSummarizedNameClassifierOptimizer()

  # Print the input
  print("=" * 80)
  print("INPUT:")
  print("=" * 80)
  print(json.dumps(transaction_history_pairs, indent=2))
  print("=" * 80)
  print()

  try:
    out = detector.detect_similarity(transaction_history_pairs)
    result = out["results"]

    # Print the output
    print("=" * 80)
    print("OUTPUT:")
    print("=" * 80)
    print(json.dumps(result, indent=2))
    print("=" * 80)
    print()

    if run_sandbox and result:
      print("=" * 80)
      print("SANDBOX EXECUTION (Checker):")
      print("=" * 80)
      _run_sandbox_check(transaction_history_pairs, result, "Optimizer run")
      print()

    return result
  except Exception as e:
    print(f"**Error**: {str(e)}")
    import traceback
    print(traceback.format_exc())
    print("=" * 80)
    return None


def test_batch_1(detector: SameSummarizedNameClassifierOptimizer = None):
  """
  Test batch 1: Basic cases & Payment Processors
  """
  transaction_history_pairs = [
    {
      "match_id": "MSFT-01",
      "left": {
        "short_name": "Microsoft",
        "raw_names": ["MICROSOFT", "Microsoft Corp Billing"],
        "description": "sells software licenses, cloud services, and developer tools",
        "amounts": [12.00, 8.50]
      },
      "right": {
        "short_name": "Microsoft Xbox Game Pass",
        "raw_names": ["Xbox Game Pass Ultimate"],
        "description": "sells gaming subscription for console and PC access",
        "amounts": [14.99, 14.99]
      }
    },
    {
      "match_id": "DISNEY-01",
      "left": {
        "short_name": "Disney+ Basic",
        "raw_names": ["Disney Plus Monthly"],
        "description": "Subscription for streaming Disney movies and series.",
        "amounts": [7.99]
      },
      "right": {
        "short_name": "Disney+ Bundle",
        "raw_names": ["DISNEY PLUS HULU ESPN"],
        "description": "Premium bundle including Disney, Hulu, and ESPN+ streaming.",
        "amounts": [14.99]
      }
    },
    {
      "match_id": "DD-01",
      "left": {
        "short_name": "DoorDash: Chipotle",
        "raw_names": ["DOORDASH *CHIPOTLE"],
        "description": "Food delivery order placed through DoorDash from a Mexican grill chain.",
        "amounts": [22.40]
      },
      "right": {
        "short_name": "Chipotle",
        "raw_names": ["CHIPOTLE MEXICAN GRILL"],
        "description": "Fast-casual Mexican restaurant for dine-in and takeout.",
        "amounts": [18.00, 14.25]
      }
    }
  ]
  return _run_test_with_logging(transaction_history_pairs, detector)


def test_batch_2(detector: SameSummarizedNameClassifierOptimizer = None):
  """
  Test batch 2: Name variations
  """
  transaction_history_pairs = [
    {
      "match_id": "NETFLIX-01",
      "left": {
        "short_name": "Netflix.com",
        "raw_names": ["NETFLIX.COM"],
        "description": "Streaming service for movies and television series.",
        "amounts": [15.49]
      },
      "right": {
        "short_name": "Netflix",
        "raw_names": ["Netflix"],
        "description": "Online video streaming subscription.",
        "amounts": [15.49, 15.49]
      }
    },
    {
      "match_id": "VENMO-01",
      "left": {
        "short_name": "Venmo Retry Payment",
        "raw_names": ["VENMO *RETRY PAYMENT"],
        "description": "A second attempt for a Venmo transaction.",
        "amounts": [32.00]
      },
      "right": {
        "short_name": "Venmo Payment",
        "raw_names": ["Venmo payment"],
        "description": "A standard payment made via Venmo.",
        "amounts": [32.00]
      }
    },
    {
      "match_id": "PAPA-01",
      "left": {
        "short_name": "Papa John's Pizza",
        "raw_names": ["Papa John's Pizza"],
        "description": "pizza restaurant that sells pizzas, sides, and drinks",
        "amounts": [28.50, 34.20]
      },
      "right": {
        "short_name": "Papa Johns Pizza",
        "raw_names": ["PAPA JOHNS REF# 601200089012 AUSTIN,TX Card ending in 8821"],
        "description": "A payment for food at a pizza restaurant",
        "amounts": [19.99, 24.00, 31.45]
      }
    }
  ]
  return _run_test_with_logging(transaction_history_pairs, detector)


def test_batch_3(detector: SameSummarizedNameClassifierOptimizer = None):
  """
  Test batch 3: Sub-brands and Locations
  """
  transaction_history_pairs = [
    {
      "match_id": "TGT-01",
      "left": {
        "short_name": "Target Optical",
        "raw_names": ["Target Optical"],
        "description": "sells eyewear, contact lenses, and eye exams within Target stores",
        "amounts": [89.00, 120.50]
      },
      "right": {
        "short_name": "Target",
        "raw_names": ["Target"],
        "description": "sells general merchandise, groceries, and household goods",
        "amounts": [45.00, 78.30, 112.00]
      }
    },
    {
      "match_id": "HILTON-01",
      "left": {
        "short_name": "Hilton Garden Inn",
        "raw_names": ["Hilton Garden Inn"],
        "description": "sells hotel rooms, breakfast, and meeting space at midscale properties",
        "amounts": [142.00]
      },
      "right": {
        "short_name": "Hilton Garden Inn Phoenix",
        "raw_names": ["HILTON GARDEN INN PHOENIX AIRPORT"],
        "description": "sells hotel accommodations and amenities at an airport location",
        "amounts": [168.00, 155.00]
      }
    },
    {
      "match_id": "SBUX-01",
      "left": {
        "short_name": "Starbucks Downtown Seattle",
        "raw_names": [
          "STARBUCKS STORE 18492 SEATTLE WA",
          "STARBUCKS 18492 DOWNTOWN SEATTLE",
          "STARBUCKS SEATTLE WA"
        ],
        "description": "sells coffee, espresso drinks, and light food at a café location",
        "amounts": [5.85, 7.20, 6.45, 4.90]
      },
      "right": {
        "short_name": "Starbucks",
        "raw_names": ["Starbucks"],
        "description": "sells coffee and café beverages and food",
        "amounts": [6.00, 8.50]
      }
    }
  ]
  return _run_test_with_logging(transaction_history_pairs, detector)
  
  
def test_batch_4(detector: SameSummarizedNameClassifierOptimizer = None):
  """
  Test batch 4: Edge Cases & Ambiguity
  """
  transaction_history_pairs = [
    {
      "match_id": "CASHAPP-01",
      "left": {
        "short_name": "Cash App to Marcus",
        "raw_names": ["Cash App payment to Marcus"],
        "description": "Peer-to-peer money transfer.",
        "amounts": [75.00]
      },
      "right": {
        "short_name": "Cash App to Marcus: Concert tickets",
        "raw_names": ["Cash App to Marcus: Concert tickets"],
        "description": "Peer-to-peer money transfer with a memo.",
        "amounts": [120.00]
      }
    },
    {
      "match_id": "MISC-01",
      "left": {
        "short_name": "Income",
        "raw_names": ["Income"],
        "description": "Undetermined",
        "amounts": [2400.00, 2400.00]
      },
      "right": {
        "short_name": "Uber Eats",
        "raw_names": ["Uber Eats"],
        "description": "sells food delivery from restaurants via the Uber platform",
        "amounts": [35.20, 18.90]
      }
    },
    {
      "match_id": "CVS-01",
      "left": {
        "short_name": "CVS Pharmacy",
        "raw_names": [
          "CVS/PHARMACY #12345 BOSTON MA",
          "CVS PHARMACY BOSTON MA",
          "CVS"
        ],
        "description": "A pharmacy and retail store selling prescriptions, health, and convenience items.",
        "amounts": [12.99, 24.50, 8.75]
      },
      "right": {
        "short_name": "CVS Pharmacy Payment",
        "raw_names": ["CVS/PHARMACY REF# 789012 BOSTON MA Card ending in 4412"],
        "description": "payment to CVS Pharmacy with reference number 789012",
        "amounts": [12.99]
      }
    }
  ]
  return _run_test_with_logging(transaction_history_pairs, detector)


def main(batch: int = 0):
  """
  Main function to test the similarity detector optimizer
  
  Args:
    batch: Batch number (1, 2, 3, or 4) to determine which tests to run. 0 runs all.
  """
  print("Testing SameSummarizedNameClassifierOptimizer\n")
  detector = SameSummarizedNameClassifierOptimizer()
  
  if batch == 1 or batch == 0:
    print("Test Batch 1: Basic cases")
    print("-" * 80)
    test_batch_1(detector)
    print("\n")
    
  if batch == 2 or batch == 0:
    print("Test Batch 2: Name variations")
    print("-" * 80)
    test_batch_2(detector)
    print("\n")
    
  if batch == 3 or batch == 0:
    print("Test Batch 3: Similar names")
    print("-" * 80)
    test_batch_3(detector)
    print("\n")
    
  if batch == 4 or batch == 0:
    print("Test Batch 4: Edge cases")
    print("-" * 80)
    test_batch_4(detector)
    print("\n")
    
  if batch not in [0, 1, 2, 3, 4]:
    raise ValueError("batch must be 0, 1, 2, 3, or 4")
  
  print("All tests completed!")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Run tests in batches')
  parser.add_argument('--batch', type=int, default=0, choices=[0, 1, 2, 3, 4],
                      help='Batch number to run (1, 2, 3, or 4). 0 runs all batches.')
  args = parser.parse_args()
  main(batch=args.batch)
