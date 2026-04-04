from google import genai
from google.genai import types
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CONFIG = {
  "json": True,
  "sanitize": True,
  "gen_config": {
    "top_k": 40,
    "top_p": 0.95,
    "temperature": 0.5,
    "thinking_budget": 1024,
    "max_output_tokens": 4096,
  },
  "model_name": "gemini-flash-lite-latest",
}

CHECK_RESPONSE_SCHEMA = types.Schema(
  type=types.Type.OBJECT,
  properties={
    "good_copy": types.Schema(type=types.Type.BOOLEAN),
    "info_correct": types.Schema(type=types.Type.BOOLEAN),
    "eval_text": types.Schema(type=types.Type.STRING),
  },
  required=["good_copy", "info_correct"],
)

SYSTEM_PROMPT = """You are a checker for SameSummarizedNameClassifier outputs (`same` / `different` per pair).

## Inputs
**EVAL_INPUT** (pairs), **PAST_REVIEW_OUTCOMES**, **REVIEW_NEEDED** (classifier JSON array).

## Your JSON
`{"good_copy": boolean, "info_correct": boolean, "eval_text": string}`
- **good_copy**: REVIEW_NEEDED is a JSON array, one entry per EVAL_INPUT pair in order, each with `match_id`, `reasoning`, `result`, `confidence` (see Output).
- **info_correct**: Every `result` matches **Policy** below. Correctness uses `result` only; ignore `reasoning` unless debugging a wrong `result`.
- **eval_text**: Empty if both true. Else one short line per bad `match_id` (≤25 words, plain English, no rule numbers); use `\\n` between lines; optional `match_id:` prefix. Never cite PAST_REVIEW_OUTCOMES; do not repeat the model’s `result` verbatim.

## Past reviews
If REVIEW_NEEDED repeats an error already flagged in PAST_REVIEW_OUTCOMES, mark **info_correct** false. Do not mention past reviews inside **eval_text**.

## How to verify
1. PAST_REVIEW_OUTCOMES check. 2. **good_copy** shape/order. 3. **info_correct**: `raw_names` first, then `description`; then Policy. 4. **eval_text** only if needed.

## Policy (ground truth for `result`)

**Goal:** same **merchant/establishment** and same **kind of charge** (rail/channel/product line).

**Read order:** `raw_names` → `description` → `short_name`. If `raw_names` agree on merchant but `short_name` conflicts, trust `raw_names` when interchangeability still holds.

**Always ignore:** city/branch/store/location in the name (e.g. Jollibee Manila = Jollibee San Francisco); legal noise (LLC, Inc.); **payment platform** in front of a merchant (PayPal/Venmo/Apple Pay *Merchant vs Merchant) **unless** it changes transfer/check/ACH **type** below.

**P2P payee names:** Treat **same** if the payee is clearly one person with **minor spelling, initials, or incompleteness** (e.g. “Alex J.” vs “Alexander Jones”). **Different** if memo/purpose differs, **\*tails** differ, beneficiaries differ, or accounts differ.

**ACH:** Any ACH-labeled charge vs a non-ACH card/standard charge to the **same brand name** → **different** (e.g. ACH Amazon ≠ Amazon card). ACH withdrawal vs ACH deposit vs card → treat as **different** kinds where labels differ.

**`different` if any holds**
- **Different products/plans/tiers** from one company → **different** (Ring Yearly vs Ring Standard vs Ring; diaper SKU vs store brand).
- Marketplace / delivery wrapper vs direct merchant (DoorDash McDonald’s vs McDonald’s).
- Inflow vs outflow (e.g. Airbnb income vs payment).
- **Physical** store identity vs **web** storefront when the business is not online-only (**Adidas** vs **Adidas.com** → different). **Online-only** → **.com** vs bare name → **same** (Netflix.com vs Netflix).
- **Transfer type words** differ: Bank Transfer ≠ Mobile Payment ≠ Check Payment; retry ≠ standard; **ACH ≠ non-ACH**.
- Pending vs posted.
- P2P/B2B: different **\*tails**, beneficiaries, source accounts; **different memo/purpose** on P2P.

**`same` when**
- Only location/branch/city among outlets of one chain.
- Merchant via platform vs direct (same merchant); ignore processor-only noise except rail-changing cases above.
- **One company, abbreviation or carrier/service line** in `raw_names` (e.g. Federal Express vs FEDEX / FEDEX GROUND) → **same** when clearly the same corporate brand, not a separate sub-brand product.

<EXAMPLES>
Permutations (apply all): Netflix.com|Netflix→same (online-only); Amazon|Amazon.com→same (online carve-out); Adidas|Adidas.com→diff (physical vs web); PAYPAL*Chipotle|Chipotle→same (ignore platform); DOORDASH*Chipotle|Chipotle→diff (marketplace); ACH Amazon|Amazon card→diff (rail); Jollibee MNL|Jollibee SF→same (location); Venmo A.Kim|Venmo Alex Kim→same (P2P name variance); Venmo Alex|Venmo Alex:Rent→diff (memo); Ring Yearly|Ring Standard|Ring→pairwise diff (products); *3232|*4545 P2P→diff (tails); Bank Transfer vs ACH vs Check vs plain Payment→diff (labels).
</EXAMPLES>

## Output (classifier copy)
JSON array, input order: `match_id`, short `reasoning`, `result` ∈ {same, different}, `confidence` ∈ {high, medium, low}. Use `high` rarely.
"""

class CheckSameSummarizedNameClassifier:
  """Handles all Gemini API interactions for checking SameSummarizedNameClassifier outputs against rules"""
  
  def __init__(self, model_name=None, thinking_budget=None):
    """Initialize the Gemini agent with API configuration for checking SameSummarizedNameClassifier evaluations"""
    # API Configuration
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file or environment.")
    self.client = genai.Client(api_key=api_key)

    gc = CONFIG["gen_config"]

    # Model Configuration
    self.model_name = model_name if model_name is not None else CONFIG["model_name"]
    self.thinking_budget = (
      thinking_budget if thinking_budget is not None else gc["thinking_budget"]
    )

    self.response_json = CONFIG["json"]
    self.sanitize = CONFIG["sanitize"]

    # Generation Configuration Constants
    self.top_k = gc["top_k"]
    self.top_p = gc["top_p"]
    self.temperature = gc["temperature"]
    self.max_output_tokens = gc["max_output_tokens"]
    
    # Safety Settings
    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
    ]
    
    # System Prompt
    self.system_prompt = SYSTEM_PROMPT

  
  def generate_response(self, eval_input: list, past_review_outcomes: list, review_needed: list) -> dict:
    """
    Generate a response using Gemini API for checking SameSummarizedNameClassifier outputs.
    
    Args:
      eval_input: A JSON array of transaction pairs.
      past_review_outcomes: An array of past review outcomes, each containing `output`, `good_copy`, `info_correct`, and `eval_text`.
      review_needed: The SameSummarizedNameClassifier output that needs to be reviewed (JSON array).
      
    Returns:
      Dictionary with good_copy, info_correct, eval_text, and thought_summary (when the API
      emits thought parts; same streaming extraction pattern as PennyAppUsageInfoOptimizer).
    """
    # Create request text with the new input structure
    request_text_str = f"""<EVAL_INPUT>

{json.dumps(eval_input, indent=2)}

</EVAL_INPUT>

<PAST_REVIEW_OUTCOMES>

{json.dumps(past_review_outcomes, indent=2)}

</PAST_REVIEW_OUTCOMES>

<REVIEW_NEEDED>

{json.dumps(review_needed, indent=2)}

</REVIEW_NEEDED>

Output:"""
    
    print(request_text_str)
    print(f"\n{'='*80}\n")
    
    request_text = types.Part.from_text(text=request_text_str)
    
    # Create content and configuration
    contents = [types.Content(role="user", parts=[request_text])]
    
    config_kwargs = dict(
      top_k=self.top_k,
      top_p=self.top_p,
      temperature=self.temperature,
      max_output_tokens=self.max_output_tokens,
      safety_settings=self.safety_settings,
      system_instruction=[types.Part.from_text(text=self.system_prompt)],
      thinking_config=types.ThinkingConfig(
        thinking_budget=self.thinking_budget,
        include_thoughts=True,
      ),
    )
    if self.response_json:
      config_kwargs["response_mime_type"] = "application/json"
      config_kwargs["response_schema"] = CHECK_RESPONSE_SCHEMA
    generate_content_config = types.GenerateContentConfig(**config_kwargs)

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

    # Check if response is empty
    if not output_text or not output_text.strip():
      raise ValueError(f"Empty response from model. Check API key and model availability.")
    
    # Parse JSON response
    try:
      # Remove markdown code blocks if present
      if "```json" in output_text:
        json_start = output_text.find("```json") + 7
        json_end = output_text.find("```", json_start)
        if json_end != -1:
          output_text = output_text[json_start:json_end].strip()
      elif "```" in output_text:
        # Try to find JSON in code blocks
        json_start = output_text.find("```") + 3
        json_end = output_text.find("```", json_start)
        if json_end != -1:
          output_text = output_text[json_start:json_end].strip()
      
      # Extract JSON object from the response
      json_start = output_text.find('{')
      json_end = output_text.rfind('}') + 1
      
      if json_start != -1 and json_end > json_start:
        json_str = output_text[json_start:json_end]
        parsed = json.loads(json_str)
      else:
        parsed = json.loads(output_text.strip())
      parsed["thought_summary"] = thought_summary.strip()
      return parsed
    except json.JSONDecodeError as e:
      raise ValueError(f"Failed to parse JSON response: {e}\nResponse length: {len(output_text)}\nResponse preview: {output_text[:500]}")


def run_test_case(test_name: str, eval_input: list, review_needed: list, past_review_outcomes: list = None, checker: 'CheckSameSummarizedNameClassifier' = None):
  """
  Utility function to run a test case with custom inputs, common error handling, and output formatting.
  
  Args:
    test_name: Name of the test case
    eval_input: A list of transaction pairs to be evaluated.
    review_needed: The classifier output that needs to be reviewed (list of dicts).
    past_review_outcomes: An array of past review outcomes. Defaults to empty list.
    checker: Optional CheckSameSummarizedNameClassifier instance. If None, creates a new one.
    
  Returns:
    Dictionary with good_copy, info_correct, eval_text, and thought_summary keys, or None if error occurred
  """
  if past_review_outcomes is None:
    past_review_outcomes = []
  if checker is None:
    checker = CheckSameSummarizedNameClassifier()

  print(f"\n{'='*80}")
  print(f"Running test: {test_name}")
  print(f"{'='*80}")

  try:
    # Directly call the checker's response with the provided inputs.
    result = checker.generate_response(eval_input, past_review_outcomes, review_needed)
    print(f"Result:")
    print(json.dumps(result, indent=2))
    print(f"{'='*80}")
    return result
  except Exception as e:
    print(f"ERROR: {type(e).__name__}: {str(e)}")
    print(f"{'='*80}")
    return None


def run_correct_response(checker: CheckSameSummarizedNameClassifier = None):
  """
  Run the test case for a correct response.
  """
  eval_input = [
    {
      "match_id": "2112:2113",
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
        "amounts": [21.28, 20.00]
      }
    },
    {
        "match_id": "SPOT-01",
        "left": {
            "short_name": "Spotify Silver Plan",
            "raw_names": ["Spotify Silver Plan"],
            "description": "Subscription for music streaming service.",
            "amounts": [9.99]
        },
        "right": {
            "short_name": "Spotify Platinum",
            "raw_names": ["SPOTIFY PLATINUM"],
            "description": "Premium subscription for music and podcast streaming.",
            "amounts": [15.99]
        }
    },
    {
        "match_id": "GRAB-01",
        "left": {
            "short_name": "Grab: Wendy's",
            "raw_names": ["GrabFood*Wendy's"],
            "description": "Food delivery from a fast-food chain.",
            "amounts": [12.50]
        },
        "right": {
            "short_name": "Wendy's",
            "raw_names": ["WENDY'S"],
            "description": "Fast-food restaurant specializing in hamburgers.",
            "amounts": [18.75]
        }
    }
  ]
  
  review_needed = [
    {
      "match_id": "2112:2113",
      "reasoning": "OpenAI API access and ChatGPT subscriptions are distinct services from the same company, making them different.",
      "result": "different",
      "confidence": "high"
    },
    {
      "match_id": "SPOT-01",
      "reasoning": "Spotify Silver and Platinum are different subscription tiers for the same service.",
      "result": "different",
      "confidence": "high"
    },
    {
      "match_id": "GRAB-01",
      "reasoning": "A transaction made through the Grab marketplace is different from a direct transaction with Wendy's.",
      "result": "different",
      "confidence": "high"
    }
  ]
  
  return run_test_case("correct_response", eval_input, review_needed, [], checker)


def run_edge_cases(checker: CheckSameSummarizedNameClassifier = None):
  """
  Run the test case for edge cases including online vs physical and different locations.
  """
  eval_input = [
    {
        "match_id": "AMZN-01",
        "left": {
            "short_name": "Amazon.com",
            "raw_names": ["AMAZON.COM"],
            "description": "Online retailer.",
            "amounts": [50.0]
        },
        "right": {
            "short_name": "Amazon Go",
            "raw_names": ["Amazon Go Store"],
            "description": "Physical convenience store.",
            "amounts": [12.50]
        }
    },
    {
        "match_id": "MGM-01",
        "left": {
            "short_name": "MGM Grand",
            "raw_names": ["MGM Grand Las Vegas"],
            "description": "Hotel and casino in Las Vegas.",
            "amounts": [300.0]
        },
        "right": {
            "short_name": "MGM Grand Detroit",
            "raw_names": ["MGM Grand Detroit"],
            "description": "Hotel and casino in Detroit.",
            "amounts": [250.0]
        }
    }
  ]
  
  review_needed = [
    {
      "match_id": "AMZN-01",
      "reasoning": "Amazon.com is an online store and Amazon Go is a physical store, so they are different.",
      "result": "different",
      "confidence": "high"
    },
    {
      "match_id": "MGM-01",
      "reasoning": "These are the same hotel chain, just different locations.",
      "result": "same",
      "confidence": "high"
    }
  ]
  
  return run_test_case("edge_cases", eval_input, review_needed, [], checker)


def run_test_group_1(checker: CheckSameSummarizedNameClassifier = None):
  """
  Test Group 1: Name variations (abbreviations, punctuation, online .com)
  """
  eval_input = [
    {
      "match_id": "NETFLIX-01",
      "left": { "short_name": "Netflix.com", "raw_names": ["NETFLIX.COM"], "description": "Streaming subscription.", "amounts": [15.99, 18.50] },
      "right": { "short_name": "Netflix", "raw_names": ["Netflix"], "description": "Online video streaming service.", "amounts": [15.99] }
    },
    {
      "match_id": "FEDEX-01",
      "left": { "short_name": "Federal Express", "raw_names": ["FEDERAL EXPRESS"], "description": "Shipping and logistics.", "amounts": [42.00] },
      "right": { "short_name": "FedEx", "raw_names": ["FEDEX GROUND"], "description": "Package delivery company.", "amounts": [28.75] }
    },
    {
      "match_id": "CHIPOTLE-01",
      "left": { "short_name": "Chipotle Mexican Grill", "raw_names": ["CHIPOTLE MEXICAN GRILL"], "description": "Fast-casual Mexican restaurant.", "amounts": [12.45, 14.20] },
      "right": { "short_name": "Chipotle", "raw_names": ["CHIPOTLE"], "description": "Mexican grill chain.", "amounts": [11.80] }
    }
  ]
  review_needed = [
    { "match_id": "NETFLIX-01", "reasoning": "Both are online-only; .com suffix does not make them different.", "result": "same", "confidence": "high" },
    { "match_id": "FEDEX-01", "reasoning": "Federal Express and FedEx are the same company; one is an abbreviation.", "result": "same", "confidence": "high" },
    { "match_id": "CHIPOTLE-01", "reasoning": "Full name vs shortened name refer to the same restaurant chain.", "result": "same", "confidence": "high" }
  ]
  return run_test_case("test_group_1_name_variations", eval_input, review_needed, [], checker)


def run_test_group_2(checker: CheckSameSummarizedNameClassifier = None):
  """
  Test Group 2: Sub-brands, store formats, and locations
  """
  eval_input = [
    {
      "match_id": "TARGET-01",
      "left": { "short_name": "Target Optical", "raw_names": ["TARGET OPTICAL"], "description": "Eyewear and eye exams within Target.", "amounts": [89.00, 120.50] },
      "right": { "short_name": "Target", "raw_names": ["TARGET"], "description": "General merchandise retailer.", "amounts": [45.00, 78.30] }
    },
    {
      "match_id": "HILTON-01",
      "left": { "short_name": "Hilton Chicago", "raw_names": ["HILTON CHICAGO"], "description": "Hotel in Chicago.", "amounts": [320.00] },
      "right": { "short_name": "Hilton", "raw_names": ["HILTON AUSTIN"], "description": "Hotel chain.", "amounts": [285.00] }
    },
    {
      "match_id": "CVS-01",
      "left": { "short_name": "CVS Pharmacy", "raw_names": ["CVS/PHARMACY"], "description": "Pharmacy and retail store.", "amounts": [22.40, 15.99] },
      "right": { "short_name": "CVS", "raw_names": ["CVS"], "description": "Pharmacy and convenience retailer.", "amounts": [8.50] }
    }
  ]
  review_needed = [
    { "match_id": "TARGET-01", "reasoning": "Target Optical is a distinct department/service within Target; not interchangeable with general Target.", "result": "different", "confidence": "high" },
    { "match_id": "HILTON-01", "reasoning": "Same hotel brand at different cities; location alone does not make them different.", "result": "same", "confidence": "high" },
    { "match_id": "CVS-01", "reasoning": "CVS Pharmacy and CVS are the same chain; 'Pharmacy' is store format wording.", "result": "same", "confidence": "high" }
  ]
  return run_test_case("test_group_2_sub_brands_and_locations", eval_input, review_needed, [], checker)


def run_test_group_3(checker: CheckSameSummarizedNameClassifier = None):
  """
  Test Group 3: Transfers with memos, marketplaces, and payment wording
  """
  eval_input = [
    {
      "match_id": "VENMO-01",
      "left": { "short_name": "Venmo to Alex", "raw_names": ["Venmo payment to Alex"], "description": "P2P payment.", "amounts": [35.00] },
      "right": { "short_name": "Venmo to Alex: Rent", "raw_names": ["Venmo to Alex Rent December"], "description": "P2P payment with memo.", "amounts": [850.00] }
    },
    {
      "match_id": "UBER-01",
      "left": { "short_name": "Uber Eats: McDonald's", "raw_names": ["UBER EATS MCDONALDS"], "description": "Food delivery via Uber Eats from McDonald's.", "amounts": [18.40] },
      "right": { "short_name": "McDonald's", "raw_names": ["MCDONALDS"], "description": "Fast food restaurant.", "amounts": [9.99] }
    },
    {
      "match_id": "SHELL-01",
      "left": { "short_name": "Shell Gas", "raw_names": ["SHELL OIL"], "description": "Gas station fuel purchase.", "amounts": [52.00] },
      "right": { "short_name": "Shell Fuel Purchase", "raw_names": ["SHELL FUEL PURCHASE"], "description": "Fuel purchase at Shell.", "amounts": [48.30] }
    }
  ]
  review_needed = [
    { "match_id": "VENMO-01", "reasoning": "Transfer with a specific memo (Rent) is different from a general transfer to the same person.", "result": "different", "confidence": "high" },
    { "match_id": "UBER-01", "reasoning": "Uber Eats delivery from McDonald's is a marketplace transaction; direct McDonald's is not.", "result": "different", "confidence": "high" },
    { "match_id": "SHELL-01", "reasoning": "'Gas' and 'Fuel Purchase' are redundant; both refer to the same merchant.", "result": "same", "confidence": "high" }
  ]
  return run_test_case("test_group_3_edge_cases", eval_input, review_needed, [], checker)


def run_test_group_4(checker: CheckSameSummarizedNameClassifier = None):
  """
  Test Group 4: Location-only differences and legal/format noise
  """
  eval_input = [
    {
      "match_id": "STARBUCKS-01",
      "left": {
        "short_name": "Starbucks Seattle",
        "raw_names": ["STARBUCKS SEATTLE WA"],
        "description": "Coffee shop in Seattle.",
        "amounts": [6.50, 8.20]
      },
      "right": {
        "short_name": "Starbucks Portland",
        "raw_names": ["STARBUCKS PORTLAND OR"],
        "description": "Coffee shop in Portland.",
        "amounts": [5.75]
      }
    },
    {
      "match_id": "BANK-01",
      "left": {
        "short_name": "Chase Bank",
        "raw_names": ["CHASE BANK"],
        "description": "Bank branch transaction.",
        "amounts": [0.0]
      },
      "right": {
        "short_name": "Chase",
        "raw_names": ["CHASE"],
        "description": "Banking and financial services.",
        "amounts": [0.0]
      }
    },
    {
      "match_id": "DOMINOS-01",
      "left": {
        "short_name": "Domino's Pizza LLC",
        "raw_names": ["DOMINOS PIZZA LLC"],
        "description": "Pizza delivery.",
        "amounts": [24.99]
      },
      "right": {
        "short_name": "Domino's Pizza",
        "raw_names": ["DOMINOS PIZZA"],
        "description": "Pizza chain.",
        "amounts": [18.50]
      }
    }
  ]
  review_needed = [
    {
      "match_id": "STARBUCKS-01",
      "reasoning": "Same chain at different cities; location in the name should be ignored for same/different.",
      "result": "same",
      "confidence": "high"
    },
    {
      "match_id": "BANK-01",
      "reasoning": "Chase Bank and Chase are the same institution; branch wording does not make them different.",
      "result": "same",
      "confidence": "high"
    },
    {
      "match_id": "DOMINOS-01",
      "reasoning": "LLC is legal structure and should be ignored; both refer to the same pizza chain.",
      "result": "same",
      "confidence": "high"
    }
  ]
  return run_test_case("test_group_4_misprocessed_short_names", eval_input, review_needed, [], checker)


def run_test_group_5(checker: CheckSameSummarizedNameClassifier = None):
  """
  Test Group 5: Correct result with weak reasoning
  """
  eval_input = [
    {
        "match_id": "WEAK-REASON-01",
        "left": { "short_name": "Walmart Supercenter", "raw_names": ["WALMART SUPERCENTER"], "description": "Large format Walmart store.", "amounts": [120.00] },
        "right": { "short_name": "Walmart", "raw_names": ["Walmart"], "description": "General retailer.", "amounts": [80.00] }
    }
  ]
  review_needed = [
    {
      "match_id": "WEAK-REASON-01",
      "reasoning": "Same.",
      "result": "same",
      "confidence": "high"
    }
  ]
  return run_test_case("test_group_5_weak_reasoning", eval_input, review_needed, [], checker)


def run_multiple_errors_test(checker: CheckSameSummarizedNameClassifier = None):
  """
  Test for multiple errors to check eval_text formatting and online vs. physical rule.
  """
  eval_input = [
    {
      "match_id": "AMZN-01",
      "left": { "short_name": "Amazon.com", "raw_names": ["AMAZON.COM"], "description": "Online retailer.", "amounts": [50.0] },
      "right": { "short_name": "Amazon Go", "raw_names": ["Amazon Go Store"], "description": "Physical convenience store.", "amounts": [12.50] }
    },
    {
      "match_id": "SPOT-01",
      "left": { "short_name": "Spotify Silver Plan", "raw_names": ["Spotify Silver Plan"], "description": "Subscription for music streaming service.", "amounts": [9.99] },
      "right": { "short_name": "Spotify Platinum", "raw_names": ["SPOTIFY PLATINUM"], "description": "Premium subscription for music and podcast streaming.", "amounts": [15.99] }
    }
  ]
  review_needed = [
    { "match_id": "AMZN-01", "reasoning": "These are both Amazon, so they are the same.", "result": "same", "confidence": "medium" },
    { "match_id": "SPOT-01", "reasoning": "Different plans are the same service.", "result": "same", "confidence": "low" }
  ]
  return run_test_case("multiple_errors_test", eval_input, review_needed, [], checker)


def run_bad_reasoning_test(checker: CheckSameSummarizedNameClassifier = None):
  """
  Test for a correct result with a completely nonsensical reasoning.
  """
  eval_input = [
    {
        "match_id": "BAD-REASON-01",
        "left": { "short_name": "MGM Grand", "raw_names": ["MGM Grand Las Vegas"], "description": "Hotel and casino in Las Vegas.", "amounts": [300.0] },
        "right": { "short_name": "MGM Grand Detroit", "raw_names": ["MGM Grand Detroit"], "description": "Hotel and casino in Detroit.", "amounts": [250.0] }
    }
  ]
  review_needed = [
    {
      "match_id": "BAD-REASON-01",
      "reasoning": "The sky is blue and grass is green.",
      "result": "same",
      "confidence": "high"
    }
  ]
  return run_test_case("bad_reasoning_test", eval_input, review_needed, [], checker)


def run_ambiguous_physical_store_test(checker: CheckSameSummarizedNameClassifier = None):
  """
  Test for ambiguity in online vs. physical stores.
  """
  eval_input = [
    {
        "match_id": "NIKE-AMBIGUOUS-01",
        "left": {
            "short_name": "Nike",
            "raw_names": ["Nike Store"],
            "description": "Sportswear retailer.",
            "amounts": [150.0]
        },
        "right": {
            "short_name": "Nike.com",
            "raw_names": ["NIKE.COM"],
            "description": "Online store for sportswear.",
            "amounts": [120.00]
        }
    }
  ]
  review_needed = [
    {
      "match_id": "NIKE-AMBIGUOUS-01",
      "reasoning": "An ambiguous short_name 'Nike' is assumed to be a physical store, which is different from the online store 'Nike.com'.",
      "result": "different",
      "confidence": "high"
    }
  ]
  return run_test_case("ambiguous_physical_store_test", eval_input, review_needed, [], checker)


def run_custom_user_test(checker: CheckSameSummarizedNameClassifier = None):
  """One-off test with user-provided EVAL_INPUT and REVIEW_NEEDED."""
  eval_input = [
    {
      "match_id": "319",
      "left": {
        "short_name": "DraftKings Instant Payments Credit",
        "raw_names": [
          "REAL TIME PAYMENT CREDIT RECD FROM ABA/CONTR BNK-071923909 FROM: DraftKings Instant Payments via Trustly REF: BALSJ3rM1dabe1B4 INFO: TEXT-RmtInf-89DZQ3 IID: 20251216042000314P1BOPFX01076745914 RECD: 21:36:26 TRN: 2145482350GC",
          "REAL TIME PAYMENT CREDIT RECD FROM ABA/CONTR BNK-071923909 FROM: DraftKings Instant Payments via Trustly REF: rYypBMbGT6EUK18q INFO: TEXT-RmtInf-89C30L IID: 20251216042000314P1BOPFX01076710148 RECD: 18:57:58 TRN: 1844742350GA"
        ],
        "description": "Income from an online sports betting and daily fantasy sports platform.\n",
        "amounts": [-107.3, -192.0, -286.36]
      },
      "right": {
        "short_name": "DraftKings",
        "raw_names": [
          "DK*DRAFTKINGSG3R7 6179866744 MA 11/24",
          "DK*DRAFTKINGSMW5K 6179866744 MA 01/17"
        ],
        "description": "This establishment is an online sports betting and daily fantasy sports platform.\n",
        "amounts": [72.63, 217.55, 15.0]
      }
    },
    {
      "match_id": "131",
      "left": {
        "short_name": "Green Valley Golf Range",
        "raw_names": [
          "GREEN VALLEY GOLF RANGE",
          "GREEN VALLEY GOLF RAN HANOVER PARK IL 05/28"
        ],
        "description": "offers a driving range for golf practice, lessons, and equipment",
        "amounts": [21.63, 10.3, 19.57]
      },
      "right": {
        "short_name": "Green Valley Golf",
        "raw_names": ["GREEN VALLEY GOLF RAHANOVER PARK"],
        "description": "Payment for golf-related services or products, such as green fees or pro shop purchases.\n",
        "amounts": [10.3]
      }
    },
    {
      "match_id": "855",
      "left": {
        "short_name": "Goldboys.\ncom",
        "raw_names": [
          "GOLDBOYS GOLDBOYS.\nCOM WY 08/20",
          "GOLDBOYS GOLDBOYS.\nCOM WY 05/31"
        ],
        "description": "online retailer that sells gold coins, bullion, and other precious metals",
        "amounts": [50.0]
      },
      "right": {
        "short_name": "Goldboys",
        "raw_names": [
          "GOLDBOYS GOLDBOYS.\nCOM WY 07/21",
          "GOLDBOYS GOLDBOYS.\nCOM WY 01/19"
        ],
        "description": "Online retailer for men's fashion and accessories.\n",
        "amounts": [45.0, 42.5, 40.0]
      }
    }
  ]
  review_needed = [
    {"match_id": "319", "reasoning": "One name specifies a payment mechanism for the service, the other is the general service name", "result": "different", "confidence": "medium"},
    {"match_id": "131", "reasoning": "The names are highly similar, differing only by the omission of 'Range'", "result": "same", "confidence": "high"},
    {"match_id": "855", "reasoning": "The descriptions indicate different businesses despite similar names", "result": "different", "confidence": "high"}
  ]
  return run_test_case("custom_user_test", eval_input, review_needed, [], checker)


def main(batch: int = 0):
  """Main function to test the SameSummarizedNameClassifier checker"""
  checker = CheckSameSummarizedNameClassifier()
  
  if batch == 0:
    # Run all tests if no specific batch is selected
    run_correct_response(checker)
    run_edge_cases(checker)
    run_test_group_1(checker)
    run_test_group_2(checker)
    run_test_group_3(checker)
    run_test_group_4(checker)
    run_test_group_5(checker)
    run_multiple_errors_test(checker)
    run_bad_reasoning_test(checker)
    run_ambiguous_physical_store_test(checker)
  elif batch == 1:
    run_test_group_1(checker)
  elif batch == 2:
    run_test_group_2(checker)
  elif batch == 3:
    run_test_group_3(checker)
  elif batch == 4:
    run_test_group_4(checker)
  elif batch == 5:
    run_test_group_5(checker)
  elif batch == 6:
    run_multiple_errors_test(checker)
  elif batch == 7:
    run_bad_reasoning_test(checker)
  elif batch == 8:
    run_ambiguous_physical_store_test(checker)
  elif batch == 9:
    run_custom_user_test(checker)
  else:
    print("Invalid batch number. Please choose from 0 to 9.")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Run checker tests in batches')
  parser.add_argument('--batch', type=int, default=0, choices=range(10),
                      help='Batch number to run (1-9). 0 runs all.')
  args = parser.parse_args()
  main(batch=args.batch)
