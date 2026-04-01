from google import genai
from google.genai import types
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are a checker verifying the output of a model that transforms leisure establishment data into structured attributes.

## Input:
- **EVAL_INPUT**: A JSON array of leisure establishments. Each has `id`, `name`, and `description`.
- **PAST_REVIEW_OUTCOMES**: An array of past review outcomes.
- **REVIEW_NEEDED**: The JSON output from the optimizer that needs to be reviewed (array of result objects).

## Output:
Return valid JSON only. Put each top-level key on its own line (line break after each of good_copy, info_correct, eval_text). Example format:
```
{"good_copy": true,
"info_correct": true,
"eval_text": ""}
```

- `good_copy`: True if REVIEW_NEEDED is a valid JSON array and each item has the required fields: `id`, `primary`, `secondary`. Every `id` in REVIEW_NEEDED must exist in EVAL_INPUT. There must be exactly one output item per establishment in EVAL_INPUT.
- `info_correct`: True if the `primary` and `secondary` attributes for each item in REVIEW_NEEDED are correct according to the rules.
- `eval_text`: Empty string when good_copy and info_correct are both True. Otherwise, explain why REVIEW_NEEDED is incorrect. Each line must start with "Establishment <id>: ". **Crucial: The explanation must be self-contained and descriptive (e.g., "contains forbidden generic term 'Leisure'" instead of "violates Rule 3").** One line per erroneous item (max 25 words per line). **NEVER reference rule numbers in your output.**

## Critical Rules for LeisureSpendAddAttributes:
1. `primary`: Every output must have at least one primary attribute. Options:
   - Logistics: Functional infrastructure and travel essentials (eg. flights, car rentals, trains, hotels, Airbnbs, travel insurance, passports, visas).
   - Shows: Ticketed live performances and scheduled events (eg. concerts, festivals, theater, live performances).
   - Attractions: Entry to points of interest or curated experiences (eg. zoos, museums, theme parks, guided city tours, scuba charters).
   - Sports: Physical participation and active recreation venues (eg. bowling alleys, ski resorts, gyms, golf courses).
   - Nightlife: Adult-oriented social consumption and evening entertainment (eg. bars, nightclubs, liquor stores, cannabis dispensaries, lounges).
   - Relaxation: Physical rejuvenation and mental calm (eg. day spas, massage therapy centers, saunas, hot springs).
   - Movies: The film-viewing experience across all platforms. Includes cinema tickets, theater concessions, and digital video streaming services.
   - Apps: Recurring fees or one-time purchases for non-gaming, non-movie, and non-literature digital services. Includes music streaming, productivity tools, and utility applications.
   - Gaming: Digital and interactive software, hardware, or services (eg. video game purchases, online multiplayer subscriptions, in-game content).
   - Literature: Purchase or subscription of written content (eg. physical bookstores, e-book subscriptions, newsstands, magazines).
   - Crafts and Hobbies: Materials and retailers for personal projects and skill-based interests (eg. art supply stores, musical instrument shops, photography equipment).
   - Gear: Durable physical goods for specific leisure pursuits (eg. camping equipment, luggage, specialized technical apparel).

2. `secondary`: Extract highly specific, high-value category tags. Every establishment MUST have at least 3 `secondary` tags.
   - SOURCE MATERIAL ONLY. Tags MUST be solely based on Name and Description provided.
   - DEFINE THE ESTABLISHMENT. Tags are only valid if they define what the establishment IS or DOES.
   - ABSOLUTELY NO STANDALONE GENERIC TERMS. Must prepend a specific descriptor for: "Rental", "Pass", "Admission", "Card", "Fee", "Reservation", "Stay", "Booking", "Ticket", "Venue", "Travel", "Arena", "Concession", "Content", "Online", "Digital", "Service", "Subscription", "Round-trip".
   - ABSOLUTELY NO PRIMARY REUSE. Secondary tags MUST NOT contain any of the primary attribute names (Logistics, Shows, Attractions, Sports, Nightlife, Relaxation, Movies, Apps, Gaming, Literature, Crafts, Hobbies, Gear), unless accompanied by descriptors. e.g. "Japanese Literature" is okay, but "Literature" is not
   - P2P PAYMENTS. For person-to-person payments (e.g., Venmo), only focus on the purpose (e.g., "Dinner", "Gift").
   - SPECIFICITY. eg. prefer "Basketball Game" over "Game".
   - SINGULAR NOUNS. Use singular nouns for `secondary` tags where appropriate.
   - COMPOUND TERMS. Allow compound terms (e.g., "Video game", "Theme park").
   - NO REPETITION. `primary` categories must not be repeated in `secondary`.

## Verification Steps:
1. Check PAST_REVIEW_OUTCOMES for repeated mistakes.
2. Verify good_copy: structure, required fields, one-to-one mapping of IDs.
3. Verify info_correct: Check primary/secondary choices against the rules above.
4. eval_text: Only when incorrect. Each line starts with "Establishment <id>: ". **NEVER reference rule numbers in eval_text. Use descriptive language only.**
"""

class CheckLeisureSpendAddAttributesOptimizer:
  """Handles all Gemini API interactions for checking LeisureSpendAddAttributesOptimizer outputs"""

  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set.")
    self.client = genai.Client(api_key=api_key)

    self.model_name = model_name
    self.top_k = 40
    self.top_p = 0.95
    self.temperature = 0.5
    self.thinking_budget = 0
    self.max_output_tokens = 4096

    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
    ]
    self.system_prompt = SYSTEM_PROMPT

  def generate_response(self, eval_input: list, past_review_outcomes: list, review_needed: list) -> dict:
    """Generate a response using Gemini API for checking optimizer outputs."""
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

    request_text = types.Part.from_text(text=request_text_str)
    contents = [types.Content(role="user", parts=[request_text])]

    generate_content_config = types.GenerateContentConfig(
      top_k=self.top_k,
      top_p=self.top_p,
      temperature=self.temperature,
      max_output_tokens=self.max_output_tokens,
      safety_settings=self.safety_settings,
      system_instruction=[types.Part.from_text(text=self.system_prompt)],
      thinking_config=types.ThinkingConfig(
        thinking_budget=self.thinking_budget,
        include_thoughts=True
      ),
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

    try:
      return json.loads(output_text.strip())
    except json.JSONDecodeError:
      # Fallback for potential markdown or extra text
      json_start = output_text.find('{')
      json_end = output_text.rfind('}') + 1
      if json_start != -1 and json_end > json_start:
        return json.loads(output_text[json_start:json_end])
      raise ValueError(f"Failed to parse JSON: {output_text}")

def run_test_case(test_name: str, eval_input: list, review_needed: list, past_review_outcomes: list = None, checker: 'CheckLeisureSpendAddAttributesOptimizer' = None):
  if past_review_outcomes is None:
    past_review_outcomes = []
  if checker is None:
    checker = CheckLeisureSpendAddAttributesOptimizer()

  print(f"\n{'='*80}\nRunning test: {test_name}\n{'='*80}")
  try:
    result = checker.generate_response(eval_input, past_review_outcomes, review_needed)
    print("Result:")
    print(json.dumps(result, indent=2))
    return result
  except Exception as e:
    print(f"ERROR: {str(e)}")
    return None

def run_correct_response(checker: CheckLeisureSpendAddAttributesOptimizer = None):
  eval_input = [
    {"id": 101, "name": "Netflix", "description": "monthly subscription for streaming movies and tv shows"}
  ]
  review_needed = [
    {"id": 101, "primary": ["Movies"], "secondary": ["Video streaming", "Film content", "TV series subscription"]}
  ]
  return run_test_case("correct_response", eval_input, review_needed, [], checker)

def run_wrong_secondary_count(checker: CheckLeisureSpendAddAttributesOptimizer = None):
  eval_input = [
    {"id": 102, "name": "AMC Theatres", "description": "movie theater."}
  ]
  review_needed = [
    {"id": 102, "primary": ["Movies"], "secondary": ["Cinema"]} # Only 1 tag, needs 3-7
  ]
  return run_test_case("wrong_secondary_count", eval_input, review_needed, [], checker)

def run_forbidden_primary_reuse(checker: CheckLeisureSpendAddAttributesOptimizer = None):
  eval_input = [
    {"id": 103, "name": "Disneyland", "description": "Amusement park with rides and characters"}
  ]
  review_needed = [
    {"id": 103, "primary": ["Attractions"], "secondary": ["Theme park", "Amusement ride", "Attractions venue"]} # "Attractions" is primary
  ]
  return run_test_case("forbidden_primary_reuse", eval_input, review_needed, [], checker)

def run_external_inference(checker: CheckLeisureSpendAddAttributesOptimizer = None):
  eval_input = [
    {"id": 104, "name": "Michaels", "description": "sells art supplies, framing, and seasonal decor"}
  ]
  review_needed = [
    {"id": 104, "primary": ["Crafts and Hobbies"], "secondary": ["Art supply", "Knitting material", "Painting kit"]} # "Knitting" and "Painting" are not in description
  ]
  return run_test_case("external_inference", eval_input, review_needed, [], checker)

def run_generic_terms_without_descriptor(checker: CheckLeisureSpendAddAttributesOptimizer = None):
  eval_input = [
    {"id": 105, "name": "Delta Air Lines", "description": "Airline tickets for a round-trip flight to London."}
  ]
  review_needed = [
    {"id": 105, "primary": ["Logistics"], "secondary": ["Airline flight", "Ticket", "Round-trip travel"]} # "Ticket" is generic and needs a descriptor
  ]
  return run_test_case("generic_terms_without_descriptor", eval_input, review_needed, [], checker)

def main(batch: int = 0):
  checker = CheckLeisureSpendAddAttributesOptimizer()
  if batch == 0:
    run_correct_response(checker)
    run_wrong_secondary_count(checker)
    run_forbidden_primary_reuse(checker)
    run_external_inference(checker)
    run_generic_terms_without_descriptor(checker)
  elif batch == 1:
    run_correct_response(checker)
  elif batch == 2:
    run_wrong_secondary_count(checker)
  elif batch == 3:
    run_forbidden_primary_reuse(checker)
  elif batch == 4:
    run_external_inference(checker)
  elif batch == 5:
    run_generic_terms_without_descriptor(checker)

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch', type=int, default=0)
  args = parser.parse_args()
  main(batch=args.batch)
