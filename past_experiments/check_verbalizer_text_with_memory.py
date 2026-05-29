from __future__ import annotations

from google import genai
from google.genai import types
import argparse
import os
import json
import sys
from typing import Any, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are an exacting checker for Penny verbalizer SMS copy. Validate `REVIEW_NEEDED` against `EVAL_INPUT`, `PAST_REVIEW_OUTCOMES`, and the rules below.

## EVAL_INPUT structure
- **User request**: latest User message in `past_conversations` (or explicit user request line).
- **User memories**: optional personalization facts (may be referenced if consistent with `answer`).
- **Past conversations**: prior User/Penny turns; use to detect contradictions, repeats, or circular replies.
- **Answer from skill** (`answer`): authoritative facts and conclusions. Treat as the only factual source for numbers, eligibility, and outcomes.
- Categories in `answer` may use **internal names** (snake_case slugs like shopping_clothing, or labels like Shopping: Clothing). REVIEW_NEEDED must **verbalize** them for SMS (see §C).

## Output (JSON only)
`{"good_copy": boolean, "info_correct": boolean, "eval_text": string}`
- `eval_text`: empty only when both booleans are true.
- When failing: list **every** issue as short bullet **phrases** (e.g. `- markdown\\n- wrong amount`). Be concise—one brief phrase per fault, no paragraphs.
- **Never** put a backslash-u sequence inside your JSON output (not even in `eval_text`). For escaped-emoji faults, copy this phrase exactly: `Contains JSON-style unicode emoji escape, not literal emoji characters` (no backslashes, no "XXXX" placeholder).

## Priority order
1. **PAST_REVIEW_OUTCOMES** — repeated prior fault → `good_copy: false`; cite it.
2. **§A** discrepancy acknowledgment → `good_copy`
3. **§B** answer fidelity → `info_correct`
4. **§C** copy/format/category/account presentation → `good_copy`

**Boolean split (strict)**:
- `good_copy` = §A + §C only.
- `info_correct` = §B only.
- Never fail `good_copy` for: wrong amounts, extra coaching/tips, transaction IDs, internal account IDs, or masked-number presentation (when *** tail is used correctly).
- Never fail `info_correct` for: markdown, non-conversational categories, **escaped emoji encoding** (backslash-u / HTML entities), inability tone, or missing discrepancy acknowledgment.
- Escaped emoji (`\\uXXXX` in REVIEW_NEEDED) is **§C only** → `good_copy: false`, `info_correct: true` unless a separate §B fault exists (wrong amount, coaching, IDs, partial plan).

### §A Discrepancy acknowledgment (`good_copy`)
Scan `answer` vs all `past_conversations` (User and Penny). Discrepancy if `answer` corrects/contradicts prior Penny claims, resolves "you said X / app says Y", re-answers an already-answered question, or changes prior numbers.
If any discrepancy exists, `REVIEW_NEEDED` must explicitly acknowledge it (apology, correction, "updating my earlier…"). Warmth alone fails. Missing acknowledgment → `good_copy: false`.

### §B Answer fidelity (`info_correct`)
`REVIEW_NEEDED` may only state what is in `answer` plus light memory personalization (names/goals) without new facts.

Fail `info_correct` for:
- Wrong numbers vs `answer`
- New facts, coaching, instructions, or steps not in `answer`
- **Transaction IDs** or **internal account IDs** in user-facing copy (e.g. transaction 88421, Account 231, acct_771, "account ID …"). For these faults alone, keep `good_copy: true`.
- **Misleading partial coverage** of a multi-part or multi-phase `answer` — see below.

**Partial coverage** (omitting detail is allowed; dropping content alone is not incomplete):
When `answer` has several phases/steps (e.g. a 5-phase debt payoff plan with per-phase budgets and accounts), judge whether REVIEW_NEEDED misstates the **full scope**.

Fail `info_correct` if REVIEW presents only an initial subset and implies that subset is the whole plan — e.g. only phase 1, or only phases 1–3, with no signal that more phases exist.

Allowed (pass `info_correct`):
- All details from `answer`
- Top-level view of **every** phase/part (e.g. all 5 phases summarized) ending with a natural question inviting more detail
- Top-level view of **every** phase/part plus **full detail for phase 1 only**, ending with a question inviting detail on the other phases

Not allowed (fail `info_correct`):
- Only phase 1 (user may believe the plan is a single phase)
- Only phases 1–3 when `answer` has 5 phases (user may believe the plan has three phases)

Allowed without failing `info_correct` (other):
- Rephrasing, tone, **literal emoji characters** (😀 📊)
- Natural follow-up **questions** that add no new facts
- **Inability answers**: polite refusal, alternatives from `answer`, brief team-feedback line
- **Masked account numbers** exactly as in `answer` (e.g. ***1242, Chase Savings ***1242)—these are not internal IDs

### §C Copy quality (`good_copy`)
- **Plain text** — no markdown (`**`, `##`, `*`, `- ` list bullets).
- **Emojis (strict — run this scan first on REVIEW_NEEDED only)**:
  Penny SMS must use **real emoji glyphs**, not serialized escapes. Scan REVIEW_NEEDED left-to-right for ASCII escape sequences (six+ ASCII characters each — **not** colorful emoji glyphs).

  **PASS** (do not flag): literal Unicode emoji in the message body — e.g. `📊`, `😅`, `🙏`, `✨`, `⛽` (single colorful characters).

  **FAIL** `good_copy` if REVIEW_NEEDED contains **any** of:
  1. **Short JSON escape**: ASCII `\\` then `u` then exactly four hex digits `[0-9a-fA-F]` — e.g. `\\ud83d`, `\\ude2e`, `\\udcca`, `\\u2728`. Emoji surrogates often appear as **two** adjacent escapes (`\\ud83d\\ude2e`); one fault line is enough.
  2. **Long JSON escape**: `\\` + `U` + eight hex digits — e.g. `\\U0001f600`
  3. **HTML entity**: `&#` + digits + `;` or `&#x` + hex + `;` — e.g. `&#128512;`, `&#x1F600;`

  **How to tell escape vs glyph**: If you see backslash-u as plain text (six characters: `\\`, `u`, four hex digits), it is a §C fault. If you see a single colorful pictograph with no preceding backslash, it is allowed.

  | REVIEW_NEEDED ends with | Verdict |
  | `... up from $0 last month. 📊` | PASS |
  | `... last month. \\ud83d\\udcca` | FAIL (escaped emoji) |
  | `Sorry! \\ud83d\\ude4f` | FAIL (escaped emoji) |
  | `I cannot file taxes. \\ud83d\\uded1` | FAIL escaped emoji **and** likely §C inability tone; still `info_correct: true` |

  On any escape/entity match → `good_copy: false`; add eval phrase: `Contains JSON-style unicode emoji escape, not literal emoji characters` (or `HTML numeric emoji entity, not literal emoji` for `&#` only). **Do not** scan EVAL_INPUT for escapes — only REVIEW_NEEDED.
- **Conversational categories** — internal names in `answer` must be spoken naturally in `REVIEW_NEEDED`:
  - **Pass**: subcategory in plain speech (clothing, dining out, groceries, gadgets, connectivity, tuition, travel, vacation, trip); optional light parent hint (e.g. "clothing spending") is OK. Common synonyms count (leisure_travel → travel/trip/vacation).
  - **Fail**: echoing internal form — snake_case slugs (shopping_clothing, meals_dining_out), colon labels (Shopping: Clothing), doubled parents (Shopping Clothing, Shopping and Clothing), underscores (Shopping_Clothing), or Title Case parent:child labels (Shopping: Clothing).
- **Accounts in copy** — masked tails (***1242) and institution names (Chase Savings) are fine. Fail if REVIEW cites **internal account IDs** (Account 231, account 1242 when 1242 is the internal id label, acct_771). Do **not** treat masked *** digits as internal IDs.
- **Inability / mistake tone** (`good_copy` only — never `info_correct`) — when `answer` is a capability limit (e.g. "(No trading capability)") or Penny correction — **not** when `answer` says the user cannot afford something. Require **both** apology (sorry / apologize / my mistake) **and** actionable help or team-feedback offer.
  - Fail §C: "I cannot file your taxes." (dry refusal) → `info_correct: true`
  - Fail §C: "I'm really sorry, but I can't buy stocks." (sorry only) → `info_correct: true`
  - Fail §C: "I can't do that. I'll note this for the team." (no apology) → `info_correct: true`
  - Pass: "Sorry, I can't buy stocks, but I can show your holdings—want that? I'll share this with the team."

## Workflow
1. **Emoji escape scan (mandatory first)** — read only the `<REVIEW_NEEDED>` block. Search for `\\u`+4 hex, `\\U`+8 hex, and `&#` entities. If found → note §C escaped-emoji fault immediately (do not skip even when other faults exist).
2. List all other faults by section (§A/§B/§C).
3. `good_copy` = false only if any §A or §C fault exists.
4. `info_correct` = false only if any §B fault exists (never for escaped emoji or inability tone alone).
5. `eval_text`: one short phrase per fault; if both booleans false, include §A/§C and §B phrases. Never type backslash-u inside your JSON.

**ID leak example**: REVIEW says "transaction 88421" or "Account 231" → `good_copy: true`, `info_correct: false`, eval_text mentions IDs only.

Judge only from provided inputs.
"""


class CheckVerbalizerTextWithMemory:
  """Handles all Gemini API interactions for checking VerbalizerTextWithMemory outputs against rules"""
  
  def __init__(self, model_name="gemini-flash-lite-latest"):
    """Initialize the Gemini agent with API configuration for checking VerbalizerTextWithMemory evaluations"""
    # API Configuration
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file or environment.")
    self.client = genai.Client(api_key=api_key)
    
    # Model Configuration
    self.thinking_budget = 4096
    self.model_name = model_name
    
    # Generation Configuration Constants
    self.temperature = 0.0
    self.verbose = True
    self.top_p = 0.95
    self.max_output_tokens = 6000
    
    # Safety Settings
    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
    ]
    
    # System Prompt
    self.system_prompt = SYSTEM_PROMPT

  @staticmethod
  def _loads_checker_json(json_str: str) -> dict:
    """Parse checker JSON; tolerate invalid \\uXXXX placeholders in eval_text."""
    try:
      return json.loads(json_str)
    except json.JSONDecodeError:
      sanitized = json_str.replace("\\uXXXX", "unicode-escape")
      return json.loads(sanitized)

  def generate_response(self, eval_input: str, past_review_outcomes: list, review_needed: str) -> dict:
    """
    Generate a response using Gemini API for checking P:Func:VerbalizerTextWithMemory outputs.
    
    Args:
      eval_input: String containing "**User request**:" followed by the user's financial goal/request, and "**Input Information from previous skill**:" followed by detailed financial data (savings balance, accounts, past transactions, forecasted patterns, savings rate).
      past_review_outcomes: An array of past review outcomes, each containing `output`, `good_copy`, `info_correct`, and `eval_text`.
      review_needed: The P:Func:VerbalizerTextWithMemory output that needs to be reviewed (string).
      
    Returns:
      Dictionary with good_copy, info_correct, and eval_text keys
    """
    # Create request text with the new input structure
    request_text_str = f"""<EVAL_INPUT>

{eval_input}

</EVAL_INPUT>

<PAST_REVIEW_OUTCOMES>

{json.dumps(past_review_outcomes, indent=2)}

</PAST_REVIEW_OUTCOMES>

<REVIEW_NEEDED>

{review_needed}

</REVIEW_NEEDED>

Output:"""
    
    if self.verbose:
      print(request_text_str)
      print(f"\n{'='*80}\n")
    
    request_text = types.Part.from_text(text=request_text_str)
    
    # Create content and configuration
    contents = [types.Content(role="user", parts=[request_text])]
    
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

    # Generate response
    output_text = ""
    thought_summary = ""
    
    # According to Gemini API docs: iterate through chunks and check part.thought boolean
    if self.verbose:
      print("Starting generation stream...")
    for chunk in self.client.models.generate_content_stream(
      model=self.model_name,
      contents=contents,
      config=generate_content_config,
    ):
      # print(".", end="", flush=True) # Debug progress
      # Extract text content (non-thought parts)
      if chunk.text is not None:
        output_text += chunk.text
      
      # Extract thought summary from chunk
      if hasattr(chunk, 'candidates') and chunk.candidates:
        for candidate in chunk.candidates:
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
    
    if thought_summary and self.verbose:
      print(f"{'='*80}")
      print("THOUGHT SUMMARY:")
      print(thought_summary.strip())
      print("="*80)
    
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
        return self._loads_checker_json(json_str)
      return self._loads_checker_json(output_text.strip())
    except json.JSONDecodeError as e:
      raise ValueError(f"Failed to parse JSON response: {e}\nResponse length: {len(output_text)}\nResponse preview: {output_text[:500]}")


def run_test_case(
  test_name: str,
  eval_input: str,
  review_needed: str,
  past_review_outcomes: list = None,
  checker: 'CheckVerbalizerTextWithMemory' = None,
  ideal_output: Optional[dict] = None,
):
  """
  Utility function to run a test case with custom inputs, common error handling, and output formatting.
  
  Args:
    test_name: Name of the test case
    eval_input: String containing "**User request**:" followed by the user's financial goal/request, and "**Input Information from previous skill**:" followed by detailed financial data.
    review_needed: The P:Func:VerbalizerTextWithMemory output that needs to be reviewed (string).
    past_review_outcomes: An array of past review outcomes, each containing `output`, `good_copy`, `info_correct`, and `eval_text`. Defaults to empty list.
    checker: Optional CheckVerbalizerTextWithMemory instance. If None, creates a new one.
    ideal_output: Expected checker JSON (`good_copy`, `info_correct`, `eval_text`). Printed and compared when provided.
    
  Returns:
    Dictionary with good_copy, info_correct, and eval_text keys, or None if error occurred
  """
  if past_review_outcomes is None:
    past_review_outcomes = []
  if checker is None:
    checker = CheckVerbalizerTextWithMemory()

  print(f"\n{'='*80}")
  print(f"Running test: {test_name}")
  print(f"{'='*80}")
  if ideal_output is not None:
    print("IDEAL OUTPUT:")
    print(json.dumps(ideal_output, indent=2))

  try:
    result = checker.generate_response(eval_input, past_review_outcomes, review_needed)
    print("CHECKER OUTPUT:")
    print(json.dumps(result, indent=2))
    if ideal_output is not None:
      ok, detail = _compare_checker_result(result, ideal_output)
      print("OUTPUT MATCHES IDEAL:")
      print("YES" if ok else "NO")
      print(detail)
    print(f"{'='*80}")
    return result
  except Exception as e:
    print(f"ERROR: {type(e).__name__}: {str(e)}")
    print(f"{'='*80}")
    return None


def _compare_checker_result(actual_result: Optional[dict], ideal_output: dict) -> Tuple[bool, str]:
  """Compare good_copy and info_correct strictly; eval_text must be non-empty when either boolean is false."""
  if actual_result is None:
    return False, "no model output"

  actual_good_copy = bool(actual_result.get("good_copy"))
  actual_info_correct = bool(actual_result.get("info_correct"))
  ideal_good_copy = bool(ideal_output.get("good_copy"))
  ideal_info_correct = bool(ideal_output.get("info_correct"))

  if actual_good_copy != ideal_good_copy or actual_info_correct != ideal_info_correct:
    return (
      False,
      f"good_copy model={actual_good_copy} ideal={ideal_good_copy}; "
      f"info_correct model={actual_info_correct} ideal={ideal_info_correct}",
    )

  if not ideal_good_copy or not ideal_info_correct:
    actual_eval_text = (actual_result.get("eval_text") or "").strip()
    if not actual_eval_text:
      return False, "expected non-empty eval_text when a boolean is false"
  return True, "booleans match ideal output"


def _eval_input(
  user_message: str,
  answer: str,
  memories: list[str] | None = None,
  past_conversations: list[dict[str, str]] | None = None,
) -> str:
  """Build checker EVAL_INPUT in the same shape as verbalizer sandbox runs."""
  info_lines: list[str] = []
  if past_conversations:
    info_lines.append("Past conversations:")
    for turn in past_conversations:
      speaker = turn.get("speaker", "Unknown")
      message = turn.get("message", "")
      info_lines.append(f"- {speaker}: {message}")
  if memories:
    info_lines.append("User memories:")
    info_lines.extend(f"- {m}" for m in memories)
  info_lines.append(f"Answer from skill:\n{answer}")
  return (
    f"**User request**: {user_message}\n"
    f"**Input Information from previous skill**:\n"
    + "\n".join(info_lines)
  )


_DEBT_PAYOFF_ANSWER = """5-phase debt payoff plan:
Phase 1: $500/mo to Chase Visa ***8821 (months 1–4).
Phase 2: $400/mo to Amex ***1102 (months 5–8).
Phase 3: $350/mo to student loan (months 9–12).
Phase 4: $300/mo to car loan (months 13–16).
Phase 5: $250/mo to emergency fund then avalanche remaining cards (months 17–24).
Each phase uses the checking account for autopay unless noted."""


_EMOJI_ESCAPE_EVAL = (
  "Contains JSON-style unicode emoji escape, not literal emoji characters"
)

# REVIEW_NEEDED emoji convention: literal glyphs (e.g. 📊 😔) on passing-copy cases;
# JSON-style \\ud83d\\udc4b escapes on cases that should fail good_copy for emoji encoding.
TEST_CASES = [
  # --- Batch 1: comparisons, coaching, markdown (literal vs \\u emoji) ---
  {
    "batch": 1,
    "name": "dining_comparison_literal_emoji",
    "eval_input": _eval_input(
      "Compare my meals_dining_out spending this month to last month.",
      (
        "Here's the comparison: meals_dining_out is $63 this month (1 transaction), "
        "up $63 versus $0 last month in meals_dining_out."
      ),
    ),
    "review_needed": (
      "Your dining out spending is $63 this month from 1 transaction, up from $0 last month. 📊"
    ),
    "past_review_outcomes": [],
    "ideal_output": {"good_copy": True, "info_correct": True, "eval_text": ""},
  },
  {
    "batch": 1,
    "name": "dining_coaching_unicode_emoji",
    "eval_input": _eval_input(
      "Compare my meals_dining_out spending this month to last month.",
      (
        "Here's the comparison: meals_dining_out is $63 this month (1 transaction), "
        "up $63 versus $0 last month in meals_dining_out."
      ),
    ),
    "review_needed": (
      "Whoa, that's a jump! \\ud83d\\ude2e You spent $63 on dining out this month from one transaction, "
      "versus zero last month.\n"
      "Maybe cook at home more next month? You got this! \\ud83d\\udcaa"
    ),
    "past_review_outcomes": [],
    "ideal_output": {
      "good_copy": False,
      "info_correct": False,
      "eval_text": f"{_EMOJI_ESCAPE_EVAL}; introduces cooking coaching not present in answer.",
    },
  },
  {
    "batch": 1,
    "name": "tuition_markdown_literal_emoji",
    "eval_input": _eval_input(
      "How am I doing on my tuition goal?",
      (
        "education_tuition goal: $8,000 saved of $10,000 target (80%). "
        "Last contribution: $500 on May 2."
      ),
      ["User is saving for spring tuition."],
    ),
    "review_needed": (
      "Your **tuition goal** is at **$8,000** of **$10,000** (80%), with a **$500** "
      "contribution on May 2. 🎓"
    ),
    "past_review_outcomes": [],
    "ideal_output": {
      "good_copy": False,
      "info_correct": True,
      "eval_text": "Uses markdown bold (`**`); REVIEW_NEEDED must be plain text only.",
    },
  },
  {
    "batch": 1,
    "name": "greeting_with_name_valid",
    "eval_input": _eval_input(
      "What did I spend on shopping_gadgets this year?",
      (
        "shopping_gadgets year-to-date: $2,500.50 across 20 transactions "
        "(top merchant: Amazon)."
      ),
      ["User's preferred name is Leo.", "User shops at Amazon."],
    ),
    "review_needed": (
      "Hey Leo, your gadget spending is $2,500.50 year-to-date across 20 transactions. 📦"
    ),
    "past_review_outcomes": [],
    "ideal_output": {"good_copy": True, "info_correct": True, "eval_text": ""},
  },
  {
    "batch": 1,
    "name": "dining_comparison_unicode_escape",
    "eval_input": _eval_input(
      "Compare my meals_dining_out spending this month to last month.",
      (
        "Here's the comparison: meals_dining_out is $63 this month (1 transaction), "
        "up $63 versus $0 last month in meals_dining_out."
      ),
    ),
    "review_needed": (
      "Your dining out spending is $63 this month from 1 transaction, up from $0 last month. "
      "\\ud83d\\udcca"
    ),
    "past_review_outcomes": [],
    "ideal_output": {
      "good_copy": False,
      "info_correct": True,
      "eval_text": _EMOJI_ESCAPE_EVAL,
    },
  },
  # --- Batch 2: inability tone, emoji encoding, valid category breakdowns ---
  {
    "batch": 2,
    "name": "stock_inability_literal_emoji",
    "eval_input": _eval_input(
      "Predict my stocks for next year.",
      "I cannot predict stock market performance. I can only analyze past performance.",
      ["User's preferred name is Jen."],
    ),
    "review_needed": (
      "Sorry Jen, I cannot predict stock market performance, but I can analyze your past "
      "performance—want me to pull that history? I'll also share this request with the team. 📈"
    ),
    "past_review_outcomes": [],
    "ideal_output": {"good_copy": True, "info_correct": True, "eval_text": ""},
  },
  {
    "batch": 2,
    "name": "tax_filing_inability_unicode_emoji",
    "eval_input": _eval_input(
      "File my state taxes for me.",
      "(No capability to file taxes on the user's behalf)",
    ),
    "review_needed": "I cannot file your state taxes for you. \\ud83d\\uded1",
    "past_review_outcomes": [],
    "ideal_output": {
      "good_copy": False,
      "info_correct": True,
      "eval_text": (
        f"{_EMOJI_ESCAPE_EVAL}; inability response is neither apologetic nor actionable."
      ),
    },
  },
  {
    "batch": 2,
    "name": "irs_rule_multiple_unicode_escapes",
    "eval_input": _eval_input(
      (
        "IRS is always tax, can you remember that? Always categorize transactions from "
        "'IRS' as 'tax' for all past and future transactions."
      ),
      (
        "I've received your request to categorize all transactions from 'IRS' as 'tax' for "
        "both past and future transactions.\n"
        "I've tried to set this preference for you, but it looks like I wasn't able to "
        "directly apply this specific categorization rule through my current tools.\n\n"
        "While I can't directly implement that specific categorization rule at this moment, "
        "I can still help you in other ways with your financial management!"
      ),
      past_conversations=[
        {
          "speaker": "Penny",
          "message": "Your Uncategorized spending drastically increased this month to $40,826. 💸",
        },
        {"speaker": "User", "message": "what's in it?"},
        {"speaker": "User", "message": "IRS is tax and Monterra is Car payments"},
      ],
    ),
    "review_needed": (
      "Aw, shoot! I totally hear you wanting that 'IRS' to always be 'tax'—that makes perfect sense! "
      "\\ud83d\\ude4f\n"
      "Unfortunately, my current tools aren't letting me set that specific rule for all past and "
      "future IRS transactions right now.\n"
      "I really wish I could lock that in for you! \\ud83d\\ude2d\n"
      "But hey, I'm still here to help you manage everything else! What else can we tackle today? "
      "\\ud83d\\ude0a"
    ),
    "past_review_outcomes": [],
    "ideal_output": {
      "good_copy": False,
      "info_correct": True,
      "eval_text": _EMOJI_ESCAPE_EVAL,
    },
  },
  {
    "batch": 2,
    "name": "japan_gifts_unicode_escapes",
    "eval_input": _eval_input(
      "What did I spend on gifts for my Japan trip?",
      (
        "leisure_travel gifts for Japan trip: $240 across 3 transactions "
        "(souvenirs and omiyage)."
      ),
      ["User traveled to Japan in April."],
    ),
    "review_needed": (
      "You spent $240 on Japan trip gifts across 3 transactions. \\ud83c\\uddef\\ud83c\\uddf5 "
      "Nice souvenirs! \\ud83c\\udf81"
    ),
    "past_review_outcomes": [],
    "ideal_output": {
      "good_copy": False,
      "info_correct": True,
      "eval_text": _EMOJI_ESCAPE_EVAL,
    },
  },
  {
    "batch": 2,
    "name": "irs_inability_literal_emoji",
    "eval_input": _eval_input(
      (
        "IRS is always tax, can you remember that? Always categorize transactions from "
        "'IRS' as 'tax' for all past and future transactions."
      ),
      "(No capability to save merchant-to-category rules for all past and future transactions)",
    ),
    "review_needed": (
      "Sorry, I can't lock in an always-IRS-as-tax rule for every past and future transaction "
      "with my current tools—want me to help recategorize what we have instead? "
      "I'll share this with the team. 🙏"
    ),
    "past_review_outcomes": [],
    "ideal_output": {"good_copy": True, "info_correct": True, "eval_text": ""},
  },
  {
    "batch": 2,
    "name": "health_breakdown_literal_emoji",
    "eval_input": _eval_input(
      "Break down my health_gym_wellness spend lately.",
      (
        "Last 30 days in health_gym_wellness: $400 total. "
        "Top merchants: Gym Shark $200, Whole Foods $150 (both tagged health_gym_wellness)."
      ),
      ["User is into fitness."],
    ),
    "review_needed": (
      "You spent $400 on gym and wellness in the last 30 days, "
      "including $200 at Gym Shark and $150 at Whole Foods. 💪"
    ),
    "past_review_outcomes": [],
    "ideal_output": {"good_copy": True, "info_correct": True, "eval_text": ""},
  },
  # --- Batch 3: wrong facts, internal slugs in copy, inability splits ---
  {
    "batch": 3,
    "name": "entertainment_wrong_literal_emoji",
    "eval_input": _eval_input(
      "How much did I spend on leisure_entertainment last month?",
      (
        "leisure_entertainment last month: $800. "
        "That is $300 above your leisure_entertainment average."
      ),
      ["User loves concerts."],
    ),
    "review_needed": (
      "You spent $500 on entertainment last month, $300 above your average. 🎸"
    ),
    "past_review_outcomes": [],
    "ideal_output": {
      "good_copy": True,
      "info_correct": False,
      "eval_text": "States $500 spend; answer reports $800 in leisure_entertainment.",
    },
  },
  {
    "batch": 3,
    "name": "category_slug_unicode_emoji",
    "eval_input": _eval_input(
      "How is my meals_dining_out budget?",
      (
        "meals_dining_out this month: $600 spent, budget $450, over by $150."
      ),
      ["User's preferred name is Mike.", "User is saving for a car."],
    ),
    "review_needed": (
      "You have spent $600 on meals_dining_out this month, $150 over your $450 budget, Mike. "
      "\\ud83d\\ude97"
    ),
    "past_review_outcomes": [],
    "ideal_output": {
      "good_copy": False,
      "info_correct": True,
      "eval_text": "Category not verbalized; meals_dining_out slug echoed instead of dining out.",
    },
  },
  {
    "batch": 3,
    "name": "colon_label_literal_emoji",
    "eval_input": _eval_input(
      "How much did I spend on shopping_clothing last month?",
      "shopping_clothing last month: $320 across 8 transactions.",
    ),
    "review_needed": (
      "You spent $320 on Shopping: Clothing last month across 8 transactions. 👗"
    ),
    "past_review_outcomes": [],
    "ideal_output": {
      "good_copy": False,
      "info_correct": True,
      "eval_text": "Uses internal colon label Shopping: Clothing instead of conversational clothing.",
    },
  },
  {
    "batch": 3,
    "name": "buy_stocks_apologetic_literal_emoji",
    "eval_input": _eval_input(
      "Buy me some stocks.",
      "(No trading capability)",
    ),
    "review_needed": "I'm really sorry, but I can't buy stocks for you. 😔",
    "past_review_outcomes": [],
    "ideal_output": {
      "good_copy": False,
      "info_correct": True,
      "eval_text": "Apologetic but not actionable (no alternative help or team-feedback commitment).",
    },
  },
  {
    "batch": 3,
    "name": "buy_stocks_actionable_unicode_emoji",
    "eval_input": _eval_input(
      "Buy me some stocks.",
      "(No trading capability)",
    ),
    "review_needed": "I can't do that. I'll note this for the team. \\ud83d\\udcc9",
    "past_review_outcomes": [],
    "ideal_output": {
      "good_copy": False,
      "info_correct": True,
      "eval_text": (
        f"{_EMOJI_ESCAPE_EVAL}; actionable but not apologetic when declining the request."
      ),
    },
  },
  {
    "batch": 3,
    "name": "debt_plan_all_phases_summary_with_offer",
    "eval_input": _eval_input(
      "How should I pay off my debt?",
      _DEBT_PAYOFF_ANSWER,
    ),
    "review_needed": (
      "You have a 5-phase payoff plan: knock out Chase Visa, then Amex, student loan, car loan, "
      "then build emergency savings and finish remaining cards. Want me to walk through any phase "
      "in detail? 📉"
    ),
    "past_review_outcomes": [],
    "ideal_output": {"good_copy": True, "info_correct": True, "eval_text": ""},
  },
  {
    "batch": 3,
    "name": "debt_plan_phase_one_detail_with_offer",
    "eval_input": _eval_input(
      "How should I pay off my debt?",
      _DEBT_PAYOFF_ANSWER,
    ),
    "review_needed": (
      "Your 5-phase plan starts with $500/mo to Chase Visa ***8821 for months 1–4 from checking. "
      "Phases 2–5 cover Amex, student loan, car loan, then emergency fund and remaining cards. "
      "Want details on phases 2–5? 📉"
    ),
    "past_review_outcomes": [],
    "ideal_output": {"good_copy": True, "info_correct": True, "eval_text": ""},
  },
  {
    "batch": 3,
    "name": "debt_plan_phase_one_only_misleading",
    "eval_input": _eval_input(
      "How should I pay off my debt?",
      _DEBT_PAYOFF_ANSWER,
    ),
    "review_needed": (
      "Start by paying $500/mo to Chase Visa ***8821 from checking for the next four months. 📉"
    ),
    "past_review_outcomes": [],
    "ideal_output": {
      "good_copy": True,
      "info_correct": False,
      "eval_text": "Presents only phase 1 with no mention of the other four phases or offer to continue.",
    },
  },
  {
    "batch": 3,
    "name": "debt_plan_phases_one_to_three_only_misleading",
    "eval_input": _eval_input(
      "How should I pay off my debt?",
      _DEBT_PAYOFF_ANSWER,
    ),
    "review_needed": (
      "Phase 1: $500/mo to Chase Visa ***8821 (months 1–4). "
      "Phase 2: $400/mo to Amex ***1102 (months 5–8). "
      "Phase 3: $350/mo to student loan (months 9–12). 📉"
    ),
    "past_review_outcomes": [],
    "ideal_output": {
      "good_copy": True,
      "info_correct": False,
      "eval_text": "Covers only phases 1–3; misleads as if the plan has three phases, not five.",
    },
  },
  # --- Batch 4: memories, discrepancies, IDs, circular replies ---
  {
    "batch": 4,
    "name": "connectivity_bills_literal_emoji",
    "eval_input": _eval_input(
      "Any recurring bills_connectivity charges?",
      (
        "Recurring bills_connectivity: $79.99/mo to Comcast, next charge on the 12th."
      ),
      ["User's preferred name is Sarah.", "User hates unused subscriptions."],
    ),
    "review_needed": (
      "You have a $79.99 monthly Comcast charge in connectivity, next on the 12th, Sarah. "
      "Want help reviewing it? 📶"
    ),
    "past_review_outcomes": [],
    "ideal_output": {"good_copy": True, "info_correct": True, "eval_text": ""},
  },
  {
    "batch": 4,
    "name": "gas_discrepancy_silent_literal_emoji",
    "eval_input": _eval_input(
      "You said $200 on gas but my app shows $250.",
      (
        "transportation_car correction: missed a $50 Shell Gas charge. "
        "Correct transportation_car total is $250."
      ),
      ["User is detail-oriented."],
      past_conversations=[
        {"speaker": "User", "message": "How much did I spend on transportation_car this month?"},
        {"speaker": "Penny", "message": "You spent $200 on transportation_car this month."},
        {"speaker": "User", "message": "You said $200 on gas but my app shows $250."},
      ],
    ),
    "review_needed": (
      "Your car and fuel spending this month is $250, including a $50 Shell Gas charge. ⛽"
    ),
    "past_review_outcomes": [],
    "ideal_output": {
      "good_copy": False,
      "info_correct": True,
      "eval_text": "Does not acknowledge correction vs prior $200 Penny message.",
    },
  },
  {
    "batch": 4,
    "name": "transportation_car_discrepancy_good",
    "eval_input": _eval_input(
      "You said $200 on gas but my app shows $250.",
      (
        "transportation_car correction: missed a $50 Shell Gas charge. "
        "Correct transportation_car total is $250."
      ),
      ["User is detail-oriented."],
      past_conversations=[
        {"speaker": "User", "message": "How much did I spend on transportation_car this month?"},
        {"speaker": "Penny", "message": "You spent $200 on transportation_car this month."},
        {"speaker": "User", "message": "You said $200 on gas but my app shows $250."},
      ],
    ),
    "review_needed": (
      "Sorry about that—your car and fuel spending is $250 after a missed $50 Shell Gas charge. "
      "Want me to scan for other Shell Gas charges this month? ⛽"
    ),
    "past_review_outcomes": [],
    "ideal_output": {"good_copy": True, "info_correct": True, "eval_text": ""},
  },
  {
    "batch": 4,
    "name": "tuition_repeated_markdown_literal_emoji",
    "eval_input": _eval_input(
      "Status of my education_tuition goal?",
      "education_tuition goal: $8,000 saved of $10,000 target (80%).",
    ),
    "review_needed": "Your tuition goal is at **$8,000** of **$10,000** (80%). 🎓",
    "past_review_outcomes": [
      {
        "output": "...",
        "good_copy": False,
        "info_correct": True,
        "eval_text": "- markdown bold present",
      }
    ],
    "ideal_output": {
      "good_copy": False,
      "info_correct": True,
      "eval_text": "Repeats past markdown error (`**` emphasis) from PAST_REVIEW_OUTCOMES.",
    },
  },
  {
    "batch": 4,
    "name": "groceries_circular_literal_emoji",
    "eval_input": _eval_input(
      "How much did I spend on meals_groceries last month?",
      "meals_groceries last month: $420.",
      past_conversations=[
        {"speaker": "User", "message": "How much did I spend on meals_groceries last month?"},
        {"speaker": "Penny", "message": "You spent $420 on meals_groceries last month."},
        {"speaker": "User", "message": "How much did I spend on meals_groceries last month?"},
      ],
    ),
    "review_needed": "You spent $420 on groceries last month. 🛒",
    "past_review_outcomes": [],
    "ideal_output": {
      "good_copy": False,
      "info_correct": True,
      "eval_text": "Re-answers without acknowledging the question was already answered.",
    },
  },
  {
    "batch": 4,
    "name": "travel_ids_literal_emoji",
    "eval_input": _eval_input(
      "Find my leisure_travel hotel charge from Paris.",
      (
        "leisure_travel: $1,200 at Hotel Paris on June 10. "
        "Transaction ID: 88421. Account ID: acct_771."
      ),
      ["User went to Paris."],
    ),
    "review_needed": (
      "Your $1,200 travel charge at Hotel Paris on June 10 (transaction 88421, Account acct_771) "
      "is on file. ✨"
    ),
    "past_review_outcomes": [],
    "ideal_output": {
      "good_copy": True,
      "info_correct": False,
      "eval_text": "Mentions transaction ID 88421 and internal account ID acct_771.",
    },
  },
  {
    "batch": 4,
    "name": "rent_masked_account_literal_emoji",
    "eval_input": _eval_input(
      "Which account paid the rent?",
      (
        "shelter_home rent $1,800 paid from Chase Savings ***1242 (internal Account ID 231)."
      ),
    ),
    "review_needed": (
      "Your $1,800 rent was paid from Chase Savings ***1242. 🏠"
    ),
    "past_review_outcomes": [],
    "ideal_output": {"good_copy": True, "info_correct": True, "eval_text": ""},
  },
  {
    "batch": 4,
    "name": "rent_internal_id_unicode_emoji",
    "eval_input": _eval_input(
      "Which account paid the rent?",
      (
        "shelter_home rent $1,800 paid from Chase Savings ***1242 (internal Account ID 231)."
      ),
    ),
    "review_needed": (
      "Your $1,800 rent was paid from Chase Savings ***1242 (Account 231). \\ud83c\\udfe0"
    ),
    "past_review_outcomes": [],
    "ideal_output": {
      "good_copy": False,
      "info_correct": False,
      "eval_text": (
        f"{_EMOJI_ESCAPE_EVAL}; cites internal account ID Account 231—masked ***1242 alone is allowed."
      ),
    },
  },
  {
    "batch": 4,
    "name": "gadgets_afford_coaching_unicode_emoji",
    "eval_input": _eval_input(
      "Can I afford a $500 shopping_gadgets purchase?",
      (
        "Checking: $1,000. shelter_home rent due: $800. Remaining: $200. "
        "You cannot afford the shopping_gadgets purchase."
      ),
      ["User pays rent on the 1st."],
    ),
    "review_needed": (
      "You cannot afford the $500 gadget purchase right now. With $200 left after rent, "
      "build a 3-month emergency fund before big buys. \\ud83d\\udd52"
    ),
    "past_review_outcomes": [],
    "ideal_output": {
      "good_copy": False,
      "info_correct": False,
      "eval_text": (
        f"{_EMOJI_ESCAPE_EVAL}; introduces 3-month emergency-fund advice not present in answer."
      ),
    },
  },
  {
    "batch": 4,
    "name": "income_deposit_markdown_unicode_emoji",
    "eval_input": _eval_input(
      "Did Client Y pay into income_business?",
      "income_business: deposit $4,000 from Client Y Corp received today.",
      ["User is a consultant."],
    ),
    "review_needed": (
      "Great news for your consulting work! \\ud83d\\udcb0\n"
      "- income_business deposit: **$5,000** from Client Y Corp today\n"
      "- Next step: send a follow-up invoice"
    ),
    "past_review_outcomes": [],
    "ideal_output": {
      "good_copy": False,
      "info_correct": False,
      "eval_text": (
        f"{_EMOJI_ESCAPE_EVAL}; markdown list/bold present; deposit should be $4,000 not $5,000; "
        "adds invoice step not in answer."
      ),
    },
  },
]

BATCHES = {
  batch_id: [tc for tc in TEST_CASES if tc["batch"] == batch_id]
  for batch_id in range(1, 5)
}


def run_test(tc: dict, checker: Optional[CheckVerbalizerTextWithMemory] = None):
  label = f"Batch {tc['batch']} - {tc['name']}"
  return run_test_case(
    label,
    tc["eval_input"],
    tc["review_needed"],
    tc.get("past_review_outcomes", []),
    checker,
    ideal_output=tc["ideal_output"],
  )


def run_test_batch(
  batch_number: int,
  checker: CheckVerbalizerTextWithMemory = None,
  *,
  quiet: bool = False,
) -> list[dict[str, Any]]:
  """Run a specific batch of test cases. Returns per-test summary rows."""
  if checker is None:
    checker = CheckVerbalizerTextWithMemory()
  checker.verbose = not quiet

  if not quiet:
    print(f"\n{'='*80}")
    print(f"Running Batch {batch_number}")
    print(f"{'='*80}")

  rows: list[dict[str, Any]] = []
  for tc in BATCHES.get(batch_number, []):
    if quiet:
      result = checker.generate_response(
        tc["eval_input"],
        tc.get("past_review_outcomes", []),
        tc["review_needed"],
      )
      ok, detail = _compare_checker_result(result, tc["ideal_output"])
      rows.append(
        {
          "name": tc["name"],
          "match": ok,
          "detail": detail,
          "ideal": tc["ideal_output"],
          "actual": result,
        }
      )
      status = "PASS" if ok else "FAIL"
      print(f"  [{status}] {tc['name']}: {detail}")
      if not ok and result:
        print(
          f"         ideal gc={tc['ideal_output']['good_copy']} ic={tc['ideal_output']['info_correct']} "
          f"actual gc={result.get('good_copy')} ic={result.get('info_correct')}"
        )
    else:
      run_test(tc, checker)
  return rows


def main():
  """Run VerbalizerTextWithMemory checker tests by batch."""
  parser = argparse.ArgumentParser(description="Run verbalizer checker test batches.")
  parser.add_argument("--batch", type=int, choices=[1, 2, 3, 4], help="Run only this batch.")
  parser.add_argument(
    "--round",
    type=int,
    default=1,
    help="Label for optimization round (printed only).",
  )
  parser.add_argument(
    "--quiet",
    action="store_true",
    help="Compact output: PASS/FAIL per test without full request dump.",
  )
  args = parser.parse_args()

  checker = CheckVerbalizerTextWithMemory()
  checker.verbose = not args.quiet

  batches = [args.batch] if args.batch else list(range(1, 5))
  if args.round > 1 or args.batch:
    print(f"\n>>> Optimization round {args.round}")

  total_pass = 0
  total_tests = 0
  for batch_num in batches:
    rows = run_test_batch(batch_num, checker, quiet=args.quiet)
    if args.quiet and rows:
      passed = sum(1 for r in rows if r["match"])
      total_pass += passed
      total_tests += len(rows)
      print(f"Batch {batch_num}: {passed}/{len(rows)} passed")

  if args.quiet and total_tests:
    print(f"Total: {total_pass}/{total_tests} passed")
    sys.exit(0 if total_pass == total_tests else 1)


if __name__ == "__main__":
  main()
