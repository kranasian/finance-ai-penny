"""
Optimizer runner for **M:CaptureChatMemory** (Gemini prompt tuning).

Extracts durable user memories from Penny/User chat transcripts with
captured_datetime, expiry_datetime, type, and content.

Run from ``finance-ai-penny`` repo root:

  python3 active_experiments/m_capture_chat_memory_optimizer.py --list
  python3 active_experiments/m_capture_chat_memory_optimizer.py --test 0
  python3 active_experiments/m_capture_chat_memory_optimizer.py --test habits_goals_and_categorization
  python3 active_experiments/m_capture_chat_memory_optimizer.py --test budget_next_month_specific_content
  python3 active_experiments/m_capture_chat_memory_optimizer.py --test all
  python3 active_experiments/m_capture_chat_memory_optimizer.py --test 1 --model gemini-flash-lite-latest
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Any

try:
  from dotenv import load_dotenv
except Exception:
  load_dotenv = None

try:
  from google import genai
  from google.genai import types
except Exception:
  genai = None
  types = None

if load_dotenv is not None:
  load_dotenv()

DEFAULT_MODEL = "gemini-flash-lite-latest"
TEMPLATE_NAME = "M:CaptureChatMemory"
BACK_AND_FORTH_LIMIT = 16
CHUNK_MESSAGE_OVERLAP = 2
_TEST_SEPARATOR = "=" * 72
_SECTION_RULE = "-" * 72

MEMORY_TYPES = (
  "user_facts",
  "life_events",
  "income",
  "goals",
  "spending",
  "categorization",
  "accounts",
  "communication",
)

EXPIRY_BUCKETS = (
  "1_month",
  "3_months",
  "6_months",
  "1_year",
  "2_years",
  "permanent",
)

SYSTEM_PROMPT = """You extract durable user memories from Penny (AI financial advisor) and User chat transcripts.

## Goal
Read the conversation and return note-worthy memories that Penny should remember later to personalize advice. Capture preferences, habits, goals, and personal context the user shared explicitly or clearly implied.

## Input
JSON array of chronological messages (oldest first).

Every message must include:
- `captured_datetime`: timestamp when the message was sent
- `penny_log_id`: message id
- `speaker`: `"Penny"` or `"User"`
- `message`: message text

## What to capture
Record only information that is:
- About the user (not generic financial facts Penny already computed)
- Useful for future personalization
- Stated or clearly implied by the user (not invented)

Good examples:
- Personal facts (family, job, location, pets)
- Spending or shopping habits and preferred merchants
- How the user wants transactions categorized
- Financial goals, timelines, and priorities
- Communication style preferences
- Income or employment context (self-employed, variable pay)
- Account or cash-flow preferences (keeps low checking balance, pays cards from investments)
- Budget priorities or tradeoffs
- Upcoming life events (move, wedding, baby)

## What NOT to capture
- Raw balances, transaction lists, or amounts Penny can re-query from accounts
- One-off questions with no lasting preference ("what did I spend last week?")
- Penny's analysis unless the user affirmed a preference in response
- Duplicate memories for the same unchanged fact; merge into one memory (when a fact is updated later, use Superseding rules instead)
- Sensitive credentials (account numbers, passwords, SSN)

## Memory types
Every output `type` must be exactly one of:
- `user_facts`: identity, family, location, demographics, relationships
- `life_events`: dated or upcoming events affecting finances
- `income`: employment type, pay schedule, variable income notes
- `goals`: savings targets, debt payoff, budget limits, and tradeoffs
- `spending`: recurring spend patterns, timing, channels, and preferred merchants
- `categorization`: how user wants labels applied or corrected; use actual Penny category slugs in content (e.g. `meals_dining_out`, `meals_groceries`), never invented labels like Coffee
- `accounts`: how user moves money between accounts
- `communication`: tone, frequency, channel, level of detail

## Expiry
Set `expiry_datetime` by adding the appropriate bucket to `captured_datetime`:
- `1_month`: short-lived context (trip this weekend, one-time project)
- `3_months`: seasonal or quarterly context
- `6_months`: medium-term plans or habits being established
- `1_year`: annual goals, stable habits, preferences
- `2_years`: long-horizon goals and durable personal facts
- `permanent`: core identity facts unlikely to change (name of spouse, number of children)

Compute `expiry_datetime` as ISO-8601 UTC. For `permanent`, use captured_datetime plus 10 years.

When the user states an explicit timeframe or end date, set `expiry_datetime` to the last moment of that period instead of a generic bucket:
- "for next month" on `2025-08-21` → `2025-09-30T23:59:59Z`
- A trip or event in a named month → last moment of that month or event date

When no end date is stated for an ongoing habit, goal, preference, or durable rule, set `expiry_datetime` to `null` (no expiry).

## Content specificity
- Preserve concrete details the user stated: dollar amounts, merchants, Penny category slugs, dates, months, years, and durations.
- When the user bounds a goal or budget to a timeframe ("for next month", "in December", "until the wedding"), name that period in `content` (e.g. "for the month of Sept 2025"), not a vague recurrence (e.g. "monthly budget").
- Resolve relative time from `captured_datetime` (e.g. "next month" on `2025-08-21` → Sept 2025).
- Do not replace a one-time or bounded target with an ongoing habit unless the user said it is recurring.

## Superseding
When the user updates or contradicts an earlier fact later in the conversation (same topic, same `type`):
- Output the older memory and the newer replacement memory; do not silently drop the older fact.
- Set the older memory's `expiry_datetime` to the newer memory's `captured_datetime` (when the updated fact appeared). Do not use the normal expiry bucket for the superseded memory.
- Set the replacement memory's `expiry_datetime` using the normal expiry rules above (including `null` when open-ended).

## Output (JSON only)
Return a JSON array of memory objects for this conversation window only.

Each object:
- `captured_datetime`: timestamp when the fact appeared
- `expiry_datetime`: timestamp when the memory expires, or `null` when open-ended
- `type`: memory type slug
- `content`: third-person memory sentence
- `penny_log_id`: message id

Return an empty array `[]` when no durable memories are present.
Return only the JSON array. No markdown fences or extra text."""


def _build_output_schema() -> "types.Schema":
  if types is None:
    raise RuntimeError("Install `google-genai` for this optimizer.")
  return types.Schema(
    type=types.Type.ARRAY,
    items=types.Schema(
      type=types.Type.OBJECT,
      required=["captured_datetime", "type", "content", "penny_log_id"],
      properties={
        "captured_datetime": types.Schema(type=types.Type.STRING),
        "expiry_datetime": types.Schema(type=types.Type.STRING, nullable=True),
        "type": types.Schema(type=types.Type.STRING),
        "content": types.Schema(type=types.Type.STRING),
        "penny_log_id": types.Schema(type=types.Type.INTEGER),
      },
    ),
  )


def _print_section_banner(title: str) -> None:
  print(f"\n{_SECTION_RULE}\n{title}\n{_SECTION_RULE}\n")


def _extract_stream_chunk_text(chunk: Any) -> str:
  pieces: list[str] = []
  for cand in getattr(chunk, "candidates", None) or []:
    content = getattr(cand, "content", None)
    if not content:
      continue
    for part in getattr(content, "parts", None) or []:
      if getattr(part, "thought", False):
        continue
      t = getattr(part, "text", None)
      if isinstance(t, str) and t:
        pieces.append(t)
  if pieces:
    return "".join(pieces)
  with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=r".*non-text parts in the response.*")
    agg = getattr(chunk, "text", None)
    return agg if isinstance(agg, str) else ""


def _parse_model_json_array(raw: str) -> list[dict[str, Any]]:
  text = raw.strip()
  if text.startswith("```json"):
    text = text[7:]
  elif text.startswith("```"):
    text = text[3:]
  if text.endswith("```"):
    text = text[:-3]
  parsed = json.loads(text.strip())
  if not isinstance(parsed, list):
    raise ValueError("model output must be a JSON array")
  return parsed


def _validate_conversation_messages(messages: list[dict[str, Any]], path_prefix: str) -> None:
  if not messages:
    raise ValueError(f"{path_prefix} must be a non-empty array")
  for i, msg in enumerate(messages):
    if not isinstance(msg, dict):
      raise ValueError(f"{path_prefix}[{i}] must be an object")
    if msg.get("speaker") not in ("Penny", "User"):
      raise ValueError(f"{path_prefix}[{i}].speaker must be Penny or User")
    if not isinstance(msg.get("message"), str):
      raise ValueError(f"{path_prefix}[{i}].message must be a string")
    captured = msg.get("captured_datetime")
    if not isinstance(captured, str) or not captured.strip():
      raise ValueError(f"{path_prefix}[{i}].captured_datetime is required")
    if msg.get("penny_log_id") is None:
      raise ValueError(f"{path_prefix}[{i}].penny_log_id is required")


def _normalize_input_payload(
  payload: str | list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
  if isinstance(payload, str):
    parsed = json.loads(payload.strip())
  else:
    parsed = payload
  if not isinstance(parsed, list):
    raise ValueError("input must be a JSON array of messages")
  _validate_conversation_messages(parsed, "input")
  return json.dumps(parsed, indent=2), parsed


def _memory_slot_key(item: dict[str, Any]) -> tuple[str, str]:
  return (
    str(item.get("type", "")),
    str(item.get("captured_datetime", "")),
  )


def _compare_memories(
  actual: list[dict[str, Any]],
  ideal: list[dict[str, Any]],
) -> tuple[bool, str]:
  if not isinstance(actual, list) or not isinstance(ideal, list):
    return False, "output must be arrays"
  for row in actual:
    if not isinstance(row, dict):
      return False, "each memory must be an object"
    for field in ("captured_datetime", "type", "content"):
      if field not in row or not str(row.get(field, "")).strip():
        return False, f"missing or empty field: {field}"
    if "expiry_datetime" not in row:
      return False, "missing field: expiry_datetime"
    if row.get("penny_log_id") is None:
      return False, "missing penny_log_id"
    if row.get("type") not in MEMORY_TYPES:
      return False, f"invalid type: {row.get('type')}"
  ideal_keys = {_memory_slot_key(r) for r in ideal if isinstance(r, dict)}
  actual_keys = {_memory_slot_key(r) for r in actual if isinstance(r, dict)}
  missing = ideal_keys - actual_keys
  if missing:
    return False, f"missing expected memories: {sorted(missing)}"
  extra = actual_keys - ideal_keys
  if extra:
    return False, f"unexpected extra memories: {sorted(extra)}"
  return True, f"matched {len(ideal_keys)} memories by type and captured_datetime"


def _parse_test_json_array(raw: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
  if isinstance(raw, str):
    parsed = json.loads(raw.strip())
  else:
    parsed = raw
  if not isinstance(parsed, list):
    raise ValueError("test case JSON must be an array")
  return parsed


TEST_CASES = [
  {
    "batch": 1,
    "name": "habits_goals_and_categorization",
    "input": """
[
  {
    "captured_datetime": "2025-09-01T10:00:00Z",
    "penny_log_id": 100001,
    "speaker": "Penny",
    "message": "Hey! I sorted most of your November transactions. Want help with the rest?"
  },
  {
    "captured_datetime": "2025-09-01T10:03:00Z",
    "penny_log_id": 100002,
    "speaker": "User",
    "message": "Yes please. Also I usually grocery shop at Sprouts on Sundays."
  },
  {
    "captured_datetime": "2025-09-01T10:06:00Z",
    "speaker": "Penny",
    "message": "Got it! I'll keep an eye on Sprouts trips. Anything else?",
    "penny_log_id": 200001
  },
  {
    "captured_datetime": "2025-09-01T10:09:00Z",
    "speaker": "User",
    "message": "We're saving $15,000 for a Hawaii trip next summer.",
    "penny_log_id": 200002
  },
  {
    "captured_datetime": "2025-09-01T10:12:00Z",
    "speaker": "Penny",
    "message": "Love that goal! I can track progress toward $15,000.",
    "penny_log_id": 200003
  },
  {
    "captured_datetime": "2025-09-01T10:15:00Z",
    "speaker": "User",
    "message": "When you see Starbucks, put it under dining out, not groceries.",
    "penny_log_id": 200004
  },
  {
    "captured_datetime": "2025-09-01T10:18:00Z",
    "speaker": "Penny",
    "message": "Dining out for Starbucks \u2014 noted.",
    "penny_log_id": 200005
  },
  {
    "captured_datetime": "2025-09-01T10:21:00Z",
    "speaker": "User",
    "message": "I dine out mostly Mondays and Tuesdays because my partner works late other nights.",
    "penny_log_id": 200006
  },
  {
    "captured_datetime": "2025-09-01T10:24:00Z",
    "speaker": "Penny",
    "message": "Makes sense. I'll remember your Monday/Tuesday dining pattern.",
    "penny_log_id": 200007
  },
  {
    "captured_datetime": "2025-09-01T10:27:00Z",
    "speaker": "User",
    "message": "I'm self-employed so my income varies month to month.",
    "penny_log_id": 200008
  },
  {
    "captured_datetime": "2025-09-01T10:30:00Z",
    "speaker": "Penny",
    "message": "Thanks for sharing \u2014 I'll factor variable income into forecasts.",
    "penny_log_id": 200009
  },
  {
    "captured_datetime": "2025-09-01T10:33:00Z",
    "speaker": "User",
    "message": "I keep checking low and pay my cards from my brokerage account.",
    "penny_log_id": 200010
  },
  {
    "captured_datetime": "2025-09-01T10:36:00Z",
    "speaker": "Penny",
    "message": "Helpful context on how you move money.",
    "penny_log_id": 200011
  },
  {
    "captured_datetime": "2025-09-01T10:39:00Z",
    "speaker": "User",
    "message": "Please keep messages short \u2014 I read these on the go.",
    "penny_log_id": 200012
  },
  {
    "captured_datetime": "2025-09-01T10:42:00Z",
    "speaker": "Penny",
    "message": "Short and sweet from here on!",
    "penny_log_id": 200013
  },
  {
    "captured_datetime": "2025-09-01T10:45:00Z",
    "speaker": "User",
    "message": "Thanks!",
    "penny_log_id": 200014
  }
]
""",
    "output": """
[
  {
    "captured_datetime": "2025-09-01T10:03:00Z",
    "expiry_datetime": null,
    "type": "spending",
    "content": "User usually grocery shops at Sprouts on Sundays.",
    "penny_log_id": 200001
  },
  {
    "captured_datetime": "2025-09-01T10:09:00Z",
    "expiry_datetime": "2026-03-01T10:09:00Z",
    "type": "goals",
    "content": "User is saving $15,000 for a Hawaii trip next summer.",
    "penny_log_id": 200002
  },
  {
    "captured_datetime": "2025-09-01T10:15:00Z",
    "expiry_datetime": null,
    "type": "categorization",
    "content": "User wants Starbucks transactions categorized as meals_dining_out, not meals_groceries.",
    "penny_log_id": 200003
  },
  {
    "captured_datetime": "2025-09-01T10:21:00Z",
    "expiry_datetime": null,
    "type": "spending",
    "content": "User usually dines out on Mondays and Tuesdays because their partner works late on other nights.",
    "penny_log_id": 200004
  },
  {
    "captured_datetime": "2025-09-01T10:27:00Z",
    "expiry_datetime": null,
    "type": "income",
    "content": "User is self-employed and has variable month-to-month income.",
    "penny_log_id": 200005
  },
  {
    "captured_datetime": "2025-09-01T10:33:00Z",
    "expiry_datetime": null,
    "type": "accounts",
    "content": "User keeps checking account balances low and pays credit cards from a brokerage account.",
    "penny_log_id": 200006
  },
  {
    "captured_datetime": "2025-09-01T10:39:00Z",
    "expiry_datetime": null,
    "type": "communication",
    "content": "User prefers short messages because they read them on the go.",
    "penny_log_id": 200007
  }
]
""",
  },
  {
    "batch": 2,
    "name": "family_life_event_no_memories",
    "input": """
[
  {
    "captured_datetime": "2025-09-05T16:00:00Z",
    "speaker": "Penny",
    "message": "Your grocery spend last week was $412. Want a category breakdown?",
    "penny_log_id": 200001
  },
  {
    "captured_datetime": "2025-09-05T16:03:00Z",
    "speaker": "User",
    "message": "Sure, send it.",
    "penny_log_id": 200002
  },
  {
    "captured_datetime": "2025-09-05T16:06:00Z",
    "speaker": "Penny",
    "message": "Groceries $280, household $90, personal care $42.",
    "penny_log_id": 200003
  },
  {
    "captured_datetime": "2025-09-05T16:09:00Z",
    "speaker": "User",
    "message": "What was my total eating out?",
    "penny_log_id": 200004
  },
  {
    "captured_datetime": "2025-09-05T16:12:00Z",
    "speaker": "Penny",
    "message": "Eating out was $186 across 6 transactions.",
    "penny_log_id": 200005
  },
  {
    "captured_datetime": "2025-09-05T16:15:00Z",
    "speaker": "User",
    "message": "Ok thanks.",
    "penny_log_id": 200006
  },
  {
    "captured_datetime": "2025-09-05T16:18:00Z",
    "speaker": "Penny",
    "message": "Any questions about those numbers?",
    "penny_log_id": 200007
  },
  {
    "captured_datetime": "2025-09-05T16:21:00Z",
    "speaker": "User",
    "message": "Nope, all good.",
    "penny_log_id": 200008
  }
]
""",
    "output": """
[]
""",
  },
  {
    "batch": 3,
    "name": "user_facts_and_life_event",
    "input": """
[
  {
    "captured_datetime": "2025-08-20T09:00:00Z",
    "speaker": "Penny",
    "message": "I noticed a few large transfers this week \u2014 everything look right?",
    "penny_log_id": 200001
  },
  {
    "captured_datetime": "2025-08-20T09:05:00Z",
    "speaker": "User",
    "message": "Yes, we're moving to Austin in March for my wife's new job.",
    "penny_log_id": 200002
  },
  {
    "captured_datetime": "2025-08-20T09:10:00Z",
    "speaker": "Penny",
    "message": "Exciting move! I can help budget relocation costs.",
    "penny_log_id": 200003
  },
  {
    "captured_datetime": "2025-08-20T09:15:00Z",
    "speaker": "User",
    "message": "We have two kids, ages 7 and 10.",
    "penny_log_id": 200004
  },
  {
    "captured_datetime": "2025-08-20T09:20:00Z",
    "speaker": "Penny",
    "message": "I'll keep family context in mind for planning.",
    "penny_log_id": 200005
  },
  {
    "captured_datetime": "2025-08-20T09:25:00Z",
    "speaker": "User",
    "message": "Rent in Austin will be about $2,800 so tighten the fun budget until we move.",
    "penny_log_id": 200006
  },
  {
    "captured_datetime": "2025-08-20T09:30:00Z",
    "speaker": "Penny",
    "message": "I'll treat $2,800 rent as the target and watch discretionary spend.",
    "penny_log_id": 200007
  },
  {
    "captured_datetime": "2025-08-20T09:35:00Z",
    "speaker": "User",
    "message": "Call me Dan, not Daniel.",
    "penny_log_id": 200008
  },
  {
    "captured_datetime": "2025-08-20T09:40:00Z",
    "speaker": "Penny",
    "message": "Dan it is!",
    "penny_log_id": 200009
  }
]
""",
    "output": """
[
  {
    "captured_datetime": "2025-08-20T09:05:00Z",
    "expiry_datetime": "2026-02-20T09:05:00Z",
    "type": "life_events",
    "content": "User is moving to Austin in March because their wife started a new job there.",
    "penny_log_id": 200001
  },
  {
    "captured_datetime": "2025-08-20T09:15:00Z",
    "expiry_datetime": "2035-08-20T09:15:00Z",
    "type": "user_facts",
    "content": "User has two children, ages 7 and 10.",
    "penny_log_id": 200002
  },
  {
    "captured_datetime": "2025-08-20T09:25:00Z",
    "expiry_datetime": "2026-02-20T09:25:00Z",
    "type": "goals",
    "content": "User wants discretionary fun spending reduced until they move because Austin rent will be about $2,800.",
    "penny_log_id": 200003
  },
  {
    "captured_datetime": "2025-08-20T09:35:00Z",
    "expiry_datetime": "2035-08-20T09:35:00Z",
    "type": "user_facts",
    "content": "User prefers to be called Dan, not Daniel.",
    "penny_log_id": 200004
  }
]
""",
  },
  {
    "batch": 4,
    "name": "goal_supersede_expiry",
    "input": """
[
  {
    "captured_datetime": "2025-09-10T14:00:00Z",
    "speaker": "Penny",
    "message": "Any savings goals you want me to track?",
    "penny_log_id": 200001
  },
  {
    "captured_datetime": "2025-09-10T14:05:00Z",
    "speaker": "User",
    "message": "We're saving $15,000 for a Hawaii trip next summer.",
    "penny_log_id": 200002
  },
  {
    "captured_datetime": "2025-09-10T14:10:00Z",
    "speaker": "Penny",
    "message": "I'll track progress toward $15,000.",
    "penny_log_id": 200003
  },
  {
    "captured_datetime": "2025-09-10T14:15:00Z",
    "speaker": "User",
    "message": "Actually bump that to $20,000 \u2014 flights went up.",
    "penny_log_id": 200004
  },
  {
    "captured_datetime": "2025-09-10T14:18:00Z",
    "speaker": "Penny",
    "message": "Updated to $20,000 for Hawaii.",
    "penny_log_id": 200005
  }
]
""",
    "output": """
[
  {
    "captured_datetime": "2025-09-10T14:05:00Z",
    "expiry_datetime": "2025-09-10T14:15:00Z",
    "type": "goals",
    "content": "User is saving $15,000 for a Hawaii trip next summer.",
    "penny_log_id": 200001
  },
  {
    "captured_datetime": "2025-09-10T14:15:00Z",
    "expiry_datetime": "2026-03-10T14:15:00Z",
    "type": "goals",
    "content": "User is saving $20,000 for a Hawaii trip next summer.",
    "penny_log_id": 200002
  }
]
""",
  },
  {
    "captured_datetime": "2025-09-10T14:15:00Z",
    "expiry_datetime": "2026-03-10T14:15:00Z",
    "type": "goals",
    "content": "User is saving $20,000 for a Hawaii trip next summer."
  }
]
""",
  },
  {
    "batch": 5,
    "name": "budget_next_month_specific_content",
    "input": """
[
  {
    "captured_datetime": "2025-08-21T09:10:00Z",
    "speaker": "Penny",
    "message": "Want help setting a spending target?"
  },
  {
    "captured_datetime": "2025-08-21T09:11:42Z",
    "speaker": "User",
    "message": "I'd like to set a budget of $500 on food for next month."
  }
]
""",
    "output": """
[
  {
    "captured_datetime": "2025-08-21T09:11:42Z",
    "expiry_datetime": "2025-09-30T23:59:59Z",
    "type": "goals",
    "content": "User set a budget of $500 for the month of Sept 2025 for the meals_dining_out category."
  }
]
""",
  },
]


class CaptureChatMemoryOptimizer:
  def __init__(
    self,
    model_name: str = DEFAULT_MODEL,
    *,
    thinking_budget: int = 0,
    max_output_tokens: int = 4096,
  ):
    if genai is None or types is None:
      raise RuntimeError(
        "Gemini client dependencies not available. Install `google-genai` (and optionally `python-dotenv`)."
      )
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
      raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY must be set.")
    self.client = genai.Client(api_key=api_key)
    self.model_name = model_name
    self.thinking_budget = thinking_budget
    self.max_output_tokens = max_output_tokens
    self.temperature = 0.2
    self.top_p = 0.95
    self.system_prompt = SYSTEM_PROMPT
    self.output_schema = _build_output_schema()
    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    ]

  def generate_response(
    self,
    payload: str | list[dict[str, Any]],
  ) -> list[dict[str, Any]]:
    request_text, _ = _normalize_input_payload(payload)
    t = types
    contents = [t.Content(role="user", parts=[t.Part.from_text(text=request_text)])]
    cfg_kwargs: dict[str, Any] = {
      "temperature": self.temperature,
      "top_p": self.top_p,
      "max_output_tokens": self.max_output_tokens,
      "safety_settings": self.safety_settings,
      "system_instruction": [t.Part.from_text(text=self.system_prompt)],
      "response_schema": self.output_schema,
      "response_mime_type": "application/json",
    }
    if self.thinking_budget > 0:
      cfg_kwargs["thinking_config"] = t.ThinkingConfig(thinking_budget=self.thinking_budget)
    cfg = t.GenerateContentConfig(**cfg_kwargs)
    output_text = ""
    for chunk in self.client.models.generate_content_stream(
      model=self.model_name,
      contents=contents,
      config=cfg,
    ):
      output_text += _extract_stream_chunk_text(chunk)
    if not output_text.strip():
      raise ValueError("Empty response from model.")
    return _parse_model_json_array(output_text)


def run_test(
  test_name_or_index_or_dict: int | str | dict,
  optimizer: CaptureChatMemoryOptimizer | None = None,
) -> list[dict[str, Any]] | None:
  if isinstance(test_name_or_index_or_dict, dict):
    tc = test_name_or_index_or_dict
  elif isinstance(test_name_or_index_or_dict, int):
    idx = test_name_or_index_or_dict
    if not (0 <= idx < len(TEST_CASES)):
      return None
    tc = TEST_CASES[idx]
  else:
    tc = next((t for t in TEST_CASES if t["name"] == test_name_or_index_or_dict), None)
    if tc is None:
      return None

  print(f"# Test: **{tc['name']}**  [{TEMPLATE_NAME}]\n")
  if optimizer is None:
    optimizer = CaptureChatMemoryOptimizer()
  input_text, _ = _normalize_input_payload(tc["input"])
  _print_section_banner("Input")
  print(input_text)
  result = optimizer.generate_response(tc["input"])
  _print_section_banner("Output")
  print(json.dumps(result, indent=2))
  if tc.get("output") is not None:
    ideal_output = _parse_test_json_array(tc["output"])
    _print_section_banner("Ideal output (reference)")
    print(tc["output"].strip())
    ok, detail = _compare_memories(result, ideal_output)
    _print_section_banner("Memory match")
    print("PASS" if ok else "FAIL")
    print(detail)
  return result


def run_tests(
  test_names_or_indices: list[int | str] | None = None,
  optimizer: CaptureChatMemoryOptimizer | None = None,
) -> list[list[dict[str, Any]] | None]:
  if test_names_or_indices is None:
    items: list[int | str] = list(range(len(TEST_CASES)))
  else:
    items = test_names_or_indices
  results: list[list[dict[str, Any]] | None] = []
  for i, item in enumerate(items):
    if i:
      print(f"\n{_TEST_SEPARATOR}\n")
    results.append(run_test(item, optimizer))
  return results


def main(
  test: str | None = None,
  *,
  model: str | None = None,
  list_tests: bool = False,
) -> None:
  if list_tests:
    print(f"Template: {TEMPLATE_NAME}\n")
    for i, tc in enumerate(TEST_CASES):
      print(f"  {i}: {tc['name']} (batch {tc.get('batch')})")
    return

  if test is None:
    print("Usage:")
    print("  List tests:     python3 active_experiments/m_capture_chat_memory_optimizer.py --list")
    print("  Run by index:   python3 active_experiments/m_capture_chat_memory_optimizer.py --test 0")
    print("  Run by name:    python3 active_experiments/m_capture_chat_memory_optimizer.py --test habits_goals_and_categorization")
    print("  Bounded budget: python3 active_experiments/m_capture_chat_memory_optimizer.py --test budget_next_month_specific_content")
    print("  Run all tests:  python3 active_experiments/m_capture_chat_memory_optimizer.py --test all")
    return

  kw: dict[str, Any] = {}
  if model is not None:
    kw["model_name"] = model
  optimizer = CaptureChatMemoryOptimizer(**kw)

  if test.strip().lower() == "all":
    run_tests(optimizer=optimizer)
    return

  if test.isdigit():
    run_test(int(test), optimizer)
    return

  run_test(test, optimizer)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=f"Run {TEMPLATE_NAME} optimizer tests")
  parser.add_argument("--test", type=str, default=None, help="Test index, name, or 'all'")
  parser.add_argument("--model", type=str, default=None)
  parser.add_argument("--list", action="store_true", help="List available test cases")
  args = parser.parse_args()
  main(test=args.test, model=args.model, list_tests=args.list)
