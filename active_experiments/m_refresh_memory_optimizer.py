"""
Optimizer runner for **M:RefreshMemory** (Gemini prompt tuning).

Reconciles active user memories with newly captured memory batches from
M:CaptureChatMemory. Detects superseded rows and updates expiry_datetime.

Run from ``finance-ai-penny`` repo root:

  python3 active_experiments/m_refresh_memory_optimizer.py --list
  python3 active_experiments/m_refresh_memory_optimizer.py --test 0
  python3 active_experiments/m_refresh_memory_optimizer.py --test active_superseded_by_capture
  python3 active_experiments/m_refresh_memory_optimizer.py --test all
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
TEMPLATE_NAME = "M:RefreshMemory"
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

SYSTEM_PROMPT = """You reconcile active user memories with newly captured memory batches from Penny chat.

## Goal
Compare `active_memories` (currently stored facts) with `captured_memory_batches` (one or more batches of memories extracted from chat). Determine which active memories are superseded by newer captured facts and return every memory row that must be inserted or updated in storage.

## Input
JSON object:
- `active_memories`: memories currently active in storage (status active, expiry after today). Each item:
  - `memory_item_id`, `captured_datetime`, `type`, `content`
- `captured_memory_batches`: array of batches. Each batch is an array of memories from one chat conversation chunk (oldest batch first). Every captured memory has:
  - `captured_datetime`, `expiry_datetime` (or `null` when open-ended), `type`, `content`, `penny_log_id`

Process batches in order. Later batches may supersede earlier captured rows or active rows.

## Superseding rules
When a newer memory updates or contradicts an older memory about the same topic (same `type`, same underlying fact or goal):
- Include the older memory in output with `expiry_datetime` set to the newer memory's `captured_datetime` (not the normal expiry bucket).
- Include the newer memory with its original `expiry_datetime` from capture.
- Preserve `memory_item_id` on active rows being updated. Omit `memory_item_id` on newly captured rows to insert.

Match supersession across:
- An `active_memories` row and a captured memory
- A captured memory in an earlier batch and a captured memory in a later batch

Do not supersede unrelated memories that merely share a `type`.

## Output (JSON only)
Return a JSON array of memory objects to upsert. Each object:
- `captured_datetime`, `expiry_datetime` (or `null` when open-ended), `type`, `content`, `action`
- `memory_item_id`: include only on `action: "update"`
- `penny_log_id`: include only on `action: "insert"`

Include:
- Every active row whose `expiry_datetime` must change because it was superseded
- Every captured memory that is new or replaces another row (including superseded captured rows with shortened expiry)

Omit:
- Active rows unchanged by any captured batch
- Duplicate captured rows already represented unchanged in `active_memories`

Return `[]` when nothing needs to change.
Return only the JSON array. No markdown fences or extra text."""


def _build_output_schema() -> "types.Schema":
  if types is None:
    raise RuntimeError("Install `google-genai` for this optimizer.")
  memory_schema = types.Schema(
    type=types.Type.OBJECT,
    required=[
      "captured_datetime",
      "type",
      "content",
      "action",
    ],
    properties={
      "memory_item_id": types.Schema(type=types.Type.INTEGER),
      "captured_datetime": types.Schema(type=types.Type.STRING),
      "expiry_datetime": types.Schema(type=types.Type.STRING, nullable=True),
      "type": types.Schema(type=types.Type.STRING),
      "content": types.Schema(type=types.Type.STRING),
      "penny_log_id": types.Schema(type=types.Type.INTEGER),
      "action": types.Schema(type=types.Type.STRING),
    },
  )
  return types.Schema(
    type=types.Type.ARRAY,
    items=memory_schema,
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
  return [row for row in parsed if isinstance(row, dict)]


def _validate_active_memory_row(row: dict[str, Any], path: str) -> None:
  if not isinstance(row, dict):
    raise ValueError(f"{path} must be an object")
  if row.get("memory_item_id") is None:
    raise ValueError(f"{path}.memory_item_id is required")
  for field in ("captured_datetime", "type", "content"):
    if field not in row or not str(row.get(field, "")).strip():
      raise ValueError(f"{path}.{field} is required")
  if row.get("type") not in MEMORY_TYPES:
    raise ValueError(f"{path}.type is invalid")


def _validate_captured_memory_row(row: dict[str, Any], path: str) -> None:
  if not isinstance(row, dict):
    raise ValueError(f"{path} must be an object")
  for field in ("captured_datetime", "type", "content"):
    if field not in row or not str(row.get(field, "")).strip():
      raise ValueError(f"{path}.{field} is required")
  if "expiry_datetime" not in row:
    raise ValueError(f"{path}.expiry_datetime is required")
  if row.get("type") not in MEMORY_TYPES:
    raise ValueError(f"{path}.type is invalid")
  if row.get("penny_log_id") is None:
    raise ValueError(f"{path}.penny_log_id is required")


def _normalize_input_payload(
  payload: str | dict[str, Any],
) -> tuple[str, dict[str, Any]]:
  if isinstance(payload, str):
    parsed = json.loads(payload.strip())
  else:
    parsed = payload
  if not isinstance(parsed, dict):
    raise ValueError("input must be a JSON object")
  active = parsed.get("active_memories")
  batches = parsed.get("captured_memory_batches")
  if not isinstance(active, list):
    raise ValueError("input.active_memories must be an array")
  if not isinstance(batches, list):
    raise ValueError("input.captured_memory_batches must be an array")
  for i, row in enumerate(active):
    _validate_active_memory_row(row, f"active_memories[{i}]")
  for bi, batch in enumerate(batches):
    if not isinstance(batch, list):
      raise ValueError(f"captured_memory_batches[{bi}] must be an array")
    for mi, row in enumerate(batch):
      _validate_captured_memory_row(row, f"captured_memory_batches[{bi}][{mi}]")
  normalized = {
    "active_memories": active,
    "captured_memory_batches": batches,
  }
  return json.dumps(normalized, indent=2), normalized


def _memory_result_key(row: dict[str, Any]) -> tuple[Any, ...]:
  item_id = row.get("memory_item_id")
  if item_id is not None:
    return ("id", int(item_id))
  return (
    "new",
    str(row.get("type", "")),
    str(row.get("captured_datetime", "")),
    str(row.get("content", "")),
  )


def _compare_refresh_output(
  actual: list[dict[str, Any]],
  ideal: list[dict[str, Any]],
) -> tuple[bool, str]:
  if not isinstance(actual, list) or not isinstance(ideal, list):
    return False, "output must be arrays"
  actual_rows = actual
  ideal_rows = ideal
  for i, row in enumerate(actual_rows):
    if not isinstance(row, dict):
      return False, f"memories[{i}] must be an object"
    for field in ("captured_datetime", "type", "content", "action"):
      if field not in row:
        return False, f"memories[{i}] missing {field}"
    if "expiry_datetime" not in row:
      return False, f"memories[{i}] missing expiry_datetime"
    if row.get("action") not in ("insert", "update"):
      return False, f"memories[{i}].action is invalid"
    if row.get("action") == "insert" and row.get("penny_log_id") is None:
      return False, f"memories[{i}] missing penny_log_id"
    if row.get("action") == "update" and row.get("memory_item_id") is None:
      return False, f"memories[{i}] missing memory_item_id"
  ideal_by_key = {_memory_result_key(r): r for r in ideal_rows if isinstance(r, dict)}
  actual_by_key = {_memory_result_key(r): r for r in actual_rows if isinstance(r, dict)}
  missing = set(ideal_by_key) - set(actual_by_key)
  if missing:
    return False, f"missing expected memories: {sorted(missing)}"
  extra = set(actual_by_key) - set(ideal_by_key)
  if extra:
    return False, f"unexpected extra memories: {sorted(extra)}"
  for key, ideal_row in ideal_by_key.items():
    actual_row = actual_by_key[key]
    for field in ("captured_datetime", "type", "content", "action"):
      if str(actual_row.get(field, "")) != str(ideal_row.get(field, "")):
        return False, f"{key} field {field} mismatch"
    actual_expiry = actual_row.get("expiry_datetime")
    ideal_expiry = ideal_row.get("expiry_datetime")
    if actual_expiry != ideal_expiry:
      return False, f"{key} field expiry_datetime mismatch"
    if ideal_row.get("action") == "insert":
      if int(actual_row.get("penny_log_id")) != int(ideal_row.get("penny_log_id")):
        return False, f"{key} penny_log_id mismatch"
    if ideal_row.get("memory_item_id") is not None:
      if int(actual_row.get("memory_item_id")) != int(ideal_row.get("memory_item_id")):
        return False, f"{key} memory_item_id mismatch"
  return True, f"matched {len(ideal_by_key)} memories"


def _parse_test_json_array(raw: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
  if isinstance(raw, str):
    parsed = json.loads(raw.strip())
  else:
    parsed = raw
  if not isinstance(parsed, list):
    raise ValueError("test case JSON must be an array")
  return [row for row in parsed if isinstance(row, dict)]


TEST_CASES = [
  {
    "batch": 1,
    "name": "active_superseded_by_capture",
    "input": """
{
  "active_memories": [
    {
      "memory_item_id": 101,
      "captured_datetime": "2025-09-10T14:05:00Z",
      "type": "goals",
      "content": "User is saving $15,000 for a Hawaii trip next summer."
    }
  ],
  "captured_memory_batches": [
    [
      {
        "captured_datetime": "2025-09-10T14:05:00Z",
        "expiry_datetime": "2025-09-10T14:15:00Z",
        "type": "goals",
        "content": "User is saving $15,000 for a Hawaii trip next summer.",
        "penny_log_id": 950010
      },
      {
        "captured_datetime": "2025-09-10T14:15:00Z",
        "expiry_datetime": "2026-03-10T14:15:00Z",
        "type": "goals",
        "content": "User is saving $20,000 for a Hawaii trip next summer.",
        "penny_log_id": 950011
      }
    ]
  ]
}
""",
    "output": """
[
  {
    "memory_item_id": 101,
    "captured_datetime": "2025-09-10T14:05:00Z",
    "expiry_datetime": "2025-09-10T14:15:00Z",
    "type": "goals",
    "content": "User is saving $15,000 for a Hawaii trip next summer.",
    "action": "update"
  },
  {
    "captured_datetime": "2025-09-10T14:05:00Z",
    "expiry_datetime": "2025-09-10T14:15:00Z",
    "type": "goals",
    "content": "User is saving $15,000 for a Hawaii trip next summer.",
    "penny_log_id": 950010,
    "action": "insert"
  },
  {
    "captured_datetime": "2025-09-10T14:15:00Z",
    "expiry_datetime": "2026-03-10T14:15:00Z",
    "type": "goals",
    "content": "User is saving $20,000 for a Hawaii trip next summer.",
    "penny_log_id": 950011,
    "action": "insert"
  }
]
""",
  },
  {
    "batch": 2,
    "name": "new_capture_no_active_change",
    "input": """
{
  "active_memories": [],
  "captured_memory_batches": [
    [
      {
        "captured_datetime": "2025-09-01T10:03:00Z",
        "expiry_datetime": null,
        "type": "spending",
        "content": "User usually grocery shops at Sprouts on Sundays.",
        "penny_log_id": 940003
      }
    ]
  ]
}
""",
    "output": """
[
  {
    "captured_datetime": "2025-09-01T10:03:00Z",
    "expiry_datetime": null,
    "type": "spending",
    "content": "User usually grocery shops at Sprouts on Sundays.",
    "penny_log_id": 940003,
    "action": "insert"
  }
]
""",
  },
  {
    "batch": 3,
    "name": "cross_batch_capture_supersede",
    "input": """
{
  "active_memories": [],
  "captured_memory_batches": [
    [
      {
        "captured_datetime": "2025-09-10T14:05:00Z",
        "expiry_datetime": "2026-09-10T14:05:00Z",
        "type": "goals",
        "content": "User is saving $15,000 for a Hawaii trip next summer.",
        "penny_log_id": 950010
      }
    ],
    [
      {
        "captured_datetime": "2025-09-10T14:05:00Z",
        "expiry_datetime": "2025-09-10T14:15:00Z",
        "type": "goals",
        "content": "User is saving $15,000 for a Hawaii trip next summer.",
        "penny_log_id": 950010
      },
      {
        "captured_datetime": "2025-09-10T14:15:00Z",
        "expiry_datetime": "2026-03-10T14:15:00Z",
        "type": "goals",
        "content": "User is saving $20,000 for a Hawaii trip next summer.",
        "penny_log_id": 950011
      }
    ]
  ]
}
""",
    "output": """
[
  {
    "captured_datetime": "2025-09-10T14:05:00Z",
    "expiry_datetime": "2025-09-10T14:15:00Z",
    "type": "goals",
    "content": "User is saving $15,000 for a Hawaii trip next summer.",
    "penny_log_id": 950010,
    "action": "insert"
  },
  {
    "captured_datetime": "2025-09-10T14:15:00Z",
    "expiry_datetime": "2026-03-10T14:15:00Z",
    "type": "goals",
    "content": "User is saving $20,000 for a Hawaii trip next summer.",
    "penny_log_id": 950011,
    "action": "insert"
  }
]
""",
  },
  {
    "batch": 4,
    "name": "no_changes_needed",
    "input": """
{
  "active_memories": [
    {
      "memory_item_id": 201,
      "captured_datetime": "2025-09-01T10:03:00Z",
      "type": "spending",
      "content": "User usually grocery shops at Sprouts on Sundays."
    }
  ],
  "captured_memory_batches": [
    []
  ]
}
""",
    "output": """
[]
""",
  },
]


class RefreshMemoryOptimizer:
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
    payload: str | dict[str, Any],
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
  optimizer: RefreshMemoryOptimizer | None = None,
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
    optimizer = RefreshMemoryOptimizer()
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
    ok, detail = _compare_refresh_output(result, ideal_output)
    _print_section_banner("Memory match")
    print("PASS" if ok else "FAIL")
    print(detail)
  return result


def run_tests(
  test_names_or_indices: list[int | str] | None = None,
  optimizer: RefreshMemoryOptimizer | None = None,
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
    print("  List tests:     python3 active_experiments/m_refresh_memory_optimizer.py --list")
    print("  Run by index:   python3 active_experiments/m_refresh_memory_optimizer.py --test 0")
    print("  Run by name:    python3 active_experiments/m_refresh_memory_optimizer.py --test active_superseded_by_capture")
    print("  Run all tests:  python3 active_experiments/m_refresh_memory_optimizer.py --test all")
    return

  kw: dict[str, Any] = {}
  if model is not None:
    kw["model_name"] = model
  optimizer = RefreshMemoryOptimizer(**kw)

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
