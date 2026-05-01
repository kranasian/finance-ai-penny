"""
Propose-next-steps rubric checker optimizer.

Use this to iterate on a checker system prompt (e.g. for a future Penny template)
that scores a **three-part** grader bundle (rationalize outcome, propose outcome, propose `calls`).

Run from `finance-ai-penny` repo root:

  python3 active_experiments/propose_next_steps_checker_optimizer.py --test all
  python3 active_experiments/propose_next_steps_checker_optimizer.py --test good_aligned
  python3 active_experiments/propose_next_steps_checker_optimizer.py --model gemini-flash-lite-latest

**LLM checker input**

The grader expects **three** inputs, matching production Hermes/DB shapes:

1. **`rationalize_agent_outcome`** — Markdown from **`/rationalize_by_category`** (same as the
   **`ai_agent_outcomes.agent_outcome`** row for **`type = rationalize_per_category`** that was fed
   into `/propose_next_steps`).
2. **`propose_agent_outcome`** — Markdown from **`/propose_next_steps`** (same as the
   **`ai_agent_outcomes.agent_outcome`** row for **`type = propose_next_steps`**: **`# Proposal`**
   section only).
3. **`calls`** — JSON array from the **propose_next_steps** row’s **`calls`** field (same shape as
   Hermes `ai_agent_outcomes.calls`). Use **`[]`** when there were no round-trips (checker still
   applies the no-tool-trace rule for **`accuracy`**).

These are wrapped for the model as **`<RATIONALIZE>`**, **`<PROPOSAL>`**, and **`<CALLS>`** (calls
rendered to markdown via `calls_to_markdown`). See `bundle_checker_input`.

**Rubric (two axes)**

| Axis | What it measures |
|------|------------------|
| **accuracy** | **Faithfulness + consistency:** proposed/open text matches **`<RATIONALIZE>`**; no invented facts; **`<CALLS>`** supports the story (e.g. retrieve before recategorize when transaction ids are not in rationalize markdown). |
| **completeness** | **Coverage + usefulness:** important rationalize **Next steps** / drivers appear in **`<PROPOSAL>`** or honestly under open items; concrete enough for Penny automation where relevant. |

**Multiple rounds:** Source `calls` has **one object per LLM round-trip**; the markdown lists them as
**`# Round 1`**, **`# Round 2`**, … with **`## Invoked tools`** under each when present.

Optional: grade from disk:

  python3 active_experiments/propose_next_steps_checker_optimizer.py --input-json /path/to/bundle.json

`bundle.json` shape:

  {
    "rationalize_agent_outcome": "<markdown from rationalize row>",
    "propose_agent_outcome": "<markdown from propose_next_steps row>",
    "calls": [ ... ]
  }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

try:
  from dotenv import load_dotenv
except Exception:  # pragma: no cover
  load_dotenv = None
from google import genai
from google.genai import types

if load_dotenv is not None:
  load_dotenv()

_TEST_SEPARATOR = "=" * 72
_SECTION_RULE = "-" * 72


def _print_section_banner(title: str) -> None:
  print(f"\n{_SECTION_RULE}\n{title}\n{_SECTION_RULE}\n")


def _tool_args_to_fenced_block(args: Any) -> str:
  """Format tool arguments for markdown (fenced block)."""
  if args is None:
    return ""
  if isinstance(args, str):
    s = args.strip()
    if not s:
      return ""
    try:
      parsed = json.loads(s)
      inner = json.dumps(parsed, indent=2, ensure_ascii=False)
      lang = "json"
    except Exception:
      inner = s
      lang = "text"
  else:
    inner = json.dumps(args, indent=2, ensure_ascii=False)
    lang = "json"
  return f"```{lang}\n{inner}\n```\n"


def calls_to_markdown(calls: list[Any]) -> str:
  """Render Hermes-style `calls` list as readable markdown for the grader."""
  if not isinstance(calls, list) or len(calls) == 0:
    return "_No LLM round-trips._\n"
  parts: list[str] = []
  for idx, turn in enumerate(calls, start=1):
    parts.append(f"# Round {idx}\n")
    if not isinstance(turn, dict):
      parts.append(f"- _(non-object round):_ `{type(turn).__name__}`\n\n")
      continue
    meta_lines: list[str] = []
    for key in ("call_number", "latency_ms", "input_tokens", "output_tokens", "total_tokens"):
      if key in turn and turn[key] is not None:
        meta_lines.append(f"- **{key}:** {turn[key]}")
    if meta_lines:
      parts.append("\n".join(meta_lines) + "\n\n")
    tcs = turn.get("tool_calls")
    if not isinstance(tcs, list) or not tcs:
      parts.append("_No tool calls this round._\n\n")
      continue
    parts.append("## Invoked tools\n\n")
    for tci, tc in enumerate(tcs, start=1):
      if not isinstance(tc, dict):
        parts.append(f"{tci}. _(invalid tool call entry)_\n\n")
        continue
      name = tc.get("tool_name") or tc.get("name")
      fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
      if name is None and isinstance(fn, dict):
        name = fn.get("name")
      args = tc.get("args") if "args" in tc else tc.get("arguments")
      if args is None and isinstance(fn, dict):
        args = fn.get("arguments")
      name_s = str(name or "(unknown_tool)")
      parts.append(f"{tci}. **`{name_s}`**\n\n")
      fence = _tool_args_to_fenced_block(args)
      if fence:
        parts.append(fence)
      parts.append("\n")
  return "".join(parts).rstrip() + "\n"


def bundle_checker_input(
  *,
  rationalize_agent_outcome: str,
  propose_agent_outcome: str,
  calls: list[Any],
) -> str:
  """Build grader message: ``<RATIONALIZE>``, ``<PROPOSAL>``, ``<CALLS>``. ``calls`` may be ``[]``."""
  ra = (rationalize_agent_outcome or "").strip()
  pa = (propose_agent_outcome or "").strip()
  if not ra or not pa:
    raise ValueError(
      "rationalize_agent_outcome and propose_agent_outcome must be non-empty markdown strings."
    )
  if not isinstance(calls, list):
    raise TypeError("calls must be a list (use [] if the propose row has no LLM round-trips).")
  out = (
    f"<RATIONALIZE>\n\n{ra}\n\n</RATIONALIZE>\n\n"
    f"<PROPOSAL>\n\n{pa}\n\n</PROPOSAL>\n"
  )
  calls_md = calls_to_markdown(calls)
  out += f"\n<CALLS>\n\n{calls_md}\n</CALLS>\n"
  return out


_AXIS_SCHEMA = types.Schema(
  type=types.Type.OBJECT,
  required=["score", "notes"],
  properties={
    "score": types.Schema(type=types.Type.INTEGER, description="Integer 1–5."),
    "notes": types.Schema(type=types.Type.STRING, description="One short sentence."),
  },
)

OUTPUT_SCHEMA = types.Schema(
  type=types.Type.OBJECT,
  required=["completeness", "accuracy"],
  properties={
    "completeness": _AXIS_SCHEMA,
    "accuracy": _AXIS_SCHEMA,
  },
)


SYSTEM_PROMPT = """You are a **strict rubric grader** for the checker bundle below (XML-style role wrappers; do not require table/column names).

You receive:

1. **`<RATIONALIZE>` … `</RATIONALIZE>`** — Markdown from the **rationalize-by-category** outcome (`agent_outcome` for that run): figures, drivers, next steps.

2. **`<PROPOSAL>` … `</PROPOSAL>`** — Markdown from the **propose-next-steps** outcome (`agent_outcome` for that run): normally **only** the **`# Proposal`** block with **`## Proposed next steps`** and **`## Open items (not addressed)`** (or equivalent).

3. **`<CALLS>` … `</CALLS>`** — Markdown listing **LLM round-trips** from the **same propose-next-steps** run (`calls` field): `# Round N`, optional metrics, **`## Invoked tools`** with numbered tools and fenced argument blocks. **`latency_ms` / `input_tokens` / `output_tokens`** appear only when present in source data.

If **`<CALLS>`** says there are **no rounds**, or **no tools** were invoked in any round, score the **tool-trace** aspect of **`accuracy` = 3** with `notes` stating no usable tool trace (do not infer tool order from proposal text alone). You may still score **grounding** of proposal text vs **`<RATIONALIZE>`** when evidence is clear.

Grade **only** what is in the message. Do not invent missing data.

**Axes (each `score` integer 1–5, `notes` one short sentence):**

1. **`completeness`** — Important **Next steps** / drivers from **`<RATIONALIZE>`** are reflected in **`<PROPOSAL>`** under “Proposed next steps” or **honestly** under “Open items” (no silent drops); proposals are **concrete enough** for Penny (budgets, rules, recategorize with ids) where the rationalize text calls for automation—not only generic filler unless that is truly all that is warranted.

2. **`accuracy`** — Proposed + open items are **grounded** in **`<RATIONALIZE>`** (no invented merchants/amounts/dates; no contradictions). When **`<CALLS>`** documents invocations, they **match** the proposal (correct tools, sensible args; **retrieve** before **propose_recategorize_transactions** when ids are not in **`<RATIONALIZE>`**). Apply the **score 3** rule above when no usable tool trace exists.

**Calibration:** **5** = no meaningful gap on that axis. **4** = one minor gap. **3** = clear but fixable issue. **2** = several problems. **1** = axis largely failed.

Return **only** the JSON object matching the schema.
"""


# Minimal fixtures: rationalize snippet + propose bundle + synthetic calls
_RATIONALIZE_FIXTURE = """# Rationalize What

Explain: Groceries is significantly down this month at $120. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

- Groceries: $120 this month (Apr 1–30, 2026).

## Drivers

Several Costco transactions appear miscategorized as Miscellaneous.

## Next steps

Create a categorization rule so Costco is always groceries.
"""

_GOOD_PROPOSE_ONLY = (
  "# Proposal\n\n"
  "## Proposed next steps\n\n"
  "1. **Add a categorization rule** so merchant name matches “Costco” map to Groceries (`ai_category_id` for groceries), applied to past and future.\n\n"
  "## Open items (not addressed)\n\n"
  "1. **Review** whether any non-Costco groceries are still landing in Miscellaneous.\n"
)

_GOOD_CALLS: list[dict[str, Any]] = [
  {
    "tool_calls": [
      {
        "tool_name": "propose_create_categorization_rule",
        "args": json.dumps(
          {
            "rule": {"name_sub_eq": "costco"},
            "ai_category_id": 4,
            "scope": "future_and_past",
            "rationale": "Costco should count as groceries.",
          }
        ),
      }
    ],
  },
]

_BAD_PROPOSE_ONLY = (
  "# Proposal\n\n"
  "## Proposed next steps\n\n"
  "1. **Recategorize** the $500 Whole Foods charge on 2026-04-02 from Dining to Groceries.\n\n"
  "## Open items (not addressed)\n\n"
  "1. None.\n"
)

_BAD_CALLS_NO_RETRIEVE: list[dict[str, Any]] = [
  {
    "tool_calls": [
      {
        "tool_name": "propose_recategorize_transactions",
        "args": json.dumps({"transaction_ids": [99999], "target_category": "groceries", "rationale": "fix"}),
      }
    ],
  },
]

TEST_CASES: list[dict[str, Any]] = [
  {
    "name": "good_aligned",
    "ideal_response": None,
    "payload": bundle_checker_input(
      rationalize_agent_outcome=_RATIONALIZE_FIXTURE.strip(),
      propose_agent_outcome=_GOOD_PROPOSE_ONLY.strip(),
      calls=_GOOD_CALLS,
    ),
  },
  {
    "name": "bad_hallucinated_merchant",
    "ideal_response": None,
    "payload": bundle_checker_input(
      rationalize_agent_outcome=_RATIONALIZE_FIXTURE.strip(),
      propose_agent_outcome=_BAD_PROPOSE_ONLY.strip(),
      calls=_BAD_CALLS_NO_RETRIEVE,
    ),
  },
]


class ProposeNextStepsCheckerOptimizer:
  def __init__(
    self,
    model_name: str = "gemini-flash-lite-latest",
    *,
    max_output_tokens: int = 512,
    thinking_budget: int = 0,
  ):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set.")
    self.client = genai.Client(api_key=api_key)
    self.model_name = model_name
    self.temperature = 0.0
    self.top_p = 0.95
    self.max_output_tokens = max_output_tokens
    self.thinking_budget = thinking_budget
    self.system_prompt = SYSTEM_PROMPT
    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    ]

  def grade(self, bundled_input: str) -> Dict[str, Any]:
    user_msg = (
      "Grade **completeness** and **accuracy** (two axes). Input uses `<RATIONALIZE>` / `<PROPOSAL>` / `<CALLS>`.\n\n"
      + (bundled_input or "")
      + "\n\nRespond with the JSON object only."
    )
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_msg)])]
    cfg = types.GenerateContentConfig(
      temperature=self.temperature,
      top_p=self.top_p,
      max_output_tokens=self.max_output_tokens,
      safety_settings=self.safety_settings,
      system_instruction=[types.Part.from_text(text=self.system_prompt)],
      thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget, include_thoughts=True),
      response_mime_type="application/json",
      response_schema=OUTPUT_SCHEMA,
    )
    out = self.client.models.generate_content(model=self.model_name, contents=contents, config=cfg)
    text = (out.text or "").strip()
    try:
      return json.loads(text)
    except Exception:
      s = text[text.find("{") : text.rfind("}") + 1] if ("{" in text and "}" in text) else "{}"
      return json.loads(s)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--test", type=str, default="all", help="Test name or 'all'.")
  parser.add_argument("--model", type=str, default="gemini-flash-lite-latest")
  parser.add_argument("--max-output-tokens", type=int, default=512)
  parser.add_argument("--thinking-budget", type=int, default=0)
  parser.add_argument(
    "--input-json",
    type=str,
    default=None,
    help='Path to JSON with "rationalize_agent_outcome", "propose_agent_outcome", "calls" (see module docstring).',
  )
  args = parser.parse_args()

  opt = ProposeNextStepsCheckerOptimizer(
    model_name=args.model,
    max_output_tokens=args.max_output_tokens,
    thinking_budget=args.thinking_budget,
  )

  if args.input_json:
    raw = open(args.input_json, encoding="utf-8").read()
    obj = json.loads(raw)
    if not isinstance(obj, dict):
      raise SystemExit("--input-json must contain a JSON object")
    rao = obj.get("rationalize_agent_outcome")
    pao = obj.get("propose_agent_outcome")
    calls_raw = obj.get("calls")
    if not isinstance(rao, str) or not rao.strip():
      raise SystemExit('--input-json requires non-empty string "rationalize_agent_outcome".')
    if not isinstance(pao, str) or not pao.strip():
      raise SystemExit('--input-json requires non-empty string "propose_agent_outcome".')
    if not isinstance(calls_raw, list):
      raise SystemExit(
        '--input-json requires "calls" as a JSON array (use [] if the propose row has no calls).'
      )
    bundle = bundle_checker_input(
      rationalize_agent_outcome=rao.strip(),
      propose_agent_outcome=pao.strip(),
      calls=calls_raw,
    )
    _print_section_banner("# LLM Checker Input (--input-json)")
    print(bundle)
    result = opt.grade(bundle)
    _print_section_banner("# LLM Checker Response")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return

  if args.test == "all":
    cases = TEST_CASES
  else:
    tc = next((t for t in TEST_CASES if t["name"] == args.test), None)
    if tc is None:
      raise SystemExit(f"Unknown test: {args.test!r}")
    cases = [tc]

  for i, tc in enumerate(cases):
    if i:
      print(f"\n{_TEST_SEPARATOR}\n")
    print(f"# Test: {tc['name']}\n")
    _print_section_banner("# LLM Checker Input")
    print(tc["payload"])
    result = opt.grade(tc["payload"])
    _print_section_banner("# LLM Checker Response")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if tc.get("ideal_response") is not None:
      _print_section_banner("# Ideal Response")
      print(json.dumps(tc["ideal_response"], indent=2, ensure_ascii=False))

  print(f"\n{_TEST_SEPARATOR}\n")
  print(f"# Total tests: {len(cases)}\n")


if __name__ == "__main__":
  main()
