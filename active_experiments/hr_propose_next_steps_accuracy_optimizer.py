"""
Propose-next-steps rubric optimizer — **accuracy only**.

Grades only the **accuracy** axis.

Run from `finance-ai-penny` repo root:

  python3 active_experiments/hr_propose_next_steps_accuracy_optimizer.py --test all
  python3 active_experiments/hr_propose_next_steps_accuracy_optimizer.py --test good_aligned
  python3 active_experiments/hr_propose_next_steps_accuracy_optimizer.py --model gemini-flash-lite-latest
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

try:
  from dotenv import load_dotenv
except Exception:  # pragma: no cover
  load_dotenv = None

try:
  from google import genai  # type: ignore[import-not-found]
  from google.genai import types  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
  genai = None
  types = None

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


def rationalize_calls_to_markdown(calls: list[Any]) -> str:
  """Stored rationalize ``calls`` inside ``<RATIONALIZE_CALLS>`` (Hermes-aligned)."""
  if not isinstance(calls, list) or len(calls) == 0:
    return ""
  parts: list[str] = []
  for idx, turn in enumerate(calls, start=1):
    parts.append(f"# Round {idx}\n\n")
    if not isinstance(turn, dict):
      parts.append(f"- _(non-object round):_ `{type(turn).__name__}`\n\n")
      continue
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


def _append_proposal_and_calls(out: str, propose_agent_outcome: str, calls: list[Any]) -> str:
  pa = (propose_agent_outcome or "").strip()
  if not pa:
    raise ValueError("propose_agent_outcome must be non-empty.")
  if not isinstance(calls, list):
    raise TypeError("calls must be a list (use [] if the propose row has no LLM round-trips).")
  calls_md = calls_to_markdown(calls)
  return (
    out
    + f"<PROPOSAL>\n\n{pa}\n\n</PROPOSAL>\n\n"
    + f"<CALLS>\n\n{calls_md}\n</CALLS>\n"
  )


def bundle_checker_input_multi(
  *,
  contexts: list[dict[str, Any]],
  propose_agent_outcome: str,
  calls: list[Any],
) -> str:
  """
  Grader bundle for ``propose_next_steps_multi``: ``CONTEXTS`` (same outer shape as Hermes), then ``<PROPOSAL>``, ``<CALLS>``.
  Each context dict: ``rationalize_agent_outcome`` (str), optional ``calls`` (list, rationalize row trace).
  """
  if not isinstance(contexts, list) or not contexts:
    raise ValueError("contexts must be a non-empty list of dicts.")
  inner_parts: list[str] = []
  for i, ctx in enumerate(contexts, start=1):
    if not isinstance(ctx, dict):
      raise TypeError(f"contexts[{i - 1}] must be a dict")
    ra = (ctx.get("rationalize_agent_outcome") or "").strip()
    if not ra:
      raise ValueError(f"contexts[{i - 1}].rationalize_agent_outcome must be non-empty")
    body = ra.rstrip("\n")
    chunks: list[str] = [
      f'<CONTEXT index="{i}">\n\n',
      "<RATIONALIZE>\n\n",
      body,
      "\n\n</RATIONALIZE>\n",
    ]
    calls_val = ctx.get("calls")
    if isinstance(calls_val, list) and len(calls_val) > 0:
      md = rationalize_calls_to_markdown(calls_val)
      if md.strip():
        chunks.append("\n<RATIONALIZE_CALLS>\n\n")
        chunks.append(md.rstrip("\n") + "\n")
        chunks.append("\n</RATIONALIZE_CALLS>\n")
    chunks.append("\n</CONTEXT>")
    inner_parts.append("".join(chunks))
  inner = "\n\n".join(inner_parts)
  head = (
    "Multiple prior rationalize contexts:\n\n"
    "<CONTEXTS>\n\n"
    + inner
    + "\n\n</CONTEXTS>\n\n"
  )
  return _append_proposal_and_calls(head, propose_agent_outcome, calls)


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

_SERVICE_FEES_RATIONALIZE_FIXTURE = """# Rationalize What

Explain: Service Fees are significantly down this month. (credit card interest charges) (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

- Service Fees / interest charges: down vs last month.

## Drivers

Credit card interest charges decreased.

## Next steps

1. Consider setting a budget for service fees to keep interest charges low.
2. Review APR / statement details to confirm why interest changed.
"""

_SERVICE_FEES_PROPOSE_ONLY = """# Proposal

## Proposed next steps

1. **Create** a monthly spending budget of $250 for Service Fees to support the current downward trend in credit card interest charges.

## Open items (not addressed)

1. **Review** credit card account statements and APR details to determine if the interest reduction resulted from a lower average daily balance or a rate change.
2. **Explore** debt repayment strategies or balance transfer options to eliminate remaining recurring interest fees.
"""

_SERVICE_FEES_CALLS: list[dict[str, Any]] = [
  {
    "tool_calls": [
      {
        "args": "{\"category\":\"bills_service_fees\",\"goal_type\":\"spending_budget\",\"rationale\":\"Based on the recent reduction in credit card interest charges, a $250 monthly budget for service fees will help sustain this positive trend and encourage continued debt management.\",\"time_horizon\":\"monthly\",\"goal_title\":\"Limit Service Fees\",\"target_amount\":250}",
        "tool_name": "propose_create_goal",
      }
    ],
  },
  {
    "tool_calls": [],
  },
]

_KIDS_ED_RATIONALIZE_FIXTURE = """# Rationalize What

Explain: Kids education spending this month. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

- Kids education: $180 this month.

## Next steps

Review recurring tutoring charges for consistency.
"""

_MULTI_SYNTH_PROPOSE = """# Proposal

## Proposed next steps

1. **Create** a monthly spending budget of $250 for Service Fees (credit card interest) to sustain the downward trend.
2. **Review** recurring tutoring charges under Kids education for consistency and miscategorization.

## Open items (not addressed)

1. **Confirm** whether tutoring amounts align with expected subscription cadence after categorization cleanup.

"""

_MULTI_SYNTH_CALLS: list[dict[str, Any]] = [
  {
    "tool_calls": [
      {
        "tool_name": "propose_create_goal",
        "args": json.dumps(
          {
            "category": "bills_service_fees",
            "goal_type": "spending_budget",
            "goal_title": "Limit Service Fees",
            "target_amount": 250,
            "time_horizon": "monthly",
            "rationale": "Align with reduced interest narrative.",
          }
        ),
      },
    ],
  },
]

_MULTI_IGNORES_ONE_CONTEXT_PROPOSE = """# Proposal

## Proposed next steps

1. **Create** a monthly spending budget of $250 for Service Fees.

## Open items (not addressed)

1. None.

"""

_MULTI_IGNORES_ONE_CONTEXT_CALLS = list(_MULTI_SYNTH_CALLS)

_BOTH_AXIS_TEST_CASES: list[dict[str, Any]] = [
  {
    "name": "good_aligned",
    "ideal_response": {
      "completeness": {
        "score": 5,
        "notes": "Covers the rationalize next step with a concrete proposal and a sensible open item.",
      },
      "accuracy": {"score": 5, "notes": "Proposal is grounded in the rationalize context and the tool call matches the text."},
    },
    "payload": bundle_checker_input(
      rationalize_agent_outcome=_RATIONALIZE_FIXTURE.strip(),
      propose_agent_outcome=_GOOD_PROPOSE_ONLY.strip(),
      calls=_GOOD_CALLS,
    ),
  },
  {
    "name": "bad_hallucinated_merchant",
    "ideal_response": {
      "completeness": {
        "score": 1,
        "notes": "Does not address the Costco miscategorization next step and provides an unrelated action.",
      },
      "accuracy": {
        "score": 1,
        "notes": "Invents a Whole Foods charge and tool usage is inconsistent with the rationalize context.",
      },
    },
    "payload": bundle_checker_input(
      rationalize_agent_outcome=_RATIONALIZE_FIXTURE.strip(),
      propose_agent_outcome=_BAD_PROPOSE_ONLY.strip(),
      calls=_BAD_CALLS_NO_RETRIEVE,
    ),
  },
  {
    "name": "real_service_fees_goal_two_rounds",
    "ideal_response": {
      "completeness": {
        "score": 5,
        "notes": "Creates a concrete budget and captures the remaining review items as open, covering the rationalize next steps.",
      },
      "accuracy": {
        "score": 5,
        "notes": "Proposal and tool call are grounded in the rationalize context and are actionable for Penny.",
      },
    },
    "payload": bundle_checker_input(
      rationalize_agent_outcome=_SERVICE_FEES_RATIONALIZE_FIXTURE.strip(),
      propose_agent_outcome=_SERVICE_FEES_PROPOSE_ONLY.strip(),
      calls=_SERVICE_FEES_CALLS,
    ),
  },
  {
    "name": "multi_two_contexts_synthesized",
    "ideal_response": {
      "completeness": {
        "score": 5,
        "notes": "Addresses both service-fee budgeting and kids-education tutoring review from the two rationalize contexts.",
      },
      "accuracy": {
        "score": 5,
        "notes": "Grounded in both contexts; propose_create_goal supports the service-fees thread.",
      },
    },
    "payload": bundle_checker_input_multi(
      contexts=[
        {"rationalize_agent_outcome": _SERVICE_FEES_RATIONALIZE_FIXTURE.strip(), "calls": []},
        {"rationalize_agent_outcome": _KIDS_ED_RATIONALIZE_FIXTURE.strip(), "calls": []},
      ],
      propose_agent_outcome=_MULTI_SYNTH_PROPOSE.strip(),
      calls=_MULTI_SYNTH_CALLS,
    ),
  },
  {
    "name": "multi_drops_second_context",
    "ideal_response": {
      "completeness": {
        "score": 2,
        "notes": "Ignores the kids-education tutoring thread while only covering service fees.",
      },
      "accuracy": {"score": 4, "notes": "Remaining content is grounded but incomplete versus dual-context evidence."},
    },
    "payload": bundle_checker_input_multi(
      contexts=[
        {"rationalize_agent_outcome": _SERVICE_FEES_RATIONALIZE_FIXTURE.strip(), "calls": []},
        {"rationalize_agent_outcome": _KIDS_ED_RATIONALIZE_FIXTURE.strip(), "calls": []},
      ],
      propose_agent_outcome=_MULTI_IGNORES_ONE_CONTEXT_PROPOSE.strip(),
      calls=_MULTI_IGNORES_ONE_CONTEXT_CALLS,
    ),
  },
]


CHECKER_USER_MSG_PREFIX = (
  "Grade **accuracy** only.\n"
  "Input may be **single-context** (`<RATIONALIZE>` / `<PROPOSAL>` / `<CALLS>`) or **multi-context** "
  "(`Multiple prior rationalize contexts:` + `<CONTEXTS>` …, then `<PROPOSAL>`, then `<CALLS>`).\n\n"
)


def _build_output_schema(_types: Any) -> Any:
  axis = _types.Schema(
    type=_types.Type.OBJECT,
    required=["score", "notes"],
    properties={
      "score": _types.Schema(type=_types.Type.INTEGER, description="Integer 1–5."),
      "notes": _types.Schema(type=_types.Type.STRING, description="One short sentence."),
    },
  )
  return _types.Schema(
    type=_types.Type.OBJECT,
    required=["accuracy"],
    properties={"accuracy": axis},
  )


SYSTEM_PROMPT = """You are a **strict rubric grader** for the checker bundle below (XML-style role wrappers; do not require table/column names).

You receive **evidence**, then **`<PROPOSAL>`**, then **`<CALLS>`**. Evidence may be **single** or **multi** context.

**Evidence — one of:**

1. **Single rationalize (`propose_next_steps`):** **`<RATIONALIZE>` … `</RATIONALIZE>`** — Markdown from **one** rationalize-per-category outcome: figures, drivers, next steps.

2. **Multiple rationalizes (`propose_next_steps_multi`):** **`Multiple prior rationalize contexts:`** plus **`<CONTEXTS>` … `</CONTEXTS>`**. Each **`<CONTEXT index="N">`** contains **`<RATIONALIZE>` … `</RATIONALIZE>`** for that prior run (same kind of markdown as (1)). Optionally **`<RATIONALIZE_CALLS>` … `</RATIONALIZE_CALLS>`** holds **that rationalize run’s** stored LLM trace — **not** the propose run.

For **grounding**, treat **all** **`<RATIONALIZE>`** bodies as the **combined** evidence; the proposal should synthesize across contexts.

**Then (same for single and multi):**

3. **`<PROPOSAL>` … `</PROPOSAL>`** — Markdown from the **propose** outcome (`agent_outcome`): normally **only** the **`# Proposal`** block with **`## Proposed next steps`** and **`## Open items (not addressed)`** (or equivalent **`##`** headings if `# Proposal` was omitted).

4. **`<CALLS>` … `</CALLS>`** — Markdown listing **LLM round-trips** from **this same propose / propose-multi run** (`calls` field): `# Round N`, optional metrics, **`## Invoked tools`** with numbered tools and fenced argument blocks. **`latency_ms` / `input_tokens` / `output_tokens`** appear only when present in source data.

**Do not** confuse **`<RATIONALIZE_CALLS>`** inside a **`<CONTEXT>`** with **`<CALLS>`** at the end. Grade tool consistency for the propose run using **`<CALLS>`** only.

If **`<CALLS>`** says there are **no rounds**, or **no tools** were invoked in any round, score the **tool-trace** aspect of **`accuracy` = 3** with `notes` stating no usable tool trace (do not infer tool order from proposal text alone). You may still score grounding of proposal text vs **all** rationalize evidence when clear.

Grade **only** what is in the message. Do not invent missing data.

**Axis (return ONLY `accuracy` with `score` 1–5 and `notes` one short sentence):**

**`accuracy`** — Is **`<PROPOSAL>`** both **grounded** and **helpful for Penny to execute**?
  - Grounding: No invented merchants/amounts/dates; no contradictions vs **any** **`<RATIONALIZE>`** body.
  - Actionability for Penny: Proposed steps should align with Penny’s capabilities (create goals/budgets, create categorization rules, recategorize transactions) and avoid recommending actions irrelevant to the rationalize context(s).
  - Tool consistency: When **`<CALLS>`** documents invocations, they must support the proposal (correct tools, sensible args; **retrieve** before **propose_recategorize_transactions** when ids are not in rationalize text). Apply the **score 3** rule above when no usable tool trace exists.

**Calibration:** **5** = no meaningful gap on that axis. **4** = one minor gap. **3** = clear but fixable issue. **2** = several problems. **1** = axis largely failed.

Return **only** the JSON object matching the schema.
"""


def _to_accuracy_test_cases() -> list[dict[str, Any]]:
  # Keep the same payload fixtures, but only keep the `accuracy` ideal response.
  out: list[dict[str, Any]] = []
  for tc in _BOTH_AXIS_TEST_CASES:
    ideal = tc.get("ideal_response") or None
    ideal_acc = (
      {"accuracy": ideal["accuracy"]}
      if isinstance(ideal, dict) and "accuracy" in ideal
      else None
    )
    out.append({"name": tc["name"], "payload": tc["payload"], "ideal_response": ideal_acc})
  return out


TEST_CASES: list[dict[str, Any]] = _to_accuracy_test_cases()


class ProposeNextStepsAccuracyCheckerOptimizer:
  def __init__(
    self,
    model_name: str = "gemini-flash-lite-latest",
    *,
    max_output_tokens: int = 256,
    thinking_budget: int = 0,
  ):
    if genai is None or types is None:
      raise ImportError(
        "Missing dependency for Gemini client. Install `google-genai` (and ensure your venv is active) "
        "to run this script."
      )
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set.")
    self.client = genai.Client(api_key=api_key)
    self._types = types
    self.model_name = model_name
    self.temperature = 0.0
    self.top_p = 0.95
    self.max_output_tokens = max_output_tokens
    self.thinking_budget = thinking_budget
    self.system_prompt = SYSTEM_PROMPT
    self.safety_settings = [
      self._types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      self._types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      self._types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      self._types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    ]
    self.output_schema = _build_output_schema(self._types)

  def grade(self, bundled_input: str) -> Dict[str, Any]:
    user_msg = (
      CHECKER_USER_MSG_PREFIX
      + (bundled_input or "")
      + "\n\nRespond with the JSON object only."
    )
    t = self._types
    contents = [t.Content(role="user", parts=[t.Part.from_text(text=user_msg)])]
    cfg = t.GenerateContentConfig(
      temperature=self.temperature,
      top_p=self.top_p,
      max_output_tokens=self.max_output_tokens,
      safety_settings=self.safety_settings,
      system_instruction=[t.Part.from_text(text=self.system_prompt)],
      thinking_config=t.ThinkingConfig(thinking_budget=self.thinking_budget, include_thoughts=True),
      response_mime_type="application/json",
      response_schema=self.output_schema,
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
  parser.add_argument("--max-output-tokens", type=int, default=256)
  parser.add_argument("--thinking-budget", type=int, default=0)
  args = parser.parse_args()

  opt = ProposeNextStepsAccuracyCheckerOptimizer(
    model_name=args.model,
    max_output_tokens=args.max_output_tokens,
    thinking_budget=args.thinking_budget,
  )

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

