"""
Optimizer runner for **P:NeedUpdateVerbalizer** (Gemini prompt tuning).

Input is the current verbalized need markdown plus a user tune message.
Output decides whether the verbalized need requires updating and, when yes,
returns reverbalized need copy.

Run from ``finance-ai-penny`` repo root:

  python3 active_experiments/need_update_verbalizer_optimizer.py --test 0
  python3 active_experiments/need_update_verbalizer_optimizer.py --test all
  python3 active_experiments/need_update_verbalizer_optimizer.py --test faster_paydown_pacing --print-input-only
  python3 active_experiments/need_update_verbalizer_optimizer.py --input-file path/to/input.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    from google import genai
    from google.genai import types
    from google.genai.errors import ClientError
except Exception:
    genai = None
    types = None
    ClientError = Exception

from active_experiments.need_verbalizer_optimizer import (
    GEMINI_FLASH_LITE,
    _parse_model_json_object,
)
from active_experiments.verbalizer_optimizer_db import _resolve_ideal_response

if load_dotenv is not None:
    load_dotenv()

_TEMPLATE_NAME = "P:NeedUpdateVerbalizer"
THINKING_BUDGET = 128
MAX_OUTPUT_TOKENS = 4096

_CURRENT_VERBALIZED_NEED_H1 = "# Current Verbalized Need"
_USER_TUNE_MESSAGE_H1 = "# User Tune Message"
_NEED_COPY_FIELDS = ("needs_title", "concise_need", "full_need")
_NEED_BULLET_RE = re.compile(
    r"^\s*-\s*(needs_title|concise_need|full_need):\s*(.+?)\s*$",
    re.MULTILINE,
)


def _need_copy_payload(verbalized_need: dict[str, Any]) -> dict[str, Any]:
    return {
        field: str(verbalized_need.get(field) or "").strip()
        for field in _NEED_COPY_FIELDS
    }


def parse_verbalized_need_from_user_message(user_message: str) -> dict[str, Any]:
    text = (user_message or "").strip()
    if _CURRENT_VERBALIZED_NEED_H1 not in text:
        raise ValueError(f"input must include {_CURRENT_VERBALIZED_NEED_H1}")
    after_need = text.split(_CURRENT_VERBALIZED_NEED_H1, 1)[1]
    if _USER_TUNE_MESSAGE_H1 not in after_need:
        raise ValueError(f"input must include {_USER_TUNE_MESSAGE_H1}")
    need_block = after_need.split(_USER_TUNE_MESSAGE_H1, 1)[0].strip()
    parsed: dict[str, str] = {}
    for match in _NEED_BULLET_RE.finditer(need_block):
        parsed[match.group(1)] = match.group(2).strip()
    for field in _NEED_COPY_FIELDS:
        if not parsed.get(field):
            raise ValueError(f"input must include - {field}: bullet")
    return _need_copy_payload(parsed)


def _post_process_need_update_response(
    parsed: dict[str, Any],
    verbalized_need: dict[str, Any],
) -> dict[str, Any]:
    cleaned = _validate_need_update_response(parsed)
    if not cleaned["needs_update"]:
        return cleaned
    updated_copy = cleaned["updated_need"]
    if not isinstance(updated_copy, dict):
        raise ValueError("updated_need must be a JSON object when needs_update is true")
    merged_need = {**verbalized_need, **_need_copy_payload(updated_copy)}
    for field in ("needs_title", "concise_need", "full_need"):
        merged_need[field] = str(updated_copy.get(field) or merged_need.get(field) or "").strip()
    return {
        "needs_update": True,
        "updated_need": merged_need,
    }


def resolve_verbalized_need_after_tune(
    response: dict[str, Any],
    verbalized_need: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise ValueError(f"Expected JSON object from {_TEMPLATE_NAME}")
    if not response.get("needs_update"):
        return dict(verbalized_need)
    updated_need = response.get("updated_need")
    if not isinstance(updated_need, dict):
        raise ValueError("updated_need must be a JSON object when needs_update is true")
    return {**verbalized_need, **_need_copy_payload(updated_need)}


SYSTEM_PROMPT = """You are Penny — a positive, empathetic money coach reviewing whether a user's tune message requires updating already-verbalized financial need copy.

## Input
- `# Current Verbalized Need`: the existing need title and copy as `- needs_title: ...`, `- concise_need: ...`, and `- full_need: ...`.
- `# User Tune Message`: what the user asked to change while tuning their plan.

## Decision
The tune message adjusts the user's **plan** — caps, pacing, goal targets, start month, timeline, or baseline cuts. The user is not asking to retune the financial need itself. Decide whether the existing need copy still matches after that plan change.

Set `needs_update` to **false** when the plan tune does not make anything in `needs_title`, `concise_need`, or `full_need` wrong or incomplete. Examples:
- faster or slower paydown pacing when the need is about card interest, not pacing months
- category spending caps when the need copy does not cite those cap amounts
- plan start month or timeline when dates in the need copy are unchanged
- baseline percentage cuts when the need copy does not reference those baselines
- goal target changes when the need copy does not mention the old target or milestone

Set `needs_update` to **true** when the plan tune changes a fact already stated in the need copy, even though the user only tuned the plan. Examples:
- `full_need` cites a $700k milestone and the user asks to set the target goal to $800k — reverbalize with $800k
- `full_need` cites the plan starting in Sep 2026 and the user asks to start the plan in Nov 2026 instead — reverbalize with Nov 2026

When `needs_update` is false, set `updated_need` to null.
When `needs_update` is true, fill `updated_need` with only `needs_title`, `concise_need`, and `full_need`. Update only the fields that cite the stale fact; keep unaffected wording as close as possible.

## Updated need copy rules
- Keep Penny's empathetic, second-person tone.
- `needs_title`: max 5 words and under 40 characters, sentence-like problem statement, exactly 1 emoji at the end, no period.
- `concise_need`: max 18 words, 1 to 2 emojis at the end.
- `full_need`: max 40 words, 1 to 2 emojis at the end.
- When the tune changes a dollar amount, date, milestone, or other fact already stated in the need copy, update that fact in the reverbalized copy.
- Do not invent new balances, accounts, or dates that are not supported by the current need or tune message.
- No exclamation marks or superlatives.
"""


def _build_updated_need_schema() -> "types.Schema":
    return types.Schema(
        type=types.Type.OBJECT,
        required=list(_NEED_COPY_FIELDS),
        properties={
            "needs_title": types.Schema(
                type=types.Type.STRING,
                description="Updated headline for the need. Max 5 words and under 40 characters. Include exactly 1 emoji.",
            ),
            "concise_need": types.Schema(
                type=types.Type.STRING,
                description="Updated one-sentence need summary. Max 18 words. Include 1 to 2 emojis.",
            ),
            "full_need": types.Schema(
                type=types.Type.STRING,
                description="Updated need detail. Max 40 words. Include 1 to 2 emojis.",
            ),
        },
    )


def _build_output_schema() -> "types.Schema":
    if types is None:
        raise RuntimeError("Install `google-genai` for this optimizer.")
    return types.Schema(
        type=types.Type.OBJECT,
        required=["needs_update", "updated_need"],
        properties={
            "needs_update": types.Schema(
                type=types.Type.BOOLEAN,
                description="True when the tune message makes any part of the current need copy stale, incomplete, or misaligned.",
            ),
            "updated_need": types.Schema(
                type=types.Type.OBJECT,
                nullable=True,
                required=list(_NEED_COPY_FIELDS),
                properties=_build_updated_need_schema().properties,
            ),
        },
    )


def _validate_need_update_response(parsed: Any) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        raise ValueError("Need update verbalizer response must be a JSON object")
    needs_update = parsed.get("needs_update")
    if not isinstance(needs_update, bool):
        raise ValueError("needs_update must be a boolean")
    updated_raw = parsed.get("updated_need")
    if needs_update:
        if not isinstance(updated_raw, dict):
            raise ValueError("updated_need must be a JSON object when needs_update is true")
        updated_need = _need_copy_payload(updated_raw)
        if not updated_need["needs_title"]:
            raise ValueError("needs_title must be a non-empty string")
        if not updated_need["concise_need"]:
            raise ValueError("concise_need must be a non-empty string")
        if not updated_need["full_need"]:
            raise ValueError("full_need must be a non-empty string")
    else:
        if updated_raw is not None:
            raise ValueError("updated_need must be null when needs_update is false")
        updated_need = None
    return {
        "needs_update": needs_update,
        "updated_need": updated_need,
    }


class NeedUpdatedVerbalizerOptimizer:
    def __init__(
        self,
        model_name: str = GEMINI_FLASH_LITE,
        *,
        thinking_budget: int = THINKING_BUDGET,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
    ):
        if genai is None or types is None:
            raise RuntimeError("Install `google-genai` for this optimizer.")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.thinking_budget = thinking_budget
        self.max_output_tokens = max_output_tokens
        self.temperature = 0.2
        self.top_p = 0.95
        self.top_k = 40
        self.safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
        ]
        self.system_prompt = SYSTEM_PROMPT
        self.output_schema = _build_output_schema()

    def generate_response(self, profile_input: str) -> dict[str, Any]:
        user_text = (profile_input or "").strip() + "\n"
        verbalized_need = parse_verbalized_need_from_user_message(user_text)
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_text)])]
        cfg = types.GenerateContentConfig(
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
            response_mime_type="application/json",
            response_schema=self.output_schema,
        )

        output_text = ""
        thought_summary = ""
        finish_reason = None
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r".*non-text parts in the response.*")
                warnings.filterwarnings("ignore", message=r".*automatic function calling.*")
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=cfg,
                )
            for cand in getattr(response, "candidates", None) or []:
                reason = getattr(cand, "finish_reason", None)
                if reason is not None:
                    finish_reason = reason
                content = getattr(cand, "content", None)
                if not content:
                    continue
                for part in getattr(content, "parts", None) or []:
                    t = getattr(part, "text", None)
                    if not isinstance(t, str) or not t:
                        continue
                    if getattr(part, "thought", False):
                        thought_summary = (thought_summary + t) if thought_summary else t
                    else:
                        output_text += t
            if not output_text:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=r".*non-text parts in the response.*")
                    agg = getattr(response, "text", None)
                    if isinstance(agg, str) and agg:
                        output_text = agg
        except ClientError as e:
            if self.thinking_budget == 0 and "only works in thinking mode" in (str(e) or ""):
                print(
                    "\n[NOTE] This model requires thinking mode; use default (no --no-thinking) or a different model.",
                    flush=True,
                )
                sys.exit(1)
            raise

        if thought_summary:
            print("\n" + "-" * 80)
            print("THOUGHT SUMMARY:")
            print("-" * 80)
            print(thought_summary.strip())
            print("-" * 80 + "\n")

        if not (output_text or "").strip() and not (thought_summary or "").strip():
            raise ValueError("Empty response from model. Check API key and model availability.")

        parsed: dict[str, Any] | None = None
        parse_error: Exception | None = None
        for source in (output_text, thought_summary):
            if not (source or "").strip():
                continue
            try:
                parsed = _parse_model_json_object(source)
                break
            except (json.JSONDecodeError, ValueError) as exc:
                parse_error = exc

        if parsed is None:
            reason = str(finish_reason or "unknown")
            detail = str(parse_error or "unknown parse error")
            raise ValueError(f"Invalid JSON response. finish_reason={reason!r}; {detail}") from parse_error

        cleaned = _post_process_need_update_response(parsed, verbalized_need)
        _validate_need_update_response(cleaned)
        return cleaned


TEST_CASES: list[dict[str, Any]] = [
    # 0
    {
        "name": "faster_paydown_pacing",
        "batch": 1,
        "input": """# Current Verbalized Need

- needs_title: Venture interest keeps adding to your balance 💳
- concise_need: $312 interest every 90 days on your $8,400 Venture balance. 📉
- full_need: Interest tool shows $312 on Venture in 90 days with next payment due 2026-04-18. 💸

# User Tune Message

Can we pay down the card faster by cutting meals more in the first three months?
""",
        "ideal_response": {
            "needs_update": False,
            "updated_need": None,
        },
    },
    # 1
    {
        "name": "lower_meals_cap",
        "batch": 1,
        "input": """# Current Verbalized Need

- needs_title: Venture interest keeps adding to your balance 💳
- concise_need: $312 interest every 90 days on your $8,400 Venture balance. 📉
- full_need: Interest tool shows $312 on Venture in 90 days with next payment due 2026-04-18. 💸

# User Tune Message

Keep food at $1100 instead of $1000.
""",
        "ideal_response": {
            "needs_update": False,
            "updated_need": None,
        },
    },
    # 2
    {
        "name": "set_shopping_cap",
        "batch": 1,
        "input": """# Current Verbalized Need

- needs_title: Venture interest keeps adding to your balance 💳
- concise_need: $312 interest every 90 days on your $8,400 Venture balance. 📉
- full_need: Interest tool shows $312 on Venture in 90 days with next payment due 2026-04-18. 💸

# User Tune Message

set shopping at $250.
""",
        "ideal_response": {
            "needs_update": False,
            "updated_need": None,
        },
    },
    # 3
    {
        "name": "increase_goal_not_in_need_copy",
        "batch": 1,
        "input": """# Current Verbalized Need

- needs_title: Venture interest keeps adding to your balance 💳
- concise_need: $312 interest every 90 days on your $8,400 Venture balance. 📉
- full_need: Interest tool shows $312 on Venture in 90 days with next payment due 2026-04-18. 💸

# User Tune Message

increase target goal to 850k.
""",
        "ideal_response": {
            "needs_update": False,
            "updated_need": None,
        },
    },
    # 4
    {
        "name": "shift_plan_start_month",
        "batch": 1,
        "input": """# Current Verbalized Need

- needs_title: Venture interest keeps adding to your balance 💳
- concise_need: $312 interest every 90 days on your $8,400 Venture balance. 📉
- full_need: Interest tool shows $312 on Venture in 90 days with next payment due 2026-04-18. 💸

# User Tune Message

start plan in Nov 2026 instead.
""",
        "ideal_response": {
            "needs_update": False,
            "updated_need": None,
        },
    },
    # 5
    {
        "name": "baseline_category_cut",
        "batch": 1,
        "input": """# Current Verbalized Need

- needs_title: Venture interest keeps adding to your balance 💳
- concise_need: $312 interest every 90 days on your $8,400 Venture balance. 📉
- full_need: Interest tool shows $312 on Venture in 90 days with next payment due 2026-04-18. 💸

# User Tune Message

set budget of all categories to be 25% less than baseline.
""",
        "ideal_response": {
            "needs_update": False,
            "updated_need": None,
        },
    },
    # 6
    {
        "name": "stretch_paydown_timeline",
        "batch": 1,
        "input": """# Current Verbalized Need

- needs_title: Venture interest keeps adding to your balance 💳
- concise_need: $312 interest every 90 days on your $8,400 Venture balance. 📉
- full_need: Interest tool shows $312 on Venture in 90 days with next payment due 2026-04-18. 💸

# User Tune Message

Can we stretch paydown over 12 months instead of 8?
""",
        "ideal_response": {
            "needs_update": False,
            "updated_need": None,
        },
    },
    # 7
    {
        "name": "increase_goal_target",
        "batch": 2,
        "input": """# Current Verbalized Need

- needs_title: Savings still far from target 🎯
- concise_need: You're at $412k toward the $700k milestone you set. 📉
- full_need: Portfolio sits at $412k and the plan targets a $700k milestone by 2029. 💸

# User Tune Message

increase target goal to 850k.
""",
        "ideal_response": {
            "needs_update": True,
            "updated_need": {
                "needs_title": "Savings still far from target 🎯",
                "concise_need": "You're at $412k toward the $850k milestone you set. 📉",
                "full_need": "Portfolio sits at $412k and the plan now targets an $850k milestone by 2029. 💸",
            },
        },
    },
    # 8
    {
        "name": "set_target_goal_to_800k",
        "batch": 2,
        "input": """# Current Verbalized Need

- needs_title: Savings still far from target 🎯
- concise_need: You're at $412k toward the $700k milestone you set. 📉
- full_need: Portfolio sits at $412k and the plan targets a $700k milestone by 2029. 💸

# User Tune Message

set target goal to 800k.
""",
        "ideal_response": {
            "needs_update": True,
            "updated_need": {
                "needs_title": "Savings still far from target 🎯",
                "concise_need": "You're at $412k toward the $800k milestone you set. 📉",
                "full_need": "Portfolio sits at $412k and the plan now targets an $800k milestone by 2029. 💸",
            },
        },
    },
    # 9
    {
        "name": "shift_plan_start_in_need_copy",
        "batch": 2,
        "input": """# Current Verbalized Need

- needs_title: Venture interest keeps adding to your balance 💳
- concise_need: $312 interest every 90 days on your $8,400 Venture balance. 📉
- full_need: The paydown plan starts Sep 2026 with $312 Venture interest due every 90 days. 💸

# User Tune Message

start plan in Nov 2026 instead.
""",
        "ideal_response": {
            "needs_update": True,
            "updated_need": {
                "needs_title": "Venture interest keeps adding to your balance 💳",
                "concise_need": "$312 interest every 90 days on your $8,400 Venture balance. 📉",
                "full_need": "The paydown plan starts Nov 2026 with $312 Venture interest due every 90 days. 💸",
            },
        },
    },
]


def _run_test(
    profile_input: str,
    optimizer: NeedUpdatedVerbalizerOptimizer | None = None,
    *,
    ideal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if optimizer is None:
        optimizer = NeedUpdatedVerbalizerOptimizer()
    verbalized_need = parse_verbalized_need_from_user_message(profile_input)
    print("=" * 80)
    print("LLM INPUT:")
    print("=" * 80)
    print(profile_input)
    result = optimizer.generate_response(profile_input)
    resolved = resolve_verbalized_need_after_tune(result, verbalized_need)
    print("=" * 80)
    print("LLM OUTPUT:")
    print("=" * 80)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=" * 80)
    print("RESOLVED VERBALIZED NEED:")
    print("=" * 80)
    print(json.dumps(resolved, indent=2, ensure_ascii=False))
    if ideal is not None:
        print("=" * 80)
        print("IDEAL RESPONSE:")
        print("=" * 80)
        print(json.dumps(ideal, indent=2, ensure_ascii=False))
    print("=" * 80 + "\n")
    return result


def get_test_case(test_name_or_index: str | int) -> dict[str, Any] | None:
    if isinstance(test_name_or_index, int):
        if 0 <= test_name_or_index < len(TEST_CASES):
            return TEST_CASES[test_name_or_index]
        return None
    for tc in TEST_CASES:
        if tc["name"] == test_name_or_index:
            return tc
    return None


def run_test(
    test_name_or_index_or_dict: str | int | dict[str, Any],
    optimizer: NeedUpdatedVerbalizerOptimizer | None = None,
) -> dict[str, Any] | None:
    if optimizer is None:
        optimizer = NeedUpdatedVerbalizerOptimizer()

    if isinstance(test_name_or_index_or_dict, dict):
        tc = test_name_or_index_or_dict
        name = tc.get("name", "custom_test")
        raw_input = tc.get("input")
        if not isinstance(raw_input, str) or not raw_input.strip():
            print("Invalid test dict: missing input")
            return None
        print(f"\n{'=' * 80}\nRunning test: {name}\n{'=' * 80}\n")
        ideal = _resolve_ideal_response(tc)
        return _run_test(raw_input, optimizer, ideal=ideal)

    tc = get_test_case(test_name_or_index_or_dict)
    if tc is None:
        print(f"Test case '{test_name_or_index_or_dict}' not found.")
        return None
    print(f"\n{'=' * 80}\nRunning test: {tc['name']}\n{'=' * 80}\n")
    ideal = _resolve_ideal_response(tc)
    return _run_test(tc["input"], optimizer, ideal=ideal)


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Run {_TEMPLATE_NAME} optimizer tests")
    parser.add_argument("--test", type=str, help='Test name or index (e.g. "0" or "plan_pacing_only")')
    parser.add_argument("--batch", type=int, help="Run all tests in batch N")
    parser.add_argument("--input-file", type=str, help="Path to a custom multiline input file")
    parser.add_argument(
        "--print-input-only",
        action="store_true",
        help="Only print test input (no model call)",
    )
    parser.add_argument("--model", type=str, default=GEMINI_FLASH_LITE)
    parser.add_argument("--no-thinking", action="store_true", help="Set thinking_budget=0")
    args = parser.parse_args()

    if args.input_file:
        profile_input = Path(args.input_file).read_text(encoding="utf-8")
        if args.print_input_only:
            print(profile_input)
            return
        thinking_budget = 0 if args.no_thinking else THINKING_BUDGET
        optimizer = NeedUpdatedVerbalizerOptimizer(
            model_name=args.model,
            thinking_budget=thinking_budget,
        )
        _run_test(profile_input, optimizer)
        return

    if args.batch is None and args.test is None:
        print("Usage:")
        print("  --test <name_or_index> | --test all")
        print("  --batch <N>")
        print("  --input-file <path>")
        for i, tc in enumerate(TEST_CASES):
            print(f"  [{i}] {tc['name']} (batch {tc.get('batch', '?')})")
        return

    if args.test is not None and args.print_input_only and args.test.strip().lower() != "all":
        test_val: str | int = int(args.test) if args.test.isdigit() else args.test
        tc = get_test_case(test_val)
        if tc is None:
            raise SystemExit(f"Test case '{args.test}' not found.")
        print(tc["input"])
        return

    thinking_budget = 0 if args.no_thinking else THINKING_BUDGET
    optimizer = NeedUpdatedVerbalizerOptimizer(
        model_name=args.model,
        thinking_budget=thinking_budget,
    )

    if args.batch is not None:
        cases = [tc for tc in TEST_CASES if int(tc.get("batch") or 0) == int(args.batch)]
        if not cases:
            raise SystemExit(f"No tests found for batch={args.batch}")
        for i, tc in enumerate(cases):
            if i:
                print("\n" + "-" * 80 + "\n")
            run_test(tc, optimizer)
        return

    if args.test is not None:
        if args.test.strip().lower() == "all":
            for i in range(len(TEST_CASES)):
                run_test(i, optimizer)
                if i < len(TEST_CASES) - 1:
                    print("\n" + "-" * 80 + "\n")
            return
        test_val: str | int = int(args.test) if args.test.isdigit() else args.test
        run_test(test_val, optimizer)


if __name__ == "__main__":
    main()
