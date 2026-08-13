"""
Optimizer runner for **P:NeedPlanEmailVerbalizer** (Gemini prompt tuning).

Input is verbalized ``# Financial Need`` plus ``# Your Plan`` from ``user_plans``
(need + plan verbalizer output for one plan).

Output: ``email_subject``, ``need_tldr``, ``body_text``; ``plan_chart`` is attached
post-LLM from ``verbalized_plan`` (not model-generated).

Run from finance-ai-penny repo root:

  python3 active_experiments/need_plan_email_verbalizer_optimizer.py --test 0
  python3 active_experiments/need_plan_email_verbalizer_optimizer.py --test all
  python3 active_experiments/need_plan_email_verbalizer_optimizer.py --user-id 3 --plan-id 14 --print-input-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from active_experiments.verbalizer_optimizer_db import (
    _LLM_SERVER_ROOT,
    _load_slave_db_connect_kwargs,
    _resolve_ideal_response,
)

if str(_LLM_SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(_LLM_SERVER_ROOT))

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

from propose_next_steps.need_plan_email_bundle import build_need_plan_email_verbalizer_input
from propose_next_steps.need_plan_email_verbalizer_schemas import (
    validate_need_plan_email_verbalizer_output,
)

if load_dotenv is not None:
    load_dotenv()

GEMINI_FLASH_LITE = "gemini-3.1-flash-lite"
TEMPLATE_NAME = "P:NeedPlanEmailVerbalizer"
_THINKING_BUDGET = 128
_MAX_OUTPUT_TOKENS = 2048

SUBJECT_MAX_CHARS = 60
SUBJECT_MAX_WORDS = 8
_NEED_TLDR_MAX_CHARS = 130
_NEED_TLDR_MAX_WORDS = 20
_BODY_MAX_WORDS = 90

SYSTEM_PROMPT = """You are Penny — a warm, direct personal finance coach emailing a user that their **new plan is ready**.

**Goal:** Write copy that feels like a clear roadmap, not a data dump. Validate the need, show Penny already did the heavy lifting, and make the finish line feel visible and doable. The email includes a projection chart below the copy — reference the finish line, do not describe chart axes.

**Input:** ``# Financial Need``, ``# Your Plan`` with ``## What this plan does``, ``## Key adjustments``, and ``## Finish line``. Ground every **$** only in the input. Do not invent figures. For when the goal is reached, use ``## Finish line`` (simulation result) — not month counts from ``## What this plan does``.

Return ``email_subject``, ``need_tldr``, and ``body_text`` only.

- ``need_tldr`` is the **email hero headline** — one outcome-focused, empathetic line about what this plan makes possible. Lead with the plan promise, not a stat recitation.
- ``body_text`` is **exactly 2 short paragraphs** separated by a blank line. Paragraph 1: acknowledge the need without lecturing; Penny mapped a step-by-step path. Paragraph 2: what the plan does in plain language, one concrete adjustment from ``## Key adjustments``, and point to the finish line from ``## Finish line``. Invite the user to review the setup — no homework.
- Use **whole dollars with commas** when citing amounts. Cap at **2 dollar figures** in ``body_text``.
- Warm, conversational, scannable. No corporate filler ("creates a new reality", "navigate this transition", "cash flow").
- No greeting (Hi, Hello) and no sign-off (Thanks, Best).
- No imperatives or homework (you should, try to, consider).
- No exclamation marks.
- Do not use the word budget.
"""


def _build_output_schema() -> "types.Schema":
    if types is None:
        raise RuntimeError("Install `google-genai` for this optimizer.")
    return types.Schema(
        type=types.Type.OBJECT,
        required=["email_subject", "need_tldr", "body_text"],
        property_ordering=["email_subject", "need_tldr", "body_text"],
        properties={
            "email_subject": types.Schema(
                type=types.Type.STRING,
                description=(
                    "Inbox subject (max 8 words, 60 characters, no period, exactly 1 emoji at end). "
                    "Lead with plan outcome or strongest need hook."
                ),
            ),
            "need_tldr": types.Schema(
                type=types.Type.STRING,
                description=(
                    "Email hero headline (max 20 words, 130 characters). Outcome-focused roadmap line; "
                    "empathetic, not a stat dump. No emoji required."
                ),
            ),
            "body_text": types.Schema(
                type=types.Type.STRING,
                description=(
                    "Exactly 2 paragraphs separated by two newline characters; max 90 words total; "
                    "second person. Para 1: validate need + Penny built the path. Para 2: plan outcome, "
                    "one key adjustment, finish line. Max 2 dollar figures. Max 2 emojis."
                ),
            ),
        },
    )


def _parse_model_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Response must be a JSON object")
    return parsed


def _collect_model_response(response: Any) -> tuple[str, str, Any]:
    output_text = ""
    thought_summary = ""
    finish_reason = None
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
    return output_text, thought_summary, finish_reason


def format_need_plan_email_user_message(profile_input: str) -> str:
    body = (profile_input or "").strip()
    if not body:
        raise ValueError("profile_input must be non-empty markdown.")
    if "# Financial Need" not in body:
        raise ValueError("profile_input must include # Financial Need.")
    if "# Your Plan" not in body:
        raise ValueError("profile_input must include # Your Plan.")
    return body + "\n"


class NeedPlanEmailVerbalizerOptimizer:
    def __init__(
        self,
        model_name: str = GEMINI_FLASH_LITE,
        *,
        thinking_budget: int = _THINKING_BUDGET,
        max_output_tokens: int = _MAX_OUTPUT_TOKENS,
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
        self.temperature = 0.35
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
        user_text = format_need_plan_email_user_message(profile_input)
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
                include_thoughts=False,
            ),
            response_mime_type="application/json",
            response_schema=self.output_schema,
        )
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=cfg,
            )
        except ClientError as e:
            if self.thinking_budget == 0 and "only works in thinking mode" in (str(e) or ""):
                print("\n[NOTE] This model requires thinking mode.\n", flush=True)
                sys.exit(1)
            raise

        output_text, _, finish_reason = _collect_model_response(response)
        if not (output_text or "").strip():
            raise ValueError(f"Empty JSON response from model. finish_reason={finish_reason!r}")

        raw = output_text.strip()
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Invalid JSON response from model.")
        parsed = _parse_model_json_object(raw[start:end + 1])
        return validate_need_plan_email_verbalizer_output(parsed)


TEST_CASES: list[dict[str, Any]] = [
    {
        "name": "credit_paydown_gradual",
        "batch": 1,
        "input": """
# Financial Need

**Interest keeps stacking on $8,400** 💳

$312 in interest every 90 days on your $8,400 balance while spending tracks income.

## Need Details

Interest tool: **$312** on Venture in 90 days. Next due **2026-04-18** per payment schedule. 📉

# Your Plan

**Gradual paydown** (Gentle)

## What this plan does

Pay Venture to $0 with phased cuts, then save $200/mo.

## Key adjustments

- Food: $1,000 today → $850 (15% cut), then $700 3 months later
- Leisure: $500 today → $450 (10% cut), then $350 3 months later

## Finish line

- **Interest-free credit (card balance $0)**: achieved By Oct 31, 2026.
- Projection: 5 mo, stop goal achieved.
""",
        "ideal_response": {
            "email_subject": "Your roadmap to zero is ready 💳",
            "need_tldr": "A realistic roadmap to pay Venture down without sacrificing your sanity",
            "body_text": (
                "Money gets overwhelming when there is no clear finish line — Penny mapped a "
                "step-by-step path that shows when your Venture balance gets cleared.\n\n"
                "Gradual paydown phases food and leisure cuts so the card reaches $0 over about "
                "12 months. Review the setup below and see if this path feels doable for you."
            ),
        },
    },
    {
        "name": "emergency_savings_target",
        "batch": 1,
        "input": """
# Financial Need

**Savings gap to $6,000 buffer** 🏦

You want an emergency buffer of **$6,000**, but your current savings is only **$1,000**.

## Need Details

Savings gap is **$5,000** to reach **$6,000**. Committed spend leaves little slack. 🌱

# Your Plan

**Emergency fund target** (Focused)

## What this plan does

Save $6,000 by keeping food at $520/mo and leisure at $300/mo.

## Key adjustments

- Food: $650 today → $520 (20% cut)
- Leisure: $400 today → $300 (25% cut)
- Shopping: $80 today → $50 (38% cut)

## Finish line

- **Emergency fund ($6,000)**: achieved By Jun 30, 2027.
- Projection: 12 mo, stop goal achieved.
""",
        "ideal_response": {
            "email_subject": "Your savings roadmap is ready 🏦",
            "need_tldr": "A steady path to a $6,000 cushion without guessing each month",
            "body_text": (
                "Building a safety net is easier when the steps are already laid out — Penny "
                "structured a path from $1,000 toward $6,000 in savings.\n\n"
                "Emergency fund target trims food and leisure first while holding essentials steady. "
                "The chart below shows how savings can grow over about 12 months if you follow this setup."
            ),
        },
    },
    {
        "name": "income_drop_liquidity",
        "batch": 2,
        "input": """
# Financial Need

**Income dropped sharply** 📉

Your monthly income decreased significantly over the last three months.

## Need Details

Your monthly income shifted from $86,313 in May to $20,195 in July, impacting your cash flow. 📉

# Your Plan

**Adaptive Income Stabilization Plan** (Balanced)

## What this plan does

Adjust spending to your new $20,195 income. Reach a $40,000 liquidity target by transitioning to stricter caps starting in month four.

## Key adjustments

- Food: $1,862 today → $4,000 (115% up), then $3,000 4 months later
- Leisure: $1,874 today → $6,000 (220% up), then $3,000 4 months later
- Shopping: $633 today → $1,000 (58% up), then $600 4 months later

## Finish line

- **Savings ≥ $40,000 with credit $0**: achieved By Sep 30, 2026 ($658,333 checking).
- Projection: 1 mo, stop goal achieved.
""",
        "ideal_response": {
            "email_subject": "Your stability plan is ready 📈",
            "need_tldr": "A realistic roadmap to rebuild liquidity after your income shift",
            "body_text": (
                "When income drops sharply, the hard part is not knowing what still fits — Penny "
                "mapped a step-by-step path aligned to your $20,195 monthly income.\n\n"
                "Adaptive Income Stabilization pays down credit and reaches your $40,000 liquidity "
                "target by Sep 30, 2026. Review the projection chart below to see the finish line."
            ),
        },
    },
]


def resolve_test_case_input(test_case: dict[str, Any]) -> str:
    raw = test_case.get("input")
    if isinstance(raw, str) and raw.strip():
        return raw.strip() + "\n"
    raise ValueError("test case must include non-empty string input")


def _run_test(
    profile_input: str,
    optimizer: NeedPlanEmailVerbalizerOptimizer | None = None,
    *,
    ideal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if optimizer is None:
        optimizer = NeedPlanEmailVerbalizerOptimizer()
    wrapped = format_need_plan_email_user_message(profile_input)
    print("=" * 80)
    print("LLM INPUT:")
    print("=" * 80)
    print(wrapped)
    result = optimizer.generate_response(profile_input)
    print("=" * 80)
    print("LLM OUTPUT:")
    print("=" * 80)
    print(json.dumps(result, indent=2, ensure_ascii=False))
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
    optimizer: NeedPlanEmailVerbalizerOptimizer | None = None,
) -> dict[str, Any] | None:
    if optimizer is None:
        optimizer = NeedPlanEmailVerbalizerOptimizer()
    if isinstance(test_name_or_index_or_dict, dict):
        tc = test_name_or_index_or_dict
        name = tc.get("name", "custom_test")
        try:
            payload = resolve_test_case_input(tc)
        except ValueError as exc:
            print(f"Invalid test dict: {exc}")
            return None
        print(f"\n{'=' * 80}\nRunning test: {name}\n{'=' * 80}\n")
        ideal = _resolve_ideal_response(tc)
        return _run_test(payload, optimizer, ideal=ideal)
    tc = get_test_case(test_name_or_index_or_dict)
    if tc is None:
        print(f"Test case '{test_name_or_index_or_dict}' not found.")
        return None
    print(f"\n{'=' * 80}\nRunning test: {tc['name']}\n{'=' * 80}\n")
    ideal = _resolve_ideal_response(tc)
    return _run_test(resolve_test_case_input(tc), optimizer, ideal=ideal)


def _load_input_from_db(*, user_id: int, plan_id: int) -> str:
    try:
        import psycopg2
    except Exception as exc:
        raise RuntimeError("Missing dependency `psycopg2`.") from exc
    conn = psycopg2.connect(**_load_slave_db_connect_kwargs())
    try:
        profile_input, _ = build_need_plan_email_verbalizer_input(
            conn,
            user_id=int(user_id),
            plan_id=int(plan_id),
        )
        return profile_input
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Run {TEMPLATE_NAME} optimizer tests")
    parser.add_argument("--test", type=str, help="Test name or index")
    parser.add_argument("--batch", type=int, help="Run all tests in batch N")
    parser.add_argument("--user-id", type=int, help="User id for DB-backed input")
    parser.add_argument("--plan-id", type=int, help="user_plans.plan_id for DB-backed input")
    parser.add_argument("--print-input-only", action="store_true")
    parser.add_argument("--model", type=str, default=GEMINI_FLASH_LITE)
    parser.add_argument("--no-thinking", action="store_true")
    args = parser.parse_args()

    if args.user_id is not None and args.plan_id is not None:
        built = _load_input_from_db(user_id=args.user_id, plan_id=args.plan_id)
        print(f"Using user_id={args.user_id} plan_id={args.plan_id}")
        print("BUILT NEED PLAN EMAIL INPUT")
        print("-" * 80)
        print(built)
        if args.print_input_only:
            return
        optimizer = NeedPlanEmailVerbalizerOptimizer(
            model_name=args.model,
            thinking_budget=0 if args.no_thinking else _THINKING_BUDGET,
        )
        print("\nLLM OUTPUT")
        print("-" * 80)
        print(json.dumps(optimizer.generate_response(built), indent=2, ensure_ascii=False))
        return

    if args.print_input_only:
        print("Error: --print-input-only requires --user-id and --plan-id", file=sys.stderr)
        raise SystemExit(1)

    thinking_budget = 0 if args.no_thinking else _THINKING_BUDGET
    optimizer = NeedPlanEmailVerbalizerOptimizer(
        model_name=args.model,
        thinking_budget=thinking_budget,
    )

    if args.batch is not None:
        cases = [tc for tc in TEST_CASES if int(tc.get("batch") or 0) == int(args.batch)]
        if not cases:
            raise SystemExit(f"No tests found for batch={args.batch}")
        for tc in cases:
            run_test(tc, optimizer)
        return

    if args.test is not None:
        if args.test.strip().lower() == "all":
            for i in range(len(TEST_CASES)):
                run_test(i, optimizer)
            return
        test_val: str | int = int(args.test) if args.test.isdigit() else args.test
        run_test(test_val, optimizer)
        return

    print("Usage:")
    print("  --test <name_or_index>   Run one bundled test case")
    print("  --test all               Run all bundled test cases")
    print("  --batch N                Run tests tagged with batch N")
    print("  --user-id ID --plan-id ID  Build input from user_plans")
    print("  --print-input-only       Print LLM input without calling the model")
    print("\nAvailable test cases:")
    for i, tc in enumerate(TEST_CASES):
        print(f"  [{i}] {tc['name']} (batch {tc.get('batch', '?')})")


if __name__ == "__main__":
    main()
