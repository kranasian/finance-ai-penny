"""
Second-stage optimizer: turns **Top Takeaways** markdown (the artifact produced by
``active_experiments/top_takeaways_verbose_optimizer.py`` — ``# Top Takeaways``, ``## Highlights``, ``## Lowlights``)
into **TLDR only**: ``# TLDR`` (H1), then ``## Highlights`` and ``## Lowlights`` with ``- `` bullets.
The model reply must not add any headings or sections after ``## Lowlights``.

Run from ``finance-ai-penny`` repo root:

  python3 active_experiments/top_takeaways_tldr_optimizer.py --test 0
  python3 active_experiments/top_takeaways_tldr_optimizer.py --test all

Requires ``GEMINI_API_KEY`` and ``google-genai``. Thinking stays on (budget 256).
"""

from __future__ import annotations

import argparse
import os
import warnings
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

try:
    from google import genai
    from google.genai import types
except Exception:  # pragma: no cover
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]

if load_dotenv is not None:
    load_dotenv()

GEMINI_FLASH_LITE = "gemini-flash-lite-latest"
TLDR_THINKING_BUDGET = 256


SYSTEM_PROMPT = """You are **Penny**.

# Input 
The input contains **Top Takeaways** markdown: ``# Top Takeaways``, ``## Highlights``, and ``## Lowlights`` with ``- `` bullets.

# Output
Summarize the input into ultra-compact TLDR (“too long; didn’t read”) bullets that can be skimmed in a few seconds.

Follow the structure below. Do not add a second H1, a second copy of Highlights/Lowlights, or any other headings or sections—your last message ends after the ``## Lowlights`` bullets.

# TLDR

## Highlights

- ...

## Lowlights

- ...

# Guidelines
- **Truth:** Ground everything in the input only. Do not invent facts or instructions.
- **Length:** Aim ≤18 words per bullet; never exceed 22 words.
- **Shape:** Exactly one short sentence per bullet. Focus on one point then drop the rest.
- **Date/Number Format:** Copy dates and numbers as they are in the input.
- **Category Links:** When mentioning a category, copy the links used for the category in the input.
"""


def format_tldr_user_message(top_takeaways_markdown: str) -> str:
    """Wrap verbose-optimizer output so the model sees a labeled Top Takeaways document."""
    body = (top_takeaways_markdown or "").strip()
    if not body:
        raise ValueError("top_takeaways_markdown must be non-empty.")
    return body + "\n"


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


class TopTakeawaysTldrOptimizer:
    """Gemini one-shot: Top Takeaways markdown → ``# TLDR`` with ``## Highlights`` and ``## Lowlights`` only."""

    def __init__(
        self,
        model_name: str = GEMINI_FLASH_LITE,
        *,
        thinking_budget: int = TLDR_THINKING_BUDGET,
        max_output_tokens: int = 3072,
    ):
        if genai is None or types is None:  # pragma: no cover
            raise RuntimeError("Install `google-genai` (and optionally `python-dotenv`).")
        if not isinstance(thinking_budget, int) or thinking_budget < 1:
            raise ValueError("thinking_budget must be a positive int.")
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

    def generate_markdown(self, top_takeaways_markdown: str) -> str:
        user_message = format_tldr_user_message(top_takeaways_markdown)
        request_text = types.Part.from_text(text=user_message)
        contents = [types.Content(role="user", parts=[request_text])]
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
        )
        out: list[str] = []
        thought_summary = ""
        for chunk in self.client.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=cfg,
        ):
            piece = _extract_stream_chunk_text(chunk)
            if piece:
                out.append(piece)
            if hasattr(chunk, "candidates") and chunk.candidates:
                for candidate in chunk.candidates:
                    if hasattr(candidate, "content") and candidate.content:
                        if hasattr(candidate.content, "parts") and candidate.content.parts:
                            for part in candidate.content.parts:
                                if getattr(part, "thought", False) and getattr(part, "text", None):
                                    t = part.text
                                    thought_summary = (thought_summary + t) if thought_summary else t
        if thought_summary:
            print("\n" + "-" * 80 + "\nTHOUGHT SUMMARY:\n" + "-" * 80 + "\n" + thought_summary.strip() + "\n" + "-" * 80 + "\n")
        text = "".join(out).strip()
        if not text:
            raise ValueError("Empty model response.")
        return text


TEST_CASES: list[dict[str, Any]] = [
    {
        "name": "from_verbose_style_four_themes",
        "input": """
# Top Takeaways

## Highlights

- Your early-month [Salary](/cashflow/36/monthly/2026-05) of $6,369.24 is consistent with your regular pay cycle, mirroring the amount received during the same period in April. The significant variance compared to March ($17,369.24) is fully explained by the one-time $8,800 "Genentech US Bonus" and higher "CA State Payroll" amounts received in the first half of March.
- While weekly spikes occur, your overall [Uncategorized](/cashflow/-1/monthly/2026-05) spending is trending downward this month at $6,644.34 compared to $7,681.54 in April. This improvement is driven by a lower volume of miscellaneous transactions—18 in May versus 28 in April—rather than changes to major recurring charges like the $5,700 monthly payment to "Property Group LLC."

## Lowlights

- A recent surge in [Uncategorized](/cashflow/-1/weekly/2026-05-03) spending to $6,382.12 for the week of May 3–9 is largely driven by two $2,850 payments to "Property Group LLC." Additionally, the presence of duplicate entries for recurring charges like "Community Pool" ($83.06 x 2), "Costco" ($69.71 x 2), and "BP" ($49.56 x 2) suggests potential data reconciliation issues across your accounts.
- Your [Groceries](/cashflow/4/monthly/2026-05) spending appears to contain miscategorized dining expenses, such as an $85.36 charge at "Applebee's" recorded on May 2nd. While total food spending is down to $1,859.34 for the first 10 days of May, these classification errors obscure the true distinction between your grocery budget and dining out costs.
""",
        "output": """
# TLDR

## Highlights

- **Salary:** Early [Salary](/cashflow/36/monthly/2026-05) matches April at $6,369.24 while March looked higher on a one-time bonus and payroll timing.
- **Uncategorized:** [Uncategorized](/cashflow/-1/monthly/2026-05) is at $6,644.34 in early May with fewer small transactions than April’s $7,681.54.

## Lowlights

- **Uncategorized week:** [Uncategorized](/cashflow/-1/weekly/2026-05-03) spiked to about $6,382 on Property Group LLC payments and possible duplicate recurring charges.
- **Groceries:** [Groceries](/cashflow/4/monthly/2026-05) mixes Applebee’s dining into grocery totals with early-May food spend near $1,859.
""",
    },
    {
        "name": "from_verbose_style_two_themes",
        "input": """
# Top Takeaways

## Highlights

- May 1–10 [Salary](/cashflow/36/monthly/2026-05) at $6,369.24 aligns with April’s same early-month window at the same dollar level. March’s higher early-month figure reflected one-time Genentech bonus and higher CA State Payroll in early March, not a structural pay cut in your regular rhythm.

## Lowlights

- [Groceries](/cashflow/4/monthly/2026-05) early-May totals look lower partly because only 10 days of May are in the window compared with full April and March months. Applebee’s dining charges (e.g. $85.36 on May 2) still appear categorized as groceries, which skews grocery totals versus true dining spend.
""",
        "output": """
# TLDR

## Highlights

- **Income:** May 1–10 [Salary](/cashflow/36/monthly/2026-05) tracks April’s early-month level with March higher on bonus timing.

## Lowlights

- **Groceries mix:** [Groceries](/cashflow/4/monthly/2026-05) looks down on a short May window and may tuck Applebee’s dining under groceries.
""",
    },
    {
        "name": "uncat_txn_and_large_txn_transaction_taps",
        "input": """
# Top Takeaways

## Highlights

- While weekly spikes occur, your overall [Uncategorized](/cashflow/-1/monthly/2026-05) spending is trending downward this month at $6,644.34 compared to $7,681.54 in April, with fewer miscellaneous transactions in early May even as large Property Group LLC charges remain.

## Lowlights

- A recent surge in [Uncategorized](/cashflow/-1/weekly/2026-05-03) spending to $6,382.12 for the week of May 3–9 is largely driven by two $2,850 payments to "Property Group LLC," with duplicate-looking lines for Community Pool, Costco, and BP suggesting reconciliation noise.
- A $2,850 [Property Group LLC](/cashflow/transaction/123) on May 5 repeats a similar May 3 hit, so the week’s spike is concentrated in a few big hits rather than many small unknowns.
""",
        "output": """
# TLDR

## Highlights

- **Uncategorized:** [Uncategorized](/cashflow/-1/monthly/2026-05) is down vs April at $6,644.34 with fewer small transactions while large Property Group LLC repeats remain.

## Lowlights

- **Uncategorized week:** [Uncategorized](/cashflow/-1/weekly/2026-05-03) spiked to about $6,382 on two $2,850 Property Group LLC hits and possible duplicate recurring charges.
- **Large transaction:** The $2,850 [Property Group LLC](/cashflow/transaction/123) echoes May 3 Property Group LLC so the week clusters on few big hits.
""",
    },
    {
        "name": "four_vs_forecast_plus_uncat_and_large_txn_transaction_taps",
        "input": """
# Top Takeaways

## Highlights

- Your early-month [Salary](/cashflow/36/monthly/2026-05) of $6,369.24 is consistent with your regular pay cycle, mirroring the amount received during the same period in April, while March’s higher early-month income reflected one-time bonus and payroll lifts rather than a structural pay change.
- Your [Groceries](/cashflow/4/monthly/2026-05) spending appears to contain miscategorized dining expenses, such as an $85.36 charge at "Applebee's" recorded on May 2nd, which skews grocery totals versus true dining spend in early May.
- **Forecast vs actual:** [Groceries](/cashflow/4/monthly/2026-05) is tracking a few hundred dollars below your typical monthly pace, while [Food](/cashflow/1/monthly/2026-05) is roughly on forecast for the month-to-date window.

## Lowlights

- A recent surge in [Uncategorized](/cashflow/-1/weekly/2026-05-03) spending to $6,382.12 for the week of May 3–9 is largely driven by two $2,850 payments to "Property Group LLC," with duplicate-looking lines for Community Pool, Costco, and BP suggesting reconciliation noise.
- While weekly spikes occur, your overall [Uncategorized](/cashflow/-1/monthly/2026-05) spending is trending downward this month at $6,644.34 compared to $7,681.54 in April, with fewer miscellaneous transactions in early May even as large Property Group LLC charges remain.
- A $2,850 [Property Group LLC](/cashflow/transaction/123) on May 5 repeats a similar May 3 hit, so the week’s spike is concentrated in a few big hits rather than many small unknowns.
""",
        "output": """
# TLDR

## Highlights

- **Salary:** Early [Salary](/cashflow/36/monthly/2026-05) matches April at $6,369.24 while March looked higher on bonus and payroll timing.
- **Groceries mix:** [Groceries](/cashflow/4/monthly/2026-05) tucks Applebee’s dining into grocery totals in early May.
- **Forecast vs actual:** [Groceries](/cashflow/4/monthly/2026-05) runs a few hundred under typical pace while [Food](/cashflow/1/monthly/2026-05) sits near forecast month to date.

## Lowlights

- **Uncategorized week:** [Uncategorized](/cashflow/-1/weekly/2026-05-03) spiked to about $6,382 on two $2,850 Property Group LLC hits and possible duplicate recurring charges.
- **Uncategorized month:** [Uncategorized](/cashflow/-1/monthly/2026-05) is down vs April at $6,644.34 with fewer small transactions while large Property Group LLC repeats remain.
- **Large transaction:** The $2,850 [Property Group LLC](/cashflow/transaction/123) echoes May 3 Property Group LLC so the week clusters on few big hits.
""",
    },
]

def _run_test(top_takeaways_md: str, optimizer: TopTakeawaysTldrOptimizer | None = None) -> str:
    if optimizer is None:
        optimizer = TopTakeawaysTldrOptimizer()
    wrapped = format_tldr_user_message(top_takeaways_md)
    print("TLDR LLM INPUT")
    print("-" * 80)
    print(wrapped)
    result = optimizer.generate_markdown(top_takeaways_md)
    print("TLDR LLM OUTPUT")
    print("-" * 80)
    print(result)
    return result


def get_test_case(name_or_index: str | int) -> dict[str, Any] | None:
    if isinstance(name_or_index, int):
        if 0 <= name_or_index < len(TEST_CASES):
            return TEST_CASES[name_or_index]
        return None
    for tc in TEST_CASES:
        if tc["name"] == name_or_index:
            return tc
    return None


def run_test(name_or_index: str | int, optimizer: TopTakeawaysTldrOptimizer | None = None) -> str | None:
    tc = get_test_case(name_or_index) if not isinstance(name_or_index, dict) else name_or_index
    if tc is None:
        print(f"Test case {name_or_index!r} not found.")
        return None
    print(f"\n{'=' * 80}\nRunning test: {tc['name']}\n{'=' * 80}\n")
    return _run_test(tc["input"], optimizer)


def main(*, test: str | None) -> None:
    if test is None:
        print("Usage: python3 active_experiments/top_takeaways_tldr_optimizer.py --test <index|name|all>")
        print("Tests:")
        for i, tc in enumerate(TEST_CASES):
            print(f"  {i}: {tc['name']}")
        return
    optimizer = TopTakeawaysTldrOptimizer(thinking_budget=TLDR_THINKING_BUDGET)
    if test.strip().lower() == "all":
        for i, tc in enumerate(TEST_CASES):
            run_test(i, optimizer)
            if i < len(TEST_CASES) - 1:
                print()
        return
    key: str | int = int(test) if test.isdigit() else test
    run_test(key, optimizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Top Takeaways → # TLDR with Highlights/Lowlights only (Gemini one-shot)")
    parser.add_argument("--test", type=str, default=None, help='Index, test name, or "all"')
    args = parser.parse_args()
    main(test=args.test)
