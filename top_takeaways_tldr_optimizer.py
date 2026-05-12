"""
Second-stage optimizer: turns **Top Takeaways** markdown (the artifact produced by
``top_takeaways_verbose_optimizer.py`` — ``# Top Takeaways``, ``## Highlights``, ``## Lowlights``)
into a document that opens with ``# TLDR`` (H1), then ``## Highlights`` / ``## Lowlights``, then ``# Details`` with the same pair—each subsection uses ``- `` bullets.

Run from ``finance-ai-penny`` repo root:

  python3 top_takeaways_tldr_optimizer.py --test 0
  python3 top_takeaways_tldr_optimizer.py --test all

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


def format_tldr_user_message(top_takeaways_markdown: str) -> str:
    """Wrap verbose-optimizer output so the model sees a labeled Top Takeaways document."""
    body = (top_takeaways_markdown or "").strip()
    if not body:
        raise ValueError("top_takeaways_markdown must be non-empty.")
    return body + "\n"


SYSTEM_PROMPT = """You are **Penny**.

The user message contains **Top Takeaways** markdown: ``# Top Takeaways``, ``## Highlights``, and ``## Lowlights`` with ``- `` bullets. This is produced by an upstream rollup step.

Your task is to rewrite it into **TLDR** and **Details**, each mirroring the source shape with **Highlights** and **Lowlights** subsections. The **first line** of your reply must be the H1 ``# TLDR`` (not ``# Top Takeaways``), then the tree below.

# TLDR

## Highlights

- ...

## Lowlights

- ...

# Details

## Highlights

- ...

## Lowlights

- ...

**TLDR (true “too long; didn’t read”):** **Ultra-compact** — what someone skims in a few seconds. Each bullet: **at most one short sentence**; lead with the **headline outcome** (what moved / what matters); **no** multi-clause stacks, **no** semicolon laundry lists, **no** repeating every merchant or figure line from the source (that belongs in Details). If a bullet could be dropped into Details with only light editing, it is **too long** for TLDR — cut it down. **Whenever the source bullet you are compressing includes a Markdown drill-down ``[text](/path)``, the TLDR bullet for that theme must still include at least one such link with the same ``href``** (anchor text may be tightened); keep the link **inside** the sentence (same **Link in sentence** idea as upstream: do not tack ``[Label](path)`` alone after a final period). Do **not** strip links from TLDR only. Aim **~25 words or fewer** per TLDR bullet when feasible.

**Details:** The **full** story: multiple sentences allowed; amounts, merchants, date ranges, comparisons, and nuance from the source. Each Details bullet should be **clearly longer and denser** than its TLDR counterpart for the same theme. Keep content aligned with the same subsection (highlights vs lowlights) as in the source.

Rules:
- Ground everything in the provided Top Takeaways text only. Do **not** invent transactions, amounts, or dates.
- When you cite numbers or dates, copy them exactly from the input.
- When the input includes Markdown drill-down links ``[text](/path)``, **preserve them** in TLDR and Details (same ``href`` verbatim); you may tighten anchor text but do not drop or change URLs and do not insert a space after ``(`` in links. In TLDR, keep links **inside** the sentence (not a lone trailing link after ``.``). **Do not** omit links from TLDR while keeping them in Details for the same theme.
- For bullets about **uncategorized- or large-transaction** themes (typically ``/cashflow/transaction/…`` links on those flows), keep the same **observational** tone as vs-forecast rollup bullets in the source: **do not** add action items or imperatives (“confirm”, “verify”, “should”, “until you act”) unless the input states them verbatim.
- **Bullet budget:** Under ``# TLDR``, at most **3** ``- `` bullets under ``## Highlights`` and at most **3** under ``## Lowlights``. Under ``# Details``, at most **3** under each of ``## Highlights`` and ``## Lowlights``. Use fewer if the source is thin; never exceed 3 in any subsection. Include all four ``##`` subsections (two under each H1) even when a bucket needs only one bullet or is minimal.
- Output **only** this heading tree in order: ``# TLDR`` (H1, first line) → ``## Highlights`` → ``## Lowlights`` → ``# Details`` → ``## Highlights`` → ``## Lowlights``. Do **not** emit ``# Top Takeaways`` in your reply; no greeting or sign-off; do not add any other headings beyond those six lines.
- Under each ``##`` subsection use **only** ``- `` bullets (no numbered lists).
- Preserve bold labels (``**Label:**``) when helpful; match the tone of the source (direct, friendly).
"""


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
    """Gemini one-shot: Top Takeaways markdown → ``# TLDR`` / ``# Details`` with ``## Highlights`` and ``## Lowlights`` under each."""

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

- **Salary rhythm:** Early-month [salary](/cashflow/36/monthly/2026-05) through May 10 matches the same window in April; the earlier “drop vs March” reflects a one-time bonus and payroll timing in March, not a pay cut.
- **Food spend:** [Meals](/cashflow/1/monthly/2026-05) totals are lower in early May partly because only 10 days are counted vs full prior months; drivers also flag Applebee’s charges categorized as groceries.

## Lowlights

- **Uncategorized spike:** Last week [Uncategorized](/cashflow/-1/weekly/2026-05-03) jumped to about $6,382, driven heavily by two Property Group LLC payments ($2,850 each) and possible duplicate recurring charges (Community Pool, Costco, BP).
- **Uncategorized trend:** Month-to-date [Uncategorized](/cashflow/-1/monthly/2026-05) is slightly down vs March/April; large recurring Property Group LLC charges still dominate, with fewer small miscellaneous transactions so far in May.
""",
        "output": """
# TLDR

## Highlights

- **Salary:** Early [salary](/cashflow/36/monthly/2026-05) matches April; March was higher on bonus timing, not a pay cut.
- **Food:** [Meals](/cashflow/1/monthly/2026-05) looks lower partly because May is only 10 days vs full prior months.

## Lowlights

- **Uncategorized week:** [Uncategorized](/cashflow/-1/weekly/2026-05-03) spiked near $6,382 on big Property Group LLC hits and possible duplicate recurring charges.
- **Uncategorized month:** [Uncategorized](/cashflow/-1/monthly/2026-05) is slightly down vs March/April while large Property Group LLC repeats still dominate.

# Details

## Highlights

- **Salary rhythm:** Early-month [salary](/cashflow/36/monthly/2026-05) through May 10 matches the same window in April at the same dollar level; the “drop vs March” in the Explain line reflects March’s one-time Genentech bonus and higher CA State Payroll in early March, not a structural pay cut.
- **Food spend:** [Meals](/cashflow/1/monthly/2026-05) totals for early May are lower partly because only 10 days are in the window compared with full April and March months; the rationalize text also flags Applebee’s dining charges categorized as groceries, which can skew grocery totals.

## Lowlights

- **Uncategorized spike:** Last week [Uncategorized](/cashflow/-1/weekly/2026-05-03) jumped to about $6,382, driven heavily by two Property Group LLC payments ($2,850 each) and possible duplicate recurring charges (Community Pool, Costco, BP).
- **Uncategorized trend:** Month-to-date [Uncategorized](/cashflow/-1/monthly/2026-05) is slightly down vs March/April; large recurring Property Group LLC charges still dominate, with fewer small miscellaneous transactions so far in May.
""",
    },
    {
        "name": "from_verbose_style_two_themes",
        "input": """
# Top Takeaways

## Highlights

- **Income stability:** May 1–10 [salary](/cashflow/36/monthly/2026-05) aligns with April’s same window; March’s higher early-month income included bonus and payroll lifts.

## Lowlights

- **Groceries vs dining:** [Food](/cashflow/1/monthly/2026-05) looks “down” partly from comparing 10 days of May to full months; Applebee’s may be miscategorized under groceries.
""",
        "output": """
# TLDR

## Highlights

- **Income:** May 1–10 [salary](/cashflow/36/monthly/2026-05) tracks April; March was higher on bonus and payroll timing.

## Lowlights

- **Food mix:** [Food](/cashflow/1/monthly/2026-05) looks “down” partly from a 10-day May window vs full months; Applebee’s may sit under groceries.

# Details

## Highlights

- **Income stability:** May 1–10 [salary](/cashflow/36/monthly/2026-05) aligns with April’s same window; March’s higher early-month income included bonus and payroll lifts called out in the source.

## Lowlights

- **Groceries vs dining:** [Food](/cashflow/1/monthly/2026-05) looks “down” partly from comparing 10 days of May to full months; Applebee’s may be miscategorized under groceries until you recategorize those dining charges.
""",
    },
    {
        "name": "uncat_txn_and_large_txn_transaction_taps",
        "input": """
# Top Takeaways

## Highlights

- **Uncategorized outflow:** The insight flags [Property Group LLC](/cashflow/transaction/123) as an uncategorized **$2,850** outflow with **Clothing** as the likely category.

## Lowlights

- **Uncategorized mix:** Uncategorized totals still include [Property Group LLC](/cashflow/transaction/123) at **$2,850** while that charge posts without a category.
- **Large dining outflow:** [Burger King](/cashflow/transaction/456) shows a **$40** single-ticket outflow on 05/09.
""",
        "output": """
# TLDR

## Highlights

- **Uncategorized:** [Property Group LLC](/cashflow/transaction/123) posts uncategorized at **$2,850** with Clothing as the likely category.

## Lowlights

- **Bucket mix:** Uncategorized totals still carry [Property Group LLC](/cashflow/transaction/123) while it posts without a category.
- **Dining ticket:** [Burger King](/cashflow/transaction/456) recorded a **$40** outflow on 05/09.

# Details

## Highlights

- **Uncategorized outflow:** The insight flags [Property Group LLC](/cashflow/transaction/123) as an uncategorized **$2,850** outflow with **Clothing** as the likely category.

## Lowlights

- **Uncategorized mix:** Uncategorized totals still include [Property Group LLC](/cashflow/transaction/123) at **$2,850** while that charge posts without a category.
- **Large dining outflow:** [Burger King](/cashflow/transaction/456) shows a **$40** single-ticket outflow on 05/09.
""",
    },
    {
        "name": "four_vs_forecast_plus_uncat_and_large_txn_transaction_taps",
        "input": """
# Top Takeaways

## Highlights

- **Uncategorized outflow:** The insight flags [Property Group LLC](/cashflow/transaction/123) as an uncategorized **$2,850** outflow with **Clothing** as the likely category.
- **Leisure vs plan:** [Leisure](/cashflow/7/monthly/2026-05) is up vs forecast on weekend entertainment and a concert ticket early in May.
- **Shopping lift:** [Shopping](/cashflow/11/monthly/2026-05) is up vs forecast after two online orders for accessories and home goods.

## Lowlights

- **Uncategorized mix:** Uncategorized totals still include [Property Group LLC](/cashflow/transaction/123) at **$2,850** while that charge posts without a category.
- **Large dining outflow:** [Burger King](/cashflow/transaction/456) shows a **$40** single-ticket outflow on 05/09.
- **Commutes and bills:** [Transport](/cashflow/8/monthly/2026-05) is down vs forecast with fewer commutes and no fuel fill-ups yet in early May, while [Bills](/cashflow/9/monthly/2026-05) is slightly up vs forecast from an annual software renewal posting in the first week of May.
""",
        "output": """
# TLDR

## Highlights

- **Uncategorized:** [Property Group LLC](/cashflow/transaction/123) posts uncategorized at **$2,850** with Clothing as the likely category.
- **Leisure:** [Leisure](/cashflow/7/monthly/2026-05) is up vs forecast on weekend entertainment and a concert ticket.
- **Shopping:** [Shopping](/cashflow/11/monthly/2026-05) is up vs forecast on two early-month accessory and home-goods orders.

## Lowlights

- **Bucket mix:** Uncategorized totals still carry [Property Group LLC](/cashflow/transaction/123) while it posts without a category.
- **Dining ticket:** [Burger King](/cashflow/transaction/456) recorded a **$40** outflow on 05/09.
- **Transport and bills:** [Transport](/cashflow/8/monthly/2026-05) is down vs forecast with fewer commutes and no fuel fill-ups yet; [Bills](/cashflow/9/monthly/2026-05) is slightly up on an annual software renewal in early May.

# Details

## Highlights

- **Uncategorized outflow:** The insight flags [Property Group LLC](/cashflow/transaction/123) as an uncategorized **$2,850** outflow with **Clothing** as the likely category.
- **Leisure vs plan:** [Leisure](/cashflow/7/monthly/2026-05) is up vs forecast on weekend entertainment and a concert ticket early in May.
- **Shopping lift:** [Shopping](/cashflow/11/monthly/2026-05) is up vs forecast after two online orders for accessories and home goods.

## Lowlights

- **Uncategorized mix:** Uncategorized totals still include [Property Group LLC](/cashflow/transaction/123) at **$2,850** while that charge posts without a category.
- **Large dining outflow:** [Burger King](/cashflow/transaction/456) shows a **$40** single-ticket outflow on 05/09.
- **Commutes and bills:** [Transport](/cashflow/8/monthly/2026-05) is down vs forecast with fewer commutes and no fuel fill-ups yet in early May, while [Bills](/cashflow/9/monthly/2026-05) is slightly up vs forecast from an annual software renewal posting in the first week of May.
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
        print("Usage: python3 top_takeaways_tldr_optimizer.py --test <index|name|all>")
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
    parser = argparse.ArgumentParser(description="Top Takeaways → TLDR & Details with Highlights/Lowlights each (Gemini one-shot)")
    parser.add_argument("--test", type=str, default=None, help='Index, test name, or "all"')
    args = parser.parse_args()
    main(test=args.test)
