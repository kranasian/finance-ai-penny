"""
JSON variant of ``top_takeaways_tldr_optimizer.py``: same **Top Takeaways** markdown input,
but the model returns a **single JSON object** (no markdown headings) with ``tldr`` and ``details``,
each containing ``highlights`` and ``lowlights`` as arrays of bullet strings (inline ``[label](/path)``
markdown allowed inside each string).

Run from ``finance-ai-penny`` repo root:

  python3 top_takeaways_tldr_json_optimizer.py --test 0
  python3 top_takeaways_tldr_json_optimizer.py --test all

Requires ``GEMINI_API_KEY`` and ``google-genai``. Thinking stays on (budget 256).
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
TLDR_JSON_THINKING_BUDGET = 256


def format_tldr_user_message(top_takeaways_markdown: str) -> str:
    """Same as markdown TLDR optimizer: pass through stripped Top Takeaways markdown."""
    body = (top_takeaways_markdown or "").strip()
    if not body:
        raise ValueError("top_takeaways_markdown must be non-empty.")
    return body + "\n"


SYSTEM_PROMPT = """You are **Penny**.

The user message contains **Top Takeaways** markdown: ``# Top Takeaways``, ``## Highlights``, and ``## Lowlights`` with ``- `` bullets. This is produced by an upstream rollup step.

Your task is the same semantic job as the markdown TLDR step: produce **TLDR** (ultra-compact) and **Details** (fuller), each with **highlights** and **lowlights** — but output **only** a single **JSON object**, not markdown headings.

**JSON shape (required keys and nesting — nothing else at the root):** an object with keys ``tldr`` and ``details``. Each of those is an object with keys ``highlights`` and ``lowlights``, each an **array of strings** (each string is one bullet). Example key tree: ``tldr`` → ``highlights`` (array), ``lowlights`` (array); ``details`` → ``highlights`` (array), ``lowlights`` (array).

- Each array value is **one bullet** as a plain string. Do **not** prefix strings with ``- `` or ``* ``; the array position is the bullet.
- When the source uses Markdown drill-down links ``[text](/path)``, preserve them **inside** the same string (verbatim ``href``, no space after ``(`` in ``](``). This applies **equally** to ``tldr`` and ``details`` strings: **do not** drop links from TLDR-only compression—if the source bullet for that theme had a link, the matching TLDR string must still contain at least one ``[...](...)`` with the same ``href``.
- **Bullet budget:** At most **3** strings in ``tldr.highlights``, **3** in ``tldr.lowlights``, **3** in ``details.highlights``, **3** in ``details.lowlights``. Use fewer if the source is thin; never exceed 3 per array. Always include all four arrays (use ``[]`` only if there is truly nothing for that bucket — prefer at least one string when the source has content).
- **TLDR strings:** At most one short sentence each where possible; lead with headline outcome; ~25 words or fewer when feasible; no semicolon laundry lists. **Keep drill-down links** as required above; brevity is not an excuse to drop required ``href`` values.
- **Details strings:** Clearly longer and denser than the matching TLDR strings for the same theme; multiple sentences allowed.
- Ground everything in the provided Top Takeaways only. Do **not** invent transactions, amounts, or dates. Copy numbers and dates exactly from the input.
- For **uncategorized- or large-transaction** themes (often ``/cashflow/transaction/…``), keep the same **observational** tone as vs-forecast rollup bullets in the source: **do not** add action items or imperatives (“confirm”, “verify”, “should”, “until you act”) unless the input states them verbatim.

**Output discipline:**
- Your **entire** last assistant message must be **only** the JSON object: **no** markdown code fences, **no** preamble, **no** trailing commentary.
- Use standard JSON: double quotes for keys and string values; escape internal double quotes as ``\\"``; no trailing commas.
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


_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*", re.IGNORECASE)
_JSON_FENCE_TAIL_RE = re.compile(r"\s*```\s*$", re.DOTALL)


def extract_json_object(text: str) -> str:
    """Strip optional ```json fences and surrounding whitespace."""
    s = (text or "").strip()
    if not s:
        return ""
    s = _JSON_FENCE_RE.sub("", s, count=1)
    s = _JSON_FENCE_TAIL_RE.sub("", s, count=1)
    return s.strip()


def parse_tldr_json(text: str) -> dict[str, Any]:
    """Parse model output into a dict; raises ``json.JSONDecodeError`` if invalid."""
    raw = extract_json_object(text)
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("Top Takeaways TLDR JSON root must be an object.")
    return obj


class TopTakeawaysTldrJsonOptimizer:
    """Gemini one-shot: Top Takeaways markdown → TLDR/Details JSON object."""

    def __init__(
        self,
        model_name: str = GEMINI_FLASH_LITE,
        *,
        thinking_budget: int = TLDR_JSON_THINKING_BUDGET,
        max_output_tokens: int = 4096,
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

    def generate_json_text(self, top_takeaways_markdown: str, *, indent: int | None = 2) -> str:
        """Call the model and return **pretty-printed** JSON text (validated)."""
        obj = self.generate_json(top_takeaways_markdown)
        return json.dumps(obj, ensure_ascii=False, indent=indent) + "\n"

    def generate_json(self, top_takeaways_markdown: str) -> dict[str, Any]:
        """Call the model and return the parsed JSON object."""
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
        return parse_tldr_json(text)


def _golden_json(
    *,
    tldr_h: list[str],
    tldr_l: list[str],
    det_h: list[str],
    det_l: list[str],
) -> str:
    return json.dumps(
        {"tldr": {"highlights": tldr_h, "lowlights": tldr_l}, "details": {"highlights": det_h, "lowlights": det_l}},
        ensure_ascii=False,
        indent=2,
    )


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
        "output": _golden_json(
            tldr_h=[
                "**Salary:** Early [salary](/cashflow/36/monthly/2026-05) matches April; March was higher on bonus timing, not a pay cut.",
                "**Food:** [Meals](/cashflow/1/monthly/2026-05) looks lower partly because May is only 10 days vs full prior months.",
            ],
            tldr_l=[
                "**Uncategorized week:** [Uncategorized](/cashflow/-1/weekly/2026-05-03) spiked near $6,382 on big Property Group LLC hits and possible duplicate recurring charges.",
                "**Uncategorized month:** [Uncategorized](/cashflow/-1/monthly/2026-05) is slightly down vs March/April while large Property Group LLC repeats still dominate.",
            ],
            det_h=[
                "**Salary rhythm:** Early-month [salary](/cashflow/36/monthly/2026-05) through May 10 matches the same window in April at the same dollar level; the “drop vs March” in the Explain line reflects March’s one-time Genentech bonus and higher CA State Payroll in early March, not a structural pay cut.",
                "**Food spend:** [Meals](/cashflow/1/monthly/2026-05) totals for early May are lower partly because only 10 days are in the window compared with full April and March months; the rationalize text also flags Applebee’s dining charges categorized as groceries, which can skew grocery totals.",
            ],
            det_l=[
                "**Uncategorized spike:** Last week [Uncategorized](/cashflow/-1/weekly/2026-05-03) jumped to about $6,382, driven heavily by two Property Group LLC payments ($2,850 each) and possible duplicate recurring charges (Community Pool, Costco, BP).",
                "**Uncategorized trend:** Month-to-date [Uncategorized](/cashflow/-1/monthly/2026-05) is slightly down vs March/April; large recurring Property Group LLC charges still dominate, with fewer small miscellaneous transactions so far in May.",
            ],
        ),
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
        "output": _golden_json(
            tldr_h=[
                "**Income:** May 1–10 [salary](/cashflow/36/monthly/2026-05) tracks April; March was higher on bonus and payroll timing.",
            ],
            tldr_l=[
                "**Food mix:** [Food](/cashflow/1/monthly/2026-05) looks “down” partly from a 10-day May window vs full months; Applebee’s may sit under groceries.",
            ],
            det_h=[
                "**Income stability:** May 1–10 [salary](/cashflow/36/monthly/2026-05) aligns with April’s same window; March’s higher early-month income included bonus and payroll lifts called out in the source.",
            ],
            det_l=[
                "**Groceries vs dining:** [Food](/cashflow/1/monthly/2026-05) looks “down” partly from comparing 10 days of May to full months; Applebee’s may be miscategorized under groceries until you recategorize those dining charges.",
            ],
        ),
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
        "output": _golden_json(
            tldr_h=[
                "**Uncategorized:** [Property Group LLC](/cashflow/transaction/123) posts uncategorized at **$2,850** with Clothing as the likely category.",
            ],
            tldr_l=[
                "**Bucket mix:** Uncategorized totals still carry [Property Group LLC](/cashflow/transaction/123) while it posts without a category.",
                "**Dining ticket:** [Burger King](/cashflow/transaction/456) recorded a **$40** outflow on 05/09.",
            ],
            det_h=[
                "**Uncategorized outflow:** The insight flags [Property Group LLC](/cashflow/transaction/123) as an uncategorized **$2,850** outflow with **Clothing** as the likely category.",
            ],
            det_l=[
                "**Uncategorized mix:** Uncategorized totals still include [Property Group LLC](/cashflow/transaction/123) at **$2,850** while that charge posts without a category.",
                "**Large dining outflow:** [Burger King](/cashflow/transaction/456) shows a **$40** single-ticket outflow on 05/09.",
            ],
        ),
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
        "output": _golden_json(
            tldr_h=[
                "**Uncategorized:** [Property Group LLC](/cashflow/transaction/123) posts uncategorized at **$2,850** with Clothing as the likely category.",
                "**Leisure:** [Leisure](/cashflow/7/monthly/2026-05) is up vs forecast on weekend entertainment and a concert ticket.",
                "**Shopping:** [Shopping](/cashflow/11/monthly/2026-05) is up vs forecast on two early-month accessory and home-goods orders.",
            ],
            tldr_l=[
                "**Bucket mix:** Uncategorized totals still carry [Property Group LLC](/cashflow/transaction/123) while it posts without a category.",
                "**Dining ticket:** [Burger King](/cashflow/transaction/456) recorded a **$40** outflow on 05/09.",
                "**Transport and bills:** [Transport](/cashflow/8/monthly/2026-05) is down vs forecast with fewer commutes and no fuel fill-ups yet; [Bills](/cashflow/9/monthly/2026-05) is slightly up on an annual software renewal in early May.",
            ],
            det_h=[
                "**Uncategorized outflow:** The insight flags [Property Group LLC](/cashflow/transaction/123) as an uncategorized **$2,850** outflow with **Clothing** as the likely category.",
                "**Leisure vs plan:** [Leisure](/cashflow/7/monthly/2026-05) is up vs forecast on weekend entertainment and a concert ticket early in May.",
                "**Shopping lift:** [Shopping](/cashflow/11/monthly/2026-05) is up vs forecast after two online orders for accessories and home goods.",
            ],
            det_l=[
                "**Uncategorized mix:** Uncategorized totals still include [Property Group LLC](/cashflow/transaction/123) at **$2,850** while that charge posts without a category.",
                "**Large dining outflow:** [Burger King](/cashflow/transaction/456) shows a **$40** single-ticket outflow on 05/09.",
                "**Commutes and bills:** [Transport](/cashflow/8/monthly/2026-05) is down vs forecast with fewer commutes and no fuel fill-ups yet in early May, while [Bills](/cashflow/9/monthly/2026-05) is slightly up vs forecast from an annual software renewal posting in the first week of May.",
            ],
        ),
    },
]


def _run_test(top_takeaways_md: str, optimizer: TopTakeawaysTldrJsonOptimizer | None = None) -> dict[str, Any]:
    if optimizer is None:
        optimizer = TopTakeawaysTldrJsonOptimizer()
    wrapped = format_tldr_user_message(top_takeaways_md)
    print("TLDR JSON LLM INPUT")
    print("-" * 80)
    print(wrapped)
    result = optimizer.generate_json(top_takeaways_md)
    print("TLDR JSON LLM OUTPUT")
    print("-" * 80)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print()
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


def run_test(name_or_index: str | int, optimizer: TopTakeawaysTldrJsonOptimizer | None = None) -> dict[str, Any] | None:
    tc = get_test_case(name_or_index) if not isinstance(name_or_index, dict) else name_or_index
    if tc is None:
        print(f"Test case {name_or_index!r} not found.")
        return None
    print(f"\n{'=' * 80}\nRunning test: {tc['name']}\n{'=' * 80}\n")
    return _run_test(tc["input"], optimizer)


def main(*, test: str | None) -> None:
    if test is None:
        print("Usage: python3 top_takeaways_tldr_json_optimizer.py --test <index|name|all>")
        print("Tests:")
        for i, tc in enumerate(TEST_CASES):
            print(f"  {i}: {tc['name']}")
        return
    optimizer = TopTakeawaysTldrJsonOptimizer(thinking_budget=TLDR_JSON_THINKING_BUDGET)
    if test.strip().lower() == "all":
        for i, tc in enumerate(TEST_CASES):
            run_test(i, optimizer)
            if i < len(TEST_CASES) - 1:
                print()
        return
    key: str | int = int(test) if test.isdigit() else test
    run_test(key, optimizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Top Takeaways → TLDR & Details as JSON (Gemini one-shot)")
    parser.add_argument("--test", type=str, default=None, help='Index, test name, or "all"')
    args = parser.parse_args()
    main(test=args.test)
