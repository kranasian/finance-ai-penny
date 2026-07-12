"""
Optimizer runner for **P:TopTakeawaysVerbose** (Gemini one-shot prompt tuning).

Test fixtures are pre-built markdown user messages (final shape after input prep in
``finance-ai-llm-server/insights/top_takeaways_verbose.py``): ``# {Category} Insight``, ``Explain:``,
``## Drivers``, and ``## Helpful Links to Information``.

The **system instruction** is the canonical top-takeaways prompt (``SYSTEM_PROMPT``).

Run from ``finance-ai-penny`` repo root:

  python3 active_experiments/top_takeaways_verbose_optimizer.py --test 0
  python3 active_experiments/top_takeaways_verbose_optimizer.py --test all

Thinking stays **on** (``thinking_budget`` 256, aligned with the production top-takeaways LLM template).
"""

from __future__ import annotations

import argparse
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

# Canonical prompt for **P:TopTakeawaysVerbose** (rollup instructions for multi-context top takeaways).
SYSTEM_PROMPT = """You are Penny, a friendly peer that analyzes finances.

# Input
- Insight (focus of rationalization)
- Rationalization: Figures (supporting data), Drivers (explanation to movement in Figures), Helpful Links to Information

# Output
Synthesize Top Takeaways from the input. Divide into Highlights (stability, savings, flat costs) and Lowlights (overspending, miscategorization). ≤3 bullets per section.

Structure each bullet as **Short Headline that Names the Story:** 2 to 4 sentences that summarize what, amount, comparison, and driver/s.

Follow the structure below. Do not add/remove any segments.

# Top Takeaways

## Highlights

- **Income Stability:** …

## Lowlights

- **Utilities Spike:** …

# Guidelines

- **Voice:** Friendly, informative, observational, conversational
- **Discrepancies:** Drivers is the only source of truth. When Insight does not agree with Drivers, ignore Insight. Do not acknowledge the discrepancy.
- **Periods:** this month/week, last month/week, at this point last month/week, or similar; if period is older than last month/week, format as "from April 26 to May 2" or "in March". "First X days of Y" in the input means that the period is ongoing — use "this month/week" instead, and never "first X days of Y", for this.
- **Category Links:** Use the Helpful Links to Information when mentioning a category (ie. `[display name](/cashflow/path)`)
- **Amounts:** Whole dollars, comma thousands (ex: `$6,369`).
"""

# Production top-takeaways template ``thinking_budget`` — do not set to 0.
TOP_TAKEAWAYS_THINKING_BUDGET = 256
# Generous cap so multi-context rollups (plus thinking-enabled models) are not truncated mid-markdown.
TOP_TAKEAWAYS_MAX_OUTPUT_TOKENS = 8192


def _normalize_markdown_relative_links(text: str) -> str:
    """Remove stray whitespace after ``](`` before a relative path (model sometimes emits ``]( /foo``)."""
    return re.sub(r"\]\(\s+/", "](/", text)


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


class TopTakeawaysVerboseOptimizer:
    """One-shot Gemini runner: canonical top-takeaways system prompt plus user message markdown."""

    def __init__(
        self,
        model_name: str = GEMINI_FLASH_LITE,
        *,
        thinking_budget: int = TOP_TAKEAWAYS_THINKING_BUDGET,
        max_output_tokens: int = TOP_TAKEAWAYS_MAX_OUTPUT_TOKENS,
    ):
        if genai is None or types is None:  # pragma: no cover
            raise RuntimeError("Install `google-genai` (and optionally `python-dotenv`) for this optimizer.")
        if not isinstance(thinking_budget, int) or thinking_budget < 1:
            raise ValueError("thinking_budget must be a positive int (thinking must stay enabled).")
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

    def generate_markdown(self, user_message: str) -> str:
        if not isinstance(user_message, str) or not user_message.strip():
            raise ValueError("user_message must be a non-empty markdown string.")
        request_text = types.Part.from_text(text=user_message.strip())
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
        text = _normalize_markdown_relative_links("".join(out).strip())
        if not text:
            raise ValueError("Empty model response.")
        return text


TEST_CASES: list[dict[str, Any]] = [
    {
        "name": "four_contexts_all_rationalizes_with_tap_links",
        "input": """
# Salary Insight

Explain: Salary is significantly down this month at $4200. (2026-05-01 to 2026-05-10)

## Drivers

The Explain line describes salary as significantly down at $4,200 this month, but Figures show $6,369 for May 1–10—the same as the $6,369 from the same window in April—so pay is flat month-over-month, not down. The "significant drop" compared to early March is due to a large, one-time bonus payout received in the first half of March 2026. Specifically, you received $8,800 in total "Genentech US Bonus" payments on March 11, alongside higher "CA State Payroll" amounts ($8,600 total) compared to your standard bi-weekly payroll cycle. Your income for the first 10 days of May is consistent with your regular pay cycle, mirroring the amount received during the same period in April.

## Helpful Links to Information

- [Salary](/cashflow/36/monthly/2026-05)

# Food Insight

Explain: Groceries is significantly down this month at $937.  Food is thus significantly down this month to $1859. (2026-05-01 to 2026-05-10)

## Drivers

The significant decrease in food and grocery spending is primarily due to comparing only 10 days of May activity against full-month totals for April and March.

Additionally, your transaction history shows some potential miscategorization: several transactions at "Applebee's" (a dining venue) are currently categorized under `meals_groceries`. For example, $85.36 was recorded at Applebee's on May 2nd under groceries, whereas a typical grocery bill for a similar amount is expected from retailers like "Target" or "Trader Joe's".

## Helpful Links to Information

- [Food](/cashflow/1/monthly/2026-05)
- [Dining Out](/cashflow/2/monthly/2026-05)
- [Delivered Food](/cashflow/3/monthly/2026-05)
- [Groceries](/cashflow/4/monthly/2026-05)

# Uncategorized Insight

Explain: Uncategorized is significantly up last week at $6382. (2026-05-03 to 2026-05-09)

## Drivers

The spike in uncategorized spending is primarily driven by two large payments to "Property Group LLC" totaling $5,700 ($2,850 each). Additionally, duplicate entries appear for several recurring transactions, such as "Community Pool" ($83.06 x 2), "Costco" ($69.71 x 2), and "BP" ($49.56 x 2), which suggests some transactions might be showing up twice across different accounts.

## Helpful Links to Information

- [Uncategorized](/cashflow/-1/weekly/2026-05-03)
""",
        "output": """
# Top Takeaways

## Highlights

- **Income Stability:** This month [Salary](/cashflow/36/monthly/2026-05) tracks at $6,369, which is consistent with the amount received at this point last month. While income appears significantly lower than in Mar, that variance is driven by a one-time $8,800 bonus and higher payroll amounts received during the first half of Mar, rather than a change in the standard bi-weekly pay cycle.

## Lowlights

- **Uncategorized Spikes:** Last week, [Uncategorized](/cashflow/-1/weekly/2026-05-03) spending surged to $6,382, a sharp increase from the $726 recorded from Apr 26 to May 2. This spike reflects two large payments to Property Group LLC totaling $5,700, alongside duplicate entries for recurring transactions like Community Pool, Costco, and BP, which suggests some items may be appearing twice across accounts.
- **Miscategorization:** While [Food](/cashflow/1/monthly/2026-05) spending is $1,859 and [Groceries](/cashflow/4/monthly/2026-05) is $937 this month, transaction history reveals potential miscategorization. For instance, an $85 charge at Applebee's was recorded under groceries on May 2, despite being a dining venue, which contrasts with typical grocery bills from retailers like Trader Joe's.
""",
    },
    {
        "name": "single_context_delivered_food",
        "input": """
# Delivered Food Insight

Explain: Delivered Food is significantly down this month at $0. (2026-05-01 to 2026-05-23)

## Drivers

Spending on **Delivered Food** is $0 this month, consistent with at this point last month and the same $0 pattern in Mar and Feb.

## Helpful Links to Information

- [Delivered Food](/cashflow/3/monthly/2026-05)
""",
        "output": """
# Top Takeaways
## Highlights
- **Spending Control:** You have maintained a clean streak in [Delivered Food](/cashflow/3/monthly/2026-05) this month, with total spending at $0.
This consistent behavior mirrors the $0 activity recorded at this point last month, as well as the same $0 patterns observed throughout Mar and Feb.
""",
    },
    {
        "name": "six_contexts_income_and_bills_mosaic",
        "input": """
# Salary Insight

Explain: Salary is slightly up this month at $4210. (2026-05-01 to 2026-05-10)

## Drivers

Payroll landed on the usual bi-weekly schedule with no bonus lines this month.

## Helpful Links to Information

- [Salary](/cashflow/36/monthly/2026-05)

# Side-Gig Insight

Explain: Side-Gig is significantly up this month at $1850. (2026-05-01 to 2026-05-10)

## Drivers

A $1,200 consulting deposit from "Brightline Studio" posted May 6, plus two smaller Upwork payouts ($350 and $300) that did not appear at this point last month.

## Helpful Links to Information

- [Side-Gig](/cashflow/37/monthly/2026-05)

# Connectivity Insight

Explain: Connectivity is flat this month at $189. (2026-05-01 to 2026-05-10)

## Drivers

Recurring Comcast and mobile plans unchanged.

## Helpful Links to Information

- [Connectivity](/cashflow/10/monthly/2026-05)

# Insurance Insight

Explain: Insurance is slightly up this month at $312. (2026-05-01 to 2026-05-10)

## Drivers

Auto premium installment rose $14.40 after policy renewal effective May 1; homeowners line unchanged.

## Helpful Links to Information

- [Insurance](/cashflow/11/monthly/2026-05)

# Utilities Insight

Explain: Utilities is significantly up this month at $428. (2026-05-01 to 2026-05-10)

## Drivers

PG&E bill jumped to $286.40 on May 4 versus $118.20 at this point last month, reflecting higher cooling use and a rate adjustment. Water district charge ($92.75) also posted earlier in the cycle than in Apr, when it landed on the 18th. Together those two lines explain most of the month-over-month lift; garbage service stayed flat at $49.00.

## Helpful Links to Information

- [Utilities](/cashflow/16/monthly/2026-05)

# Interest Insight

Explain: Interest is slightly up this month at $18. (2026-05-01 to 2026-05-10)

## Drivers

Savings yield credit posted May 2.

## Helpful Links to Information

- [Interest](/cashflow/39/monthly/2026-05)
""",
        "output": """
# Top Takeaways

## Highlights

- **Income Growth:** Total [Salary](/cashflow/36/monthly/2026-05) is slightly up this month at $4,210 compared to $4,106 at this point last month, driven by standard bi-weekly payroll.
Additionally, [Interest](/cashflow/39/monthly/2026-05) income rose to $18 from $12 at this point last month, following a savings yield credit posted on May 2.
- **Side-Gig Expansion:** [Side-Gig](/cashflow/37/monthly/2026-05) earnings saw a significant increase this month, reaching $1,850 compared to $420 at this point last month. This lift is driven by a $1,200 consulting deposit from Brightline Studio on May 6, alongside two Upwork payouts totaling $650 that did not occur in the prior period.
- **Stable Expenses:** [Connectivity](/cashflow/10/monthly/2026-05) costs remain flat this month at $189, consistent with the total from last month, as recurring Comcast and mobile plan charges were unchanged.

## Lowlights

- **Utilities Spike:** [Utilities](/cashflow/16/monthly/2026-05) expenses are significantly higher this month at $428, compared to $242 last month. The increase is driven by a PG&E bill of $286—up from $118 at this point last month due to higher cooling use and a rate adjustment—and a water district charge of $93 that posted earlier in the cycle than it did in April.
- **Insurance Adjustment:** [Insurance](/cashflow/11/monthly/2026-05) costs are slightly up this month at $312, compared to $298 last month. This change reflects a $14 increase in the auto premium installment following a policy renewal effective May 1, while the homeowners line remained unchanged.
""",
    },
    {
        "name": "two_contexts_shelter_and_entertainment_deep_drivers",
        "input": """
# Shelter Insight

Explain: Utilities is significantly up this month at $428.  Shelter is thus significantly up this month to $2840. (2026-05-01 to 2026-05-10)

## Drivers

Shelter costs rose this month almost entirely because **Utilities** accelerated, not because rent moved. The **Home** (mortgage) payment is identical to last month at $2,412.40 and posted on May 1 as usual. **Utilities** jumped mainly on PG&E: $286.40 on May 4 compared with $118.20 at this point last month, which Drivers tie to earlier cooling use and a tariff step-up that took effect May 1. The water district bill ($92.75) also landed May 3 instead of mid-month as in Apr, front-loading the category. **Upkeep** was $0 this month versus $42.50 in Apr when a sprinkler repair posted — that absence partially offsets the utility spike but does not fully neutralize it. Net shelter is up $187.35 versus the full prior month even though fixed housing charges are flat.

## Helpful Links to Information

- [Shelter](/cashflow/14/monthly/2026-05)
- [Home](/cashflow/15/monthly/2026-05)
- [Utilities](/cashflow/16/monthly/2026-05)
- [Upkeep](/cashflow/17/monthly/2026-05)

# Entertainment Insight

Explain: Entertainment is significantly up last week at $186. (2026-05-03 to 2026-05-09)

## Drivers

Last week’s entertainment spike is concentrated in two tickets: **AMC Theaters** $68.50 on May 4 and **StubHub — Indie Fest** $94.00 on May 6. A $23.90 **Spotify** renewal also posted May 3; that recurring charge appears most weeks but the concert and movie lines drove the step-change versus the prior week’s $24.00 total, which was essentially streaming only.

## Helpful Links to Information

- [Entertainment](/cashflow/6/weekly/2026-05-03)
""",
        "output": """
# Top Takeaways

## Highlights

- **Fixed Housing:** This month, [Home](/cashflow/15/monthly/2026-05) mortgage payments were managed to remain flat at $2,412.

## Lowlights

- **Shelter Surge:** Shelter is up this month to $2,841, primarily because of a surge in [Utilities](/cashflow/16/monthly/2026-05) to $428. This is driven by a PG&E charge of $286—significantly higher than the $118 seen at this point last month—alongside an early water district bill of $93. Although the absence of [Upkeep](/cashflow/17/monthly/2026-05) costs, which were $43 in Apr, provides a minor offset, the overall [Shelter](/cashflow/14/monthly/2026-05) category remains elevated compared to the $2,653 total from last month.
- **Entertainment Spike:** Spending on entertainment rose to $186 last week, a sharp increase from the $24 recorded in the prior week. This jump in [Entertainment](/cashflow/6/weekly/2026-05-03) is concentrated in two specific transactions: $69 for AMC Theaters and $94 for StubHub, which combined with the recurring $24 Spotify renewal to drive the category well above the previous week's streaming-only baseline.
""",
    },
    {
        "name": "five_contexts_health_education_shopping_mixed_grains",
        "input": """
# Health Insight

Explain: Medical & Pharmacy is significantly up this month at $342.  Health is thus significantly up this month to $518. (2026-05-01 to 2026-05-10)

## Drivers

Medical & Pharmacy dominates the health lift: **CVS Pharmacy** $156.80 and **Kaiser Copay** $142.00 posted May 2–5 versus $88.40 for all of last month. Gym & Wellness rose mainly because annual **Peloton** renewal $176.45 hit May 1; last month gym lines were $123.90 with no renewal.

## Helpful Links to Information

- [Health](/cashflow/28/monthly/2026-05)
- [Medical & Pharmacy](/cashflow/29/monthly/2026-05)
- [Gym & Wellness](/cashflow/30/monthly/2026-05)
- [Personal Care](/cashflow/31/monthly/2026-05)

# Tuition Insight

Explain: Tuition is slightly up last week at $650. (2026-05-03 to 2026-05-09)

## Drivers

Spring semester installment to **State University** posted May 5.

## Helpful Links to Information

- [Tuition](/cashflow/20/weekly/2026-05-03)

# Clothing Insight

Explain: Clothing is significantly down this month at $45. (2026-05-01 to 2026-05-10)

## Drivers

No department-store runs this month; the only charge is **Target** socks $45.00 on May 7. Last month included **Nordstrom** $198.00 and **Old Navy** $114.80.

## Helpful Links to Information

- [Clothing](/cashflow/22/monthly/2026-05)

# Gadgets Insight

Explain: Gadgets is significantly up this month at $279. (2026-05-01 to 2026-05-10)

## Drivers

**Apple Store** Magic Keyboard $279.99 on May 8.

## Helpful Links to Information

- [Gadgets](/cashflow/23/monthly/2026-05)

# Donations & Gifts Insight

Explain: Donations & Gifts is flat this month at $50. (2026-05-01 to 2026-05-10)

## Drivers

Recurring **Red Cross** $50.00 on May 1.

## Helpful Links to Information

- [Donations & Gifts](/cashflow/32/monthly/2026-05)
""",
        "output": """
# Top Takeaways

## Highlights

- **Clothing Savings:** Spending on clothing is significantly down this month at $45 compared to $313 last month, as the absence of department-store runs has left [Clothing](/cashflow/22/monthly/2026-05) with only a single $45 charge at Target on May 7, whereas last month included larger purchases at Nordstrom and Old Navy.
- **Consistent Giving:** Giving remains consistent this month at $50, matching the $50 total from last month, driven entirely by the recurring [Donations & Gifts](/cashflow/32/monthly/2026-05) payment to Red Cross on May 1.

## Lowlights

- **Health Surge:** Costs are significantly up this month at $519 compared to $212 last month, primarily due to a surge in [Medical & Pharmacy](/cashflow/29/monthly/2026-05) expenses—which reached $342 this month versus $88 last month—following CVS Pharmacy and Kaiser Copay charges posted May 2–5. Additionally, [Gym & Wellness](/cashflow/30/monthly/2026-05) spending rose to $176 this month, driven by an annual Peloton renewal on May 1, compared to $124 in gym lines last month.
- **Tuition Up:** Educational expenses are slightly up this week at $650, reflecting a spring semester installment to State University that posted May 5, whereas [Tuition](/cashflow/20/weekly/2026-05-03) saw no activity in the prior week.
- **Gadgets Splurge:** Spending is significantly up this month at $280, a sharp increase from $0 last month, driven by a single [Gadgets](/cashflow/23/monthly/2026-05) purchase of an Apple Store Magic Keyboard on May 8.
""",
    },
]

_RE_CENTS = re.compile(r"\$\d{1,3}(?:,\d{3})*\.\d{2}")
_RE_IDEAL_LINK = re.compile(r"\]\((/cashflow/[^)]+)\)")
_RE_MD_LINK = re.compile(r"\[[^\]]*\]\([^)]+\)")


def _prose_without_links(text: str) -> str:
    return _RE_MD_LINK.sub("", text or "")


def mechanical_sandbox_check(*, ideal: str, actual: str) -> dict[str, Any]:
    """Heuristic checks vs ideal output: structure, links, amounts, period wording."""
    issues: list[str] = []
    ideal_s = (ideal or "").strip()
    actual_s = (actual or "").strip()
    if "# Top Takeaways" not in actual_s:
        issues.append("Missing header: # Top Takeaways")
    if "## Highlights" not in actual_s:
        issues.append("Missing header: ## Highlights")
    if "## Lowlights" in ideal_s and "## Lowlights" not in actual_s:
        issues.append("Missing header: ## Lowlights")
    ideal_links = _RE_IDEAL_LINK.findall(ideal_s)
    for path in ideal_links:
        if path not in actual_s:
            issues.append(f"Missing ideal link path: {path}")
    prose = _prose_without_links(actual_s)
    for m in _RE_CENTS.findall(prose):
        issues.append(f"Unrounded amount (cents): {m}")
    banned = (
        "first 10 days",
        "first X days",
        " so far",
        "the previous week",
        "None.",
    )
    for phrase in banned:
        if phrase in prose:
            issues.append(f"Banned phrasing in prose: {phrase!r}")
    takeaway_body = actual_s.split("# Top Takeaways", 1)[-1] if "# Top Takeaways" in actual_s else actual_s
    for stop in ("## Helpful Links", "\n***", "\n# "):
        if stop in takeaway_body:
            takeaway_body = takeaway_body.split(stop, 1)[0]
    hi = takeaway_body.split("## Lowlights")[0].split("## Highlights")[-1] if "## Highlights" in takeaway_body else ""
    lo = takeaway_body.split("## Lowlights")[-1] if "## Lowlights" in takeaway_body else ""
    hi_count = hi.count("\n- ")
    lo_count = lo.count("\n- ")
    if hi_count > 3:
        issues.append(f"Too many Highlights bullets: {hi_count}")
    if lo_count > 3:
        issues.append(f"Too many Lowlights bullets: {lo_count}")
    good = not issues
    return {"good_copy": good, "issues": issues, "eval_text": "\n".join(f"- {x}" for x in issues) if issues else "OK"}


def _run_test(user_message: str, optimizer: TopTakeawaysVerboseOptimizer | None = None) -> str:
    if optimizer is None:
        optimizer = TopTakeawaysVerboseOptimizer()
    print("=" * 80)
    print("LLM USER MESSAGE (top takeaways task):")
    print("=" * 80)
    print(user_message)
    print("=" * 80 + "\n")
    result = optimizer.generate_markdown(user_message)
    print("=" * 80)
    print("LLM OUTPUT:")
    print("=" * 80)
    print(result)
    print("=" * 80 + "\n")
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


def run_test(
    name_or_index: str | int,
    optimizer: TopTakeawaysVerboseOptimizer | None = None,
    *,
    run_sandbox: bool = True,
) -> str | None:
    tc = get_test_case(name_or_index) if not isinstance(name_or_index, dict) else name_or_index
    if tc is None:
        print(f"Test case {name_or_index!r} not found.")
        return None
    print(f"\n{'=' * 80}\nRunning test: {tc['name']}\n{'=' * 80}\n")
    result = _run_test(tc["input"], optimizer)
    if run_sandbox and result and tc.get("output"):
        print("=" * 80)
        print("IDEAL OUTPUT:")
        print("=" * 80)
        print(tc["output"].strip())
        print("=" * 80)
        print("SANDBOX EXECUTION:")
        print("=" * 80)
        sand = mechanical_sandbox_check(ideal=tc["output"], actual=result)
        print(sand["eval_text"])
        print(f"{'✅' if sand['good_copy'] else '❌'} good_copy={sand['good_copy']}")
        print("=" * 80 + "\n")
    return result


def main(*, test: str | None) -> None:
    if test is None:
        print("Usage: python3 active_experiments/top_takeaways_verbose_optimizer.py --test <index|name|all>")
        print("Tests:")
        for i, tc in enumerate(TEST_CASES):
            print(f"  {i}: {tc['name']}")
        return
    optimizer = TopTakeawaysVerboseOptimizer(thinking_budget=TOP_TAKEAWAYS_THINKING_BUDGET)
    if test.strip().lower() == "all":
        for i, tc in enumerate(TEST_CASES):
            run_test(i, optimizer)
            if i < len(TEST_CASES) - 1:
                print("\n" + "-" * 80 + "\n")
        return
    key: str | int = int(test) if test.isdigit() else test
    run_test(key, optimizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Top takeaways verbose optimizer (Gemini one-shot)")
    parser.add_argument("--test", type=str, default=None, help='Index, test name, or "all"')
    args = parser.parse_args()
    main(test=args.test)
