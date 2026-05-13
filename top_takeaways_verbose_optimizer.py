"""
Optimizer runner for multi-context **top takeaways** inputs (``<CONTEXTS>`` user message).

The **user message** is built by ``build_top_takeaways_user_message_from_contexts`` (or the shorthand
``build_top_takeaways_user_message`` for all-rationalize runs). Each **context** has a required ``tap_link`` and
**exactly one** of: ``rationalize_agent_outcome`` (full rationalize markdown) or ``insight`` (plain text for
non-rationalized transaction alerts such as ``uncat_txn`` / ``large_txn``), optionally with ``insight_kind`` for the
XML ``type`` on ``<TRANSACTION_INSIGHT>``. Each ``<TAP_LINK>`` body is one line ``<Category name> - /cashflow/...`` when a name is known (or path-only for non-category taps).

The **system instruction** is the canonical top-takeaways prompt (``_TOP_TAKEAWAYS_PROMPT``).
This one-shot Gemini path has **no** finance tools; the model should rely on the pasted rationalize markdown only.

Run from ``finance-ai-penny`` repo root:

  python3 top_takeaways_verbose_optimizer.py --test 0
  python3 top_takeaways_verbose_optimizer.py --test all

Thinking stays **on** (``thinking_budget`` 256, aligned with the production top-takeaways LLM template).
"""

from __future__ import annotations

import argparse
import os
import re
import warnings
from typing import Any

from categories import get_name

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
_RE_FIRST_CASHFLOW_PATH = re.compile(r"/cashflow/[^\s)]+")


def _category_id_from_cashflow_path(path: str) -> int | None:
    """Leading integer category id in ``/cashflow/<id>/…``, else ``None``."""
    p = (path or "").strip()
    if not p.startswith("/cashflow/"):
        return None
    rest = p[len("/cashflow/") :].lstrip("/")
    if not rest or rest.lower().startswith("transaction/"):
        return None
    m = re.match(r"^-?\d+", rest)
    if not m:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


def _first_cashflow_path_in_tap_link_inner(inner: str) -> str | None:
    if not isinstance(inner, str) or not inner.strip():
        return None
    for line in inner.splitlines():
        m = _RE_FIRST_CASHFLOW_PATH.search(line.strip())
        if m:
            return m.group(0).strip()
    m = _RE_FIRST_CASHFLOW_PATH.search(inner)
    return m.group(0).strip() if m else None


def _tap_link_name_for_category_id(cat_id: int) -> str | None:
    """Display name for ``<TAP_LINK>`` when derived from a Penny category id (id ``1`` is shown as **Food**)."""
    if cat_id == 1:
        return "Food"
    nm = get_name(cat_id)
    if isinstance(nm, str) and nm.strip():
        return nm.strip()
    return None


def _tap_link_category_display_name(*, tap_path: str, ctx: dict[str, Any]) -> str:
    raw = ctx.get("tap_link_category_name")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    meta = ctx.get("metadata")
    if isinstance(meta, dict):
        for k in ("tap_link_category_name", "category_display_name", "category_name"):
            v = meta.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        cid_raw = meta.get("category_id")
        if cid_raw is not None:
            try:
                cid = int(cid_raw)
            except (TypeError, ValueError):
                cid = None
            if cid is not None:
                nm = _tap_link_name_for_category_id(cid)
                if nm:
                    return nm
    cid = _category_id_from_cashflow_path(tap_path)
    if cid is not None:
        nm = _tap_link_name_for_category_id(cid)
        if nm:
            return nm
    return ""


def _tap_link_xml_inner_line(*, tap_path: str, ctx: dict[str, Any]) -> str:
    p = tap_path.strip()
    label = _tap_link_category_display_name(tap_path=p, ctx=ctx)
    if label:
        return f"{label} - {p}"
    return p


# Production top-takeaways template ``thinking_budget`` — do not set to 0.
TOP_TAKEAWAYS_THINKING_BUDGET = 256
# Generous cap so multi-context rollups (plus thinking-enabled models) are not truncated mid-markdown.
TOP_TAKEAWAYS_MAX_OUTPUT_TOKENS = 8192


def _sanitize_insight_kind(kind: str) -> str:
    """Safe single-token value for ``<TRANSACTION_INSIGHT type="...">`` (alphanumeric, underscore, hyphen)."""
    s = re.sub(r"[^a-zA-Z0-9_\-]", "", (kind or "").strip())
    return (s[:64] or "transaction_insight")


def build_top_takeaways_user_message_from_contexts(*, contexts: list[dict[str, Any]]) -> str:
    """Build the ``<CONTEXTS>`` user task from a list of per-context dicts.

    Each dict **must** include:

    - ``tap_link`` (str): relative drill-down URL (``/cashflow/...``).

    Optional:

    - ``tap_link_category_name`` (str): display name printed before the URL inside ``<TAP_LINK>`` (defaults from ``metadata`` or from the path’s category id).
    - ``metadata`` (dict): may include ``category_id``, ``category_name``, or ``tap_link_category_name`` for the tap line label.

    And **exactly one** of:

    - ``rationalize_agent_outcome`` (str): full rationalize markdown (``# Rationalize What`` …), wrapped in ``<RATIONALIZE>``.
    - ``insight`` (str): **non-rationalized** insight text only (e.g. ``uncat_txn`` / ``large_txn`` copy), wrapped in ``<TRANSACTION_INSIGHT>``.
    - ``insight_kind`` (str, optional): attribute for ``<TRANSACTION_INSIGHT type="...">`` (e.g. ``uncat_txn``, ``large_txn``). Sanitized if needed.
    """
    if not isinstance(contexts, list) or not contexts:
        raise ValueError("contexts must be a non-empty list")
    inner_parts: list[str] = []
    for i, ctx in enumerate(contexts, start=1):
        if not isinstance(ctx, dict):
            raise TypeError(f"contexts[{i - 1}] must be a dict")
        tl_raw = ctx.get("tap_link")
        if not isinstance(tl_raw, str) or not tl_raw.strip():
            raise ValueError(f"contexts[{i - 1}].tap_link must be a non-empty string")
        tl = tl_raw.strip()
        ra = ctx.get("rationalize_agent_outcome")
        ins = ctx.get("insight")
        has_ra = isinstance(ra, str) and bool(ra.strip())
        has_ins = isinstance(ins, str) and bool(ins.strip())
        if has_ra and has_ins:
            raise ValueError(f"contexts[{i - 1}]: set only one of rationalize_agent_outcome or insight, not both")
        if not has_ra and not has_ins:
            raise ValueError(f"contexts[{i - 1}]: set exactly one of rationalize_agent_outcome or insight")
        chunks: list[str] = [f'<CONTEXT index="{i}">\n\n']
        if has_ra:
            body = ra.strip().rstrip("\n")  # type: ignore[union-attr]
            chunks.extend(["<RATIONALIZE>\n\n", body, "\n\n</RATIONALIZE>\n"])
        else:
            kind_raw = ctx.get("insight_kind")
            safe_kind = _sanitize_insight_kind(kind_raw) if isinstance(kind_raw, str) else "transaction_insight"
            body_ins = ins.strip()  # type: ignore[union-attr]
            chunks.extend(
                [
                    f'<TRANSACTION_INSIGHT type="{safe_kind}">\n\n',
                    body_ins,
                    "\n\n</TRANSACTION_INSIGHT>\n",
                ]
            )
        chunks.extend(["\n<TAP_LINK>\n\n", _tap_link_xml_inner_line(tap_path=tl, ctx=ctx), "\n\n</TAP_LINK>\n", "\n</CONTEXT>"])
        inner_parts.append("".join(chunks))
    inner = "\n\n".join(inner_parts)
    return "<CONTEXTS>\n\n" + inner + "\n\n</CONTEXTS>\n"


def build_top_takeaways_user_message(
    *,
    rationalize_agent_outcomes: list[str],
    tap_links: list[str],
) -> str:
    """Shorthand when every context is a full rationalize outcome (same as zip → ``build_top_takeaways_user_message_from_contexts``)."""
    if not isinstance(rationalize_agent_outcomes, list) or not rationalize_agent_outcomes:
        raise ValueError("rationalize_agent_outcomes must be a non-empty list")
    if not isinstance(tap_links, list) or len(tap_links) != len(rationalize_agent_outcomes):
        raise ValueError("tap_links must be a list with the same length as rationalize_agent_outcomes.")
    contexts: list[dict[str, Any]] = [
        {"rationalize_agent_outcome": ra, "tap_link": tl} for ra, tl in zip(rationalize_agent_outcomes, tap_links)
    ]
    return build_top_takeaways_user_message_from_contexts(contexts=contexts)


# Canonical ``_TOP_TAKEAWAYS_PROMPT`` (rollup instructions for multi-context top takeaways).
SYSTEM_PROMPT = """You are **Penny**.

You will be given multiple **top takeaway contexts** inside ``<CONTEXTS>...</CONTEXTS>``. Each block is one context, in order:
1. ``<CONTEXT index="1">`` (index counts up for each block)
2. **Body** — **either** ``<RATIONALIZE>`` … ``</RATIONALIZE>`` (full prior rationalize markdown) **or** ``<TRANSACTION_INSIGHT type="…">`` … ``</TRANSACTION_INSIGHT>`` (a **single insight string** only: not rationalized — no ``# Rationalize What`` / Figures / Drivers). Types include e.g. ``uncat_txn``, ``large_txn``.
3. ``<TAP_LINK>`` — one line: ``<Category display name> - <relative drill-down URL>`` when a name is provided (e.g. ``Salary - /cashflow/36/monthly/2026-05``). The part after `` - `` must be the relative URL only (starts with ``/``). **Every context includes one** — required.
4. ``</CONTEXT>``

Your goal is to synthesize a **substantive rollup** across all contexts: **highlights** (positive patterns, wins, healthy habits) and **lowlights** (concerns, risks, drag, overspending, gaps). Each bullet should read as a **mini-briefing**—rich enough that a reader grasps the **what**, **how much**, **compared to what**, and **why** (drivers) without opening the drill-down, while staying within the bullet budget.

**What to use for Highlights / Lowlights:** For ``<RATIONALIZE>`` contexts, base bullets on **Figures and Drivers** plus the ``Explain:`` line. **Go deep:** weave in **multiple** concrete details from Figures (period labels, dollar amounts, prior-window comparisons) and Drivers (named payees, payroll or bonus lines, miscategorization examples, volume or timing mechanisms). Prefer **several connected sentences** per bullet when the source supports it—do **not** compress rich rationalize into a thin headline if the Figures/Drivers give more. Copy numbers and dates **exactly** as written. For ``<TRANSACTION_INSIGHT>`` contexts: the **only** ground truth is the insight text inside that tag — there are no Figures/Drivers; do **not** invent amounts or merchants not stated there. For ``type="uncat_txn"`` and ``type="large_txn"``, write in the same **observational** register as **vs-forecast** rationalize **Drivers** (declarative, third person: what the spend reflects, what is driven by, what posted when) — **no** invented action items or imperatives (e.g. no “confirm”, “verify”, “should”, “until you act”, “tidy”, “worth checking”).

Rules:
- Ground every bullet in the provided body for that context. For rationalize contexts, ground claims in ``## Figures`` and ``## Drivers`` as well as ``Explain:``. Do not invent transactions, ids, or amounts.
- **Depth (``<RATIONALIZE>``):** Within each ``- `` bullet, use **as much relevant Figures/Drivers material as fits**—typically **two to four sentences** when the context is meaty. Include **at least two** concrete facts (e.g. two amounts, or an amount plus a named driver, or a comparison across periods plus the mechanism). If you only restate ``Explain:`` without Figures/Drivers specifics, the bullet is **too shallow**.
- **Tap link anchor:** For ``<RATIONALIZE>`` contexts, choose the phrase you wrap in ``[...](tap_link)`` **only** from ``# Rationalize What`` / ``Explain:`` (ignore ``## Figures``, ``## Drivers``, and slugs like ``meals_groceries`` for this choice):
  1. If Explain **names a parent / umbrella category** in that same line, use **that parent’s wording** (verbatim casing from Explain) as the **primary** linked label.
  2. **Otherwise** use the **first leaf** category name in Explain (first such name in reading order).
  3. For ``<TRANSACTION_INSIGHT>`` contexts, choose a **short natural phrase from the insight text itself** as the link anchor (e.g. merchant name like “Property Group LLC” or “Burger King”, the word **Uncategorized**, or **Clothing** when it names a likely category — pick the clearest single anchor that fits the sentence; use **that** visible text with this context’s ``<TAP_LINK>`` URL).
  4. At least one Markdown link per bullet that summarizes a context must use the chosen anchor and **exactly** the ``/cashflow/...`` path from that context’s ``<TAP_LINK>`` line (the substring after `` - ``) as ``href`` (add a second link only if another context is also summarized in the same bullet).
- **Tap label vs Markdown anchor (hard rule):** When ``<TAP_LINK>`` is ``Name - /cashflow/...``, the ``[...]`` anchor for **that** path must name the **same** category as **Name**—do not attach that URL to a different category label (e.g. do not write **Groceries** or **Dining** on a tap whose line says **Food**). If Explain’s wording would disagree with **Name**, align the **link anchor** with **Name** and carry Explain’s nuance in the surrounding sentence without mislabeling the drill-down.
- **Tap links:** The ``<TAP_LINK>`` line pairs a **display name** with the tap path. In Markdown, use **only** the ``/cashflow/...`` portion as ``href``—copy it verbatim; do not paste the leading ``Name - `` prefix into links. **Link syntax:** ``[text](/path)`` with **no whitespace** between ``(`` and ``/`` — never ``[text]( /path)``. If multiple contexts appear in one bullet, include one correct link per context with each context’s own label rule and path. **Never** paste one context’s ``/cashflow/...`` URL next to another context’s category name.
- **Anchor vs path:** The ``href`` must be **exactly** the ``/cashflow/...`` path from that context’s ``<TAP_LINK>`` line (same context block). The visible ``[anchor]`` must name the **same** Penny category level that URL opens (parent rollup vs leaf): align with the **display name** in ``<TAP_LINK>`` when it names a category; do not write **Food** on a tap that points at **Groceries** or **Meals** alone, or **Groceries** on a tap meant for a different rollup. Do not paste another context’s path into a sentence tied to a different Explain theme.
- **Link in sentence:** The linked category must be **woven into the sentence** as normal grammar (subject, object, or “in …” phrase) — not bolted on after a period. **Bad:** ``…following bonus payouts in March. [Salary](/cashflow/…)`` **Good:** ``…following March bonus payouts; [Salary](/cashflow/…) matches your usual early-month rhythm`` or ``Early-month [Salary](/cashflow/…) matches April…``. Do not end a clause with ``.`` and then append only a bare link.
- **Tools:** You may call finance tools when helpful—especially to **verify** totals, reconcile conflicting numbers across contexts, or confirm whether a claim in the markdown still matches current data. Prefer tools when verification would materially increase confidence; otherwise rely on the pasted rationalize text.
- Merge semantically duplicate ideas into one bullet; avoid repeating the same point under different wording. When merging, keep the **combined** bullet **fully detailed** (do not drop figures or drivers just to shorten).
- **Bullet budget:** At most **3** ``- `` bullets under ``## Highlights`` and at most **3** under ``## Lowlights`` (six total max). If contexts are very thin, use fewer; never exceed 3 per section. **Prefer longer, denser bullets** over many short ones—use the budget for **depth**, not for splitting one story into vague fragments.
- Your **written reply** must not include raw ``llm_calls`` text or pasted tool-call payloads.
- **Final reply:** After any tool calls, your **only** user-visible text follows the markdown format below — **no** standalone preamble or wrap-up outside it. The first line of your last message must be exactly ``# Top Takeaways``.

Final message format (MUST match exactly — nothing else in the last message):

# Top Takeaways

## Highlights

- ...

## Lowlights

- ...

Notes:
- Use `- ` bullets only (no numbered lists in these sections). Inline Markdown links ``[text](/path)`` inside bullets are required for each context summarized; anchor text follows **Tap link anchor** rules (Explain-based for ``<RATIONALIZE>``, insight-text-based for ``<TRANSACTION_INSIGHT>``), subject to **Tap label vs Markdown anchor** when ``<TAP_LINK>`` includes a display name; ``href`` is the ``/cashflow/...`` path from that context’s ``<TAP_LINK>`` line with **no space** after ``(``. The linked phrase must read as part of the sentence (**Link in sentence** rule), not a trailing tag after the final period. **Anchor vs path** (same subsection in Rules): label and ``href`` must describe the same category level, and you must not apply one context’s URL to another context’s category name.
- Match the rationalize voice: **direct, readable, and thorough**—high information density per bullet, not telegraphic stubs. For ``uncat_txn`` / ``large_txn`` insights, match **vs-forecast Drivers** observational tone, not advisory copy.
- Start each bullet with a **bold** short label when helpful (e.g. **Savings:** …).
- Do not add introductory or closing sentences outside the ``# Top Takeaways`` structure.
"""

# Four rationalize markdown bodies; order: salary, groceries, uncategorized (week up), uncategorized (month slightly down).
_RATIONALIZE_FIXTURES: tuple[str, str, str, str] = (
    """# Rationalize What

Explain: Salary is significantly down this month at $6369. (2026-05-01 to 2026-05-10)

# Rationalize Response

## Figures

*   **May 1–10, 2026:** $6,369.24
*   **April 1–10, 2026:** $6,369.24
*   **March 1–10, 2026:** $17,369.24

## Drivers

The "significant drop" in salary compared to early March is due to a large, one-time bonus payout received in the first half of March 2026. Specifically, you received $8,800 in total "Genentech US Bonus" payments on March 11, alongside higher "CA State Payroll" amounts ($8,600 total) compared to your standard bi-weekly payroll cycle. Your income for the first 10 days of May is consistent with your regular pay cycle, mirroring the amount received during the same period in April.""",
    """# Rationalize What

Explain: Groceries is significantly down this month at $937.  Food is thus significantly down this month to $1859. (2026-05-01 to 2026-05-10)

# Rationalize Response

## Figures

*   **Total Meals (May 1–10, 2026):** $1,859.34
*   **Total Meals (Apr 1–30, 2026):** $4,039.28
*   **Total Meals (Mar 1–31, 2026):** $3,300.28
*   **Groceries (May 1–10, 2026):** $937.22
*   **Groceries (Apr 1–30, 2026):** $1,513.40
*   **Groceries (Mar 1–31, 2026):** $1,562.82

## Drivers

The significant decrease in food and grocery spending is primarily due to comparing only 10 days of May activity against full-month totals for April and March.

Additionally, your transaction history shows some potential miscategorization: several transactions at "Applebee's" (a dining venue) are currently categorized under `meals_groceries`. For example, $85.36 was recorded at Applebee's on May 2nd under groceries, whereas a typical grocery bill for a similar amount is expected from retailers like "Target" or "Trader Joe's".""",
    """# Rationalize What

Explain: Uncategorized is significantly up last week at $6382. (2026-05-03 to 2026-05-09)

# Rationalize Response

## Figures

*   **Uncategorized:** $6,382.12 (May 3–9, 2026)
*   **Uncategorized:** $725.52 (Apr 26–May 2, 2026)
*   **Uncategorized:** $500.46 (Apr 19–25, 2026)

## Drivers

The spike in uncategorized spending is primarily driven by two large payments to "Property Group LLC" totaling $5,700 ($2,850 each). Additionally, duplicate entries appear for several recurring transactions, such as "Community Pool" ($83.06 x 2), "Costco" ($69.71 x 2), and "BP" ($49.56 x 2), which suggests some transactions might be showing up twice across different accounts.""",
    """# Rationalize What

Explain: Uncategorized is slightly down this month at $6644. (2026-05-01 to 2026-05-10)

# Rationalize Response

## Figures

* **May 1–10, 2026**: $6,644.34 (Uncategorized)
* **Apr 1–30, 2026**: $7,681.54 (Uncategorized)
* **Mar 1–31, 2026**: $8,111.90 (Uncategorized)

## Drivers

The "Uncategorized" spend is trending downward compared to prior months. A significant portion of this category consists of recurring transactions like "Property Group LLC," which appears twice monthly at $2,850.00 each time ($5,700 total). The lower total for May so far is primarily due to a lower volume of smaller miscellaneous transactions (18 transactions in May vs. 28 in April) rather than a change in the major recurring charges.""",
)

# Same order as ``_RATIONALIZE_FIXTURES``: salary/income, food, uncategorized (week), uncategorized (month).
_TAP_LINK_FIXTURES: tuple[str, str, str, str] = (
    "/cashflow/36/monthly/2026-05",
    "/cashflow/1/monthly/2026-05",
    "/cashflow/-1/weekly/2026-05-03",
    "/cashflow/-1/monthly/2026-05",
)

# Four ``spend_vs_forecast``-style rationalizes (used with ``uncat_txn`` / ``large_txn`` bodies inlined in that test).
_VS_FORECAST_RATIONALIZES: tuple[str, str, str, str] = (
    """# Rationalize What

Explain: Leisure is significantly up this month at $412. (2026-05-01 to 2026-05-10)

# Rationalize Response

## Figures

*   **Leisure (May 1–10, 2026):** $412.18
*   **Leisure (Apr 1–30, 2026):** $251.00

## Drivers

Higher leisure spend vs forecast is driven by weekend entertainment and one concert ticket purchase.""",
    """# Rationalize What

Explain: Transport is significantly down this month at $68. (2026-05-01 to 2026-05-10)

# Rationalize Response

## Figures

*   **Transport (May 1–10, 2026):** $68.40
*   **Transport (Apr 1–30, 2026):** $310.25

## Drivers

Lower transport vs forecast reflects fewer commutes and no fuel fill-ups yet in early May.""",
    """# Rationalize What

Explain: Bills is slightly up this month at $1,240. (2026-05-01 to 2026-05-10)

# Rationalize Response

## Figures

*   **Bills (May 1–10, 2026):** $1,240.55
*   **Bills (Apr 1–30, 2026):** $1,180.00

## Drivers

Slight uptick vs forecast from annual software renewal posting in the first week of May.""",
    """# Rationalize What

Explain: Shopping is significantly up this month at $295. (2026-05-01 to 2026-05-10)

# Rationalize Response

## Figures

*   **Shopping (May 1–10, 2026):** $295.00
*   **Shopping (Apr 1–30, 2026):** $142.30

## Drivers

Shopping vs forecast rose after two online orders (electronics accessories and home goods) early in the month.""",
)


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
    """One-shot Gemini runner: canonical top-takeaways system prompt plus ``<CONTEXTS>`` user message."""

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
            raise ValueError("user_message must be a non-empty <CONTEXTS> bundle string.")
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
<CONTEXTS>

<CONTEXT index="1">

<RATIONALIZE>

# Rationalize What

Explain: Salary is significantly down this month at $6369. (2026-05-01 to 2026-05-10)

# Rationalize Response

## Figures

*   **May 1–10, 2026:** $6,369.24
*   **April 1–10, 2026:** $6,369.24
*   **March 1–10, 2026:** $17,369.24

## Drivers

The "significant drop" in salary compared to early March is due to a large, one-time bonus payout received in the first half of March 2026. Specifically, you received $8,800 in total "Genentech US Bonus" payments on March 11, alongside higher "CA State Payroll" amounts ($8,600 total) compared to your standard bi-weekly payroll cycle. Your income for the first 10 days of May is consistent with your regular pay cycle, mirroring the amount received during the same period in April.

</RATIONALIZE>

<TAP_LINK>

Salary - /cashflow/36/monthly/2026-05

</TAP_LINK>

</CONTEXT>

<CONTEXT index="2">

<RATIONALIZE>

# Rationalize What

Explain: Groceries is significantly down this month at $937.  Food is thus significantly down this month to $1859. (2026-05-01 to 2026-05-10)

# Rationalize Response

## Figures

*   **Total Meals (May 1–10, 2026):** $1,859.34
*   **Total Meals (Apr 1–30, 2026):** $4,039.28
*   **Total Meals (Mar 1–31, 2026):** $3,300.28
*   **Groceries (May 1–10, 2026):** $937.22
*   **Groceries (Apr 1–30, 2026):** $1,513.40
*   **Groceries (Mar 1–31, 2026):** $1,562.82

## Drivers

The significant decrease in food and grocery spending is primarily due to comparing only 10 days of May activity against full-month totals for April and March.

Additionally, your transaction history shows some potential miscategorization: several transactions at "Applebee's" (a dining venue) are currently categorized under `meals_groceries`. For example, $85.36 was recorded at Applebee's on May 2nd under groceries, whereas a typical grocery bill for a similar amount is expected from retailers like "Target" or "Trader Joe's".

</RATIONALIZE>

<TAP_LINK>

Food - /cashflow/1/monthly/2026-05

</TAP_LINK>

</CONTEXT>

<CONTEXT index="3">

<RATIONALIZE>

# Rationalize What

Explain: Uncategorized is significantly up last week at $6382. (2026-05-03 to 2026-05-09)

# Rationalize Response

## Figures

*   **Uncategorized:** $6,382.12 (May 3–9, 2026)
*   **Uncategorized:** $725.52 (Apr 26–May 2, 2026)
*   **Uncategorized:** $500.46 (Apr 19–25, 2026)

## Drivers

The spike in uncategorized spending is primarily driven by two large payments to "Property Group LLC" totaling $5,700 ($2,850 each). Additionally, duplicate entries appear for several recurring transactions, such as "Community Pool" ($83.06 x 2), "Costco" ($69.71 x 2), and "BP" ($49.56 x 2), which suggests some transactions might be showing up twice across different accounts.

</RATIONALIZE>

<TAP_LINK>

Uncategorized - /cashflow/-1/weekly/2026-05-03

</TAP_LINK>

</CONTEXT>

<CONTEXT index="4">

<RATIONALIZE>

# Rationalize What

Explain: Uncategorized is slightly down this month at $6644. (2026-05-01 to 2026-05-10)

# Rationalize Response

## Figures

* **May 1–10, 2026**: $6,644.34 (Uncategorized)
* **Apr 1–30, 2026**: $7,681.54 (Uncategorized)
* **Mar 1–31, 2026**: $8,111.90 (Uncategorized)

## Drivers

The "Uncategorized" spend is trending downward compared to prior months. A significant portion of this category consists of recurring transactions like "Property Group LLC," which appears twice monthly at $2,850.00 each time ($5,700 total). The lower total for May so far is primarily due to a lower volume of smaller miscellaneous transactions (18 transactions in May vs. 28 in April) rather than a change in the major recurring charges.

</RATIONALIZE>

<TAP_LINK>

Uncategorized - /cashflow/-1/monthly/2026-05

</TAP_LINK>

</CONTEXT>

</CONTEXTS>
""",
        "output": """
# Top Takeaways

## Highlights

- **Salary rhythm:** Early-month [Salary](/cashflow/36/monthly/2026-05) through May 10 matches April at $6,369.24 while March’s early-month lift came from a one-time Genentech bonus and higher CA State Payroll, not a pay cut.
- **Food framing:** [Food](/cashflow/1/monthly/2026-05) looks “down” partly because only 10 days of May are counted vs full April/March totals; Applebee’s dining charges may be sitting under groceries.

## Lowlights

- **Uncategorized spike:** Last week [Uncategorized](/cashflow/-1/weekly/2026-05-03) jumped to about $6,382 on large Property Group LLC payments and possible duplicate recurring charges (Community Pool, Costco, BP).
- **Uncategorized trend:** Month-to-date [Uncategorized](/cashflow/-1/monthly/2026-05) is slightly down vs March/April while big Property Group LLC repeats still dominate with fewer small misc transactions so far in May.
""",
    },
    {
        "name": "two_contexts_salary_and_groceries_with_tap_links",
        "input": """
<CONTEXTS>

<CONTEXT index="1">

<RATIONALIZE>

# Rationalize What

Explain: Salary is significantly down this month at $6369. (2026-05-01 to 2026-05-10)

# Rationalize Response

## Figures

*   **May 1–10, 2026:** $6,369.24
*   **April 1–10, 2026:** $6,369.24
*   **March 1–10, 2026:** $17,369.24

## Drivers

The "significant drop" in salary compared to early March is due to a large, one-time bonus payout received in the first half of March 2026. Specifically, you received $8,800 in total "Genentech US Bonus" payments on March 11, alongside higher "CA State Payroll" amounts ($8,600 total) compared to your standard bi-weekly payroll cycle. Your income for the first 10 days of May is consistent with your regular pay cycle, mirroring the amount received during the same period in April.

</RATIONALIZE>

<TAP_LINK>

Salary - /cashflow/36/monthly/2026-05

</TAP_LINK>

</CONTEXT>

<CONTEXT index="2">

<RATIONALIZE>

# Rationalize What

Explain: Groceries is significantly down this month at $937.  Food is thus significantly down this month to $1859. (2026-05-01 to 2026-05-10)

# Rationalize Response

## Figures

*   **Total Meals (May 1–10, 2026):** $1,859.34
*   **Total Meals (Apr 1–30, 2026):** $4,039.28
*   **Total Meals (Mar 1–31, 2026):** $3,300.28
*   **Groceries (May 1–10, 2026):** $937.22
*   **Groceries (Apr 1–30, 2026):** $1,513.40
*   **Groceries (Mar 1–31, 2026):** $1,562.82

## Drivers

The significant decrease in food and grocery spending is primarily due to comparing only 10 days of May activity against full-month totals for April and March.

Additionally, your transaction history shows some potential miscategorization: several transactions at "Applebee's" (a dining venue) are currently categorized under `meals_groceries`. For example, $85.36 was recorded at Applebee's on May 2nd under groceries, whereas a typical grocery bill for a similar amount is expected from retailers like "Target" or "Trader Joe's".

</RATIONALIZE>

<TAP_LINK>

Food - /cashflow/1/monthly/2026-05

</TAP_LINK>

</CONTEXT>

</CONTEXTS>
""",
        "output": """
# Top Takeaways

## Highlights

- **Income stability:** May 1–10 [Salary](/cashflow/36/monthly/2026-05) aligns with April’s same window at $6,369.24; March’s higher early-month income reflected bonus and payroll timing.
- **Groceries vs dining:** [Food](/cashflow/1/monthly/2026-05) looks lower partly from comparing 10 days of May to full months; Applebee’s may be miscategorized under groceries.

## Lowlights

- **Applebee’s pattern:** Dining at Applebee’s recorded under groceries inflates grocery totals until those charges are recategorized.
""",
    },
    {
        "name": "uncat_txn_large_txn_plus_four_vs_forecast",
        "input": """
<CONTEXTS>

<CONTEXT index="1">

<TRANSACTION_INSIGHT type="uncat_txn">

Uncategorized outflow transaction for Property Group LLC for $2,850 with a likely category of Clothing.

</TRANSACTION_INSIGHT>

<TAP_LINK>

Property Group LLC - /cashflow/transaction/123

</TAP_LINK>

</CONTEXT>

<CONTEXT index="2">

<TRANSACTION_INSIGHT type="large_txn">

Large outflow transaction with Burger King for $40 last 05/09.

</TRANSACTION_INSIGHT>

<TAP_LINK>

Burger King - /cashflow/transaction/456

</TAP_LINK>

</CONTEXT>

<CONTEXT index="3">

<RATIONALIZE>

# Rationalize What

Explain: Leisure is significantly up this month at $412. (2026-05-01 to 2026-05-10)

# Rationalize Response

## Figures

*   **Leisure (May 1–10, 2026):** $412.18
*   **Leisure (Apr 1–30, 2026):** $251.00

## Drivers

Higher leisure spend vs forecast is driven by weekend entertainment and one concert ticket purchase.

</RATIONALIZE>

<TAP_LINK>

Leisure - /cashflow/7/monthly/2026-05

</TAP_LINK>

</CONTEXT>

<CONTEXT index="4">

<RATIONALIZE>

# Rationalize What

Explain: Transport is significantly down this month at $68. (2026-05-01 to 2026-05-10)

# Rationalize Response

## Figures

*   **Transport (May 1–10, 2026):** $68.40
*   **Transport (Apr 1–30, 2026):** $310.25

## Drivers

Lower transport vs forecast reflects fewer commutes and no fuel fill-ups yet in early May.

</RATIONALIZE>

<TAP_LINK>

Transport - /cashflow/8/monthly/2026-05

</TAP_LINK>

</CONTEXT>

<CONTEXT index="5">

<RATIONALIZE>

# Rationalize What

Explain: Bills is slightly up this month at $1,240. (2026-05-01 to 2026-05-10)

# Rationalize Response

## Figures

*   **Bills (May 1–10, 2026):** $1,240.55
*   **Bills (Apr 1–30, 2026):** $1,180.00

## Drivers

Slight uptick vs forecast from annual software renewal posting in the first week of May.

</RATIONALIZE>

<TAP_LINK>

Bills - /cashflow/9/monthly/2026-05

</TAP_LINK>

</CONTEXT>

<CONTEXT index="6">

<RATIONALIZE>

# Rationalize What

Explain: Shopping is significantly up this month at $295. (2026-05-01 to 2026-05-10)

# Rationalize Response

## Figures

*   **Shopping (May 1–10, 2026):** $295.00
*   **Shopping (Apr 1–30, 2026):** $142.30

## Drivers

Shopping vs forecast rose after two online orders (electronics accessories and home goods) early in the month.

</RATIONALIZE>

<TAP_LINK>

Shopping - /cashflow/44/monthly/2026-05

</TAP_LINK>

</CONTEXT>

</CONTEXTS>
""",
        "output": """
# Top Takeaways

## Highlights

- **Uncategorized outflow:** The insight flags [Property Group LLC](/cashflow/transaction/123) as an uncategorized **$2,850** outflow with **Clothing** as the likely category.
- **Leisure vs plan:** [Leisure](/cashflow/7/monthly/2026-05) is up vs forecast on weekend entertainment and a concert ticket early in May.
- **Shopping lift:** [Shopping](/cashflow/44/monthly/2026-05) is up vs forecast after two online orders for accessories and home goods.

## Lowlights

- **Uncategorized mix:** Uncategorized totals still include [Property Group LLC](/cashflow/transaction/123) at **$2,850** while that charge posts without a category.
- **Large dining outflow:** [Burger King](/cashflow/transaction/456) shows a **$40** single-ticket outflow on 05/09.
- **Commutes and bills:** [Transport](/cashflow/8/monthly/2026-05) is down vs forecast with fewer commutes and no fuel fill-ups yet, while [Bills](/cashflow/9/monthly/2026-05) ticked slightly up on an annual software renewal in early May.
""",
    },
]

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


def run_test(name_or_index: str | int, optimizer: TopTakeawaysVerboseOptimizer | None = None) -> str | None:
    tc = get_test_case(name_or_index) if not isinstance(name_or_index, dict) else name_or_index
    if tc is None:
        print(f"Test case {name_or_index!r} not found.")
        return None
    print(f"\n{'=' * 80}\nRunning test: {tc['name']}\n{'=' * 80}\n")
    return _run_test(tc["input"], optimizer)


def main(*, test: str | None) -> None:
    if test is None:
        print("Usage: python3 top_takeaways_verbose_optimizer.py --test <index|name|all>")
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
