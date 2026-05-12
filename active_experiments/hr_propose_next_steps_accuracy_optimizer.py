"""
Propose-next-steps rubric optimizer — **accuracy only**.

Grades only the **accuracy** axis.

Run from `finance-ai-penny` repo root:

  python3 active_experiments/hr_propose_next_steps_accuracy_optimizer.py --test all
  python3 active_experiments/hr_propose_next_steps_accuracy_optimizer.py --test all --check
  python3 active_experiments/hr_propose_next_steps_accuracy_optimizer.py --batch 1
  python3 active_experiments/hr_propose_next_steps_accuracy_optimizer.py --batch 1 --check
  # Batches 1–4 partition fixtures (see each test case's "batch" field).
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


def _parse_expected_output(raw: str | None) -> dict[str, Any] | None:
  if not raw:
    return None
  try:
    return json.loads(raw)
  except Exception:
    return None


TEST_CASES: list[dict[str, Any]] = [
  {
    "name": "target_rule_tools_match_costco",
    "batch": 1,
    "input": """<CONTEXTS>

<CONTEXT index="1">

<RATIONALIZE>

# Rationalize What

Explain: Groceries is significantly down this month at $120. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

- Groceries: $120 this month (Apr 1–30, 2026).

## Drivers

Several Costco transactions appear miscategorized as `miscellaneous`.

## Next steps

Create a categorization rule so Costco is always `meals_groceries`.

</RATIONALIZE>

</CONTEXT>

</CONTEXTS>

<PROPOSAL>

# Proposal

## Proposed next steps

1. **Add** a categorization rule so merchant substring “costco” maps to `meals_groceries` for past and future.

## Open items (not addressed)

1. **Review** whether any non-Costco groceries are still landing in `miscellaneous`.

</PROPOSAL>

<CALLS>

# Round 1
## Invoked tools

1. **`propose_create_categorization_rule`**

```json
{
  "rule": {
    "name_sub_eq": "costco"
  },
  "ai_category_id": 4,
  "scope": "future_and_past",
  "rationale": "Costco should count as groceries."
}
```

</CALLS>
""",
    "output": '{"score": 5, "notes": "Proposal stays inside the rationalize facts and the rule tool matches the stated `meals_groceries` intent."}',
  },
  {
    "name": "rule_intent_only_inappropriate_tools_called",
    "batch": 1,
    "input": """<CONTEXTS>

<CONTEXT index="1">

<RATIONALIZE>

# Rationalize What

Explain: Target REDcard charges keep landing in `shopping` instead of `shopping_clothing`. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

- About $310 across Target runs still tagged to the parent `shopping` bucket.

## Drivers

Descriptions look like apparel and home goods mixed; the clothing-heavy pattern is stable.

## Next steps

Add a categorization rule so Target REDcard lines default to `shopping_clothing` unless the memo clearly says grocery pickup.

</RATIONALIZE>

</CONTEXT>

</CONTEXTS>

<PROPOSAL>

# Proposal

## Proposed next steps

1. **Create** a categorization rule so merchant text matching “TARGET REDCARD” maps to `shopping_clothing` for past and future.

## Open items (not addressed)

1. None.

</PROPOSAL>

<CALLS>

# Round 1
## Invoked tools

1. **`propose_retrieve_transactions`**

```json
{
  "date_range": {
    "start": "2026-04-01",
    "end": "2026-04-30"
  },
  "ai_category_id_in": [
    21,
    44
  ],
  "rationale": "Scan shopping parents before edits."
}
```

2. **`propose_create_goal`**

```json
{
  "category": "shopping_clothing",
  "goal_type": "spending_budget",
  "goal_title": "Cap Target apparel",
  "target_amount": 400,
  "time_horizon": "monthly",
  "rationale": "Reduce mis-tagged apparel drift."
}
```

</CALLS>
""",
    "output": '{"score": 3, "notes": "The rationalize thread calls for a categorization rule, but every listed tool is the wrong family (retrieve + goal), so the trace is materially inappropriate."}',
  },
  {
    "name": "rule_intent_no_tools_in_calls",
    "batch": 1,
    "input": """<CONTEXTS>

<CONTEXT index="1">

<RATIONALIZE>

# Rationalize What

Explain: Groceries is significantly down this month at $120. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

- Groceries: $120 this month (Apr 1–30, 2026).

## Drivers

Several Costco transactions appear miscategorized as `miscellaneous`.

## Next steps

Create a categorization rule so Costco is always `meals_groceries`.

</RATIONALIZE>

</CONTEXT>

</CONTEXTS>

<PROPOSAL>

# Proposal

## Proposed next steps

1. **Add** a categorization rule so merchant substring “costco” maps to `meals_groceries` for past and future.

## Open items (not addressed)

1. **Review** whether any non-Costco groceries are still landing in `miscellaneous`.

</PROPOSAL>

<CALLS>

# Round 1
## Invoked tools

_No tool calls this round._

</CALLS>
""",
    "output": '{"score": 5, "notes": "Grounding and category intent are sound; missing rule automation is a completeness gap, not an accuracy defect."}',
  },
  {
    "name": "categorization_rule_targets_invalid_investments_slug",
    "batch": 2,
    "input": """<CONTEXTS>

<CONTEXT index="1">

<RATIONALIZE>

# Rationalize What

Explain: Recurring “FIDELITY WITHDRAWAL” debits still tagged as `transfers` this month. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

- Four $400 pulls labeled “FIDELITY WITHDRAWAL” sitting in `transfers`.

## Drivers

Memo text looks like brokerage cash movements rather than peer-to-peer transfers.

## Next steps

If the user confirms these are reinvestment sweeps, add a narrow merchant rule so “FIDELITY WITHDRAWAL” maps to the user-chosen leaf (not `transfers`).

</RATIONALIZE>

</CONTEXT>

</CONTEXTS>

<PROPOSAL>

# Proposal

## Proposed next steps

1. **Create** a categorization rule so merchant substring “FIDELITY WITHDRAWAL” maps to **`investments`** for past and future.

## Open items (not addressed)

1. None.

</PROPOSAL>

<CALLS>

# Round 1
## Invoked tools

_No tool calls this round._

</CALLS>
""",
    "output": '{"score": 4, "notes": "Otherwise tied to the stated Fidelity pattern, but `investments` is not a canonical Penny slug, so the stated rule target cannot be executed as written."}',
  },
  {
    "name": "categorization_rule_targets_invalid_bills_general_slug",
    "batch": 2,
    "input": """<CONTEXTS>

<CONTEXT index="1">

<RATIONALIZE>

# Rationalize What

Explain: Uncategorized ACH pulls from the college bursar this month. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

- Two “STATECOLLE BURSAR ONLINE” debits totaling $1,240 remain in `uncategorized`.

## Drivers

Descriptions look like tuition-related school charges rather than generic transfers.

## Next steps

Add a narrow categorization rule so merchant substring “STATECOLLE BURSAR ONLINE” stops posting as `uncategorized`.

</RATIONALIZE>

</CONTEXT>

</CONTEXTS>

<PROPOSAL>

# Proposal

## Proposed next steps

1. **Create** a categorization rule so merchant substring “STATECOLLE BURSAR ONLINE” maps to **`bills_general`** for past and future.

## Open items (not addressed)

1. None.

</PROPOSAL>

<CALLS>

# Round 1
## Invoked tools

_No tool calls this round._

</CALLS>
""",
    "output": '{"score": 4, "notes": "Grounding matches the bursar pattern, but `bills_general` is not a canonical Penny slug, so the rule target cannot be executed as stated."}',
  },
  {
    "name": "zelle_example_pairing_without_user_confirmation",
    "batch": 2,
    "input": """<CONTEXTS>

<CONTEXT index="1">

<RATIONALIZE>

# Rationalize What

Explain: Uncategorized spend is elevated this month. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

- Uncategorized volume is elevated vs prior month.

## Drivers

Uncategorized is high because of several Zelle to Jose transactions that require the user to confirmation of their categorizations.

## Next steps

(1) Confirm Zelle to Jose category, (2) Set categorization rule for Zelle to Jose -> Transfer.

</RATIONALIZE>

</CONTEXT>

</CONTEXTS>

<PROPOSAL>

# Proposal

## Proposed next steps

1. **(1) Confirm** Zelle to Jose category (user).
2. **(2) Set** categorization rule for Zelle to Jose -> `transfers` (Penny: create rule for past and future).

## Open items (not addressed)

1. None.

</PROPOSAL>

<CALLS>

# Round 1
## Invoked tools

1. **`propose_create_categorization_rule`**

```json
{
  "rule": {
    "name_sub_eq": "zelle to jose"
  },
  "ai_category_id": 45,
  "scope": "future_and_past",
  "rationale": "Apply transfer mapping once pattern is confirmed."
}
```

</CALLS>
""",
    "output": '{"score": 4, "notes": "Rationalize sequences confirm-then-rule; treating the `transfers` rule as immediate Penny work in parallel with confirmation skips the gate, so accuracy is slightly off."}',
  },
  {
    "name": "venmo_example_rule_treated_as_mandatory",
    "batch": 2,
    "input": """<CONTEXTS>

<CONTEXT index="1">

<RATIONALIZE>

# Rationalize What

Explain: Uncategorized peer-to-peer lines this month. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

- Several Venmo to Maria charges remain uncategorized.

## Drivers

Determine categorization of Venmo to Maria and set a categorization rule (eg. Venmo to Maria -> `bills_service_fees`).

## Next steps

Determine categorization of Venmo to Maria and set a categorization rule (eg. Venmo to Maria -> `bills_service_fees`).

</RATIONALIZE>

</CONTEXT>

</CONTEXTS>

<PROPOSAL>

# Proposal

## Proposed next steps

1. **(1) Set** categorization rule for Venmo to Maria -> `bills_service_fees` for past and future.

## Open items (not addressed)

1. None.

</PROPOSAL>

<CALLS>

# Round 1
## Invoked tools

1. **`propose_create_categorization_rule`**

```json
{
  "rule": {
    "name_sub_eq": "venmo to maria"
  },
  "ai_category_id": 13,
  "scope": "future_and_past",
  "rationale": "Encode the example mapping as production."
}
```

</CALLS>
""",
    "output": '{"score": 4, "notes": "The Venmo → `bills_service_fees` pairing was introduced with eg. as an illustration, not a mandate; implementing it as a definitive rule over-reads the rationalize text."}',
  },
  {
    "name": "hallucinated_wire_amount",
    "batch": 3,
    "input": """<CONTEXTS>

<CONTEXT index="1">

<RATIONALIZE>

# Rationalize What

Explain: `bills_service_fees` are flat month over month. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

- Service fees: $88 this month vs $86 last month.

## Drivers

Bank maintenance line items repeat at similar amounts.

## Next steps

Create a monthly cap goal for `bills_service_fees` if the user wants tighter control.

</RATIONALIZE>

</CONTEXT>

</CONTEXTS>

<PROPOSAL>

# Proposal

## Proposed next steps

1. **Recategorize** the $9,400 international wire on 2026-04-11 from `miscellaneous` into `bills_service_fees` before setting any cap goal.

## Open items (not addressed)

1. None.

</PROPOSAL>

<CALLS>

# Round 1
## Invoked tools

1. **`propose_recategorize_transactions`**

```json
{
  "transaction_ids": [
    888001
  ],
  "target_category": "bills_service_fees",
  "rationale": "Align the wire with service-fee narrative."
}
```

</CALLS>
""",
    "output": '{"score": 1, "notes": "Invents a large wire and transaction id that never appear in rationalize, so grounding fails outright."}',
  },
  {
    "name": "service_fees_goal_tools_match",
    "batch": 3,
    "input": """<CONTEXTS>

<CONTEXT index="1">

<RATIONALIZE>

# Rationalize What

Explain: Service Fees are significantly down this month. (credit card interest charges) (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

- Service Fees / interest charges: down vs last month.

## Drivers

Credit card interest charges decreased.

## Next steps

1. Consider setting a budget for service fees to keep interest charges low.
2. Review APR / statement details to confirm why interest changed.

</RATIONALIZE>

</CONTEXT>

</CONTEXTS>

<PROPOSAL>

# Proposal

## Proposed next steps

1. **Create** a monthly spending budget of $250 for `bills_service_fees` to support the current downward trend in credit card interest charges.

## Open items (not addressed)

1. **Review** credit card account statements and APR details to determine if the interest reduction resulted from a lower average daily balance or a rate change.
2. **Explore** debt repayment strategies or balance transfer options to eliminate remaining recurring interest fees.

</PROPOSAL>

<CALLS>

# Round 1
## Invoked tools

1. **`propose_create_goal`**

```json
{
  "category": "bills_service_fees",
  "goal_type": "spending_budget",
  "rationale": "Based on the recent reduction in credit card interest charges, a $250 monthly budget for service fees will help sustain this positive trend and encourage continued debt management.",
  "time_horizon": "monthly",
  "goal_title": "Limit Service Fees",
  "target_amount": 250
}
```

# Round 2
_No tool calls this round._

</CALLS>
""",
    "output": '{"score": 5, "notes": "Figures and drivers stay consistent with the proposal, and the goal tool matches the `bills_service_fees` automation described."}',
  },
  {
    "name": "dual_context_service_fees_thread_only",
    "batch": 4,
    "input": """<CONTEXTS>

<CONTEXT index="1">

<RATIONALIZE>

# Rationalize What

Explain: Service Fees are significantly down this month. (credit card interest charges) (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

- Service Fees / interest charges: down vs last month.

## Drivers

Credit card interest charges decreased.

## Next steps

1. Consider setting a budget for service fees to keep interest charges low.
2. Review APR / statement details to confirm why interest changed.

</RATIONALIZE>

</CONTEXT>

<CONTEXT index="2">

<RATIONALIZE>

# Rationalize What

Explain: Kids education spending this month. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

- Kids education: $180 this month.

## Next steps

Review recurring tutoring charges for consistency.

</RATIONALIZE>

</CONTEXT>

</CONTEXTS>

<PROPOSAL>

# Proposal

## Proposed next steps

1. **Create** a monthly spending budget of $250 for `bills_service_fees` to sustain the downward trend in interest charges.

## Open items (not addressed)

1. None.

</PROPOSAL>

<CALLS>

# Round 1
## Invoked tools

1. **`propose_create_goal`**

```json
{
  "category": "bills_service_fees",
  "goal_type": "spending_budget",
  "goal_title": "Limit Service Fees",
  "target_amount": 250,
  "time_horizon": "monthly",
  "rationale": "Align with reduced interest narrative."
}
```

</CALLS>
""",
    "output": '{"score": 5, "notes": "Service-fees narrative and goal tool stay faithful to context 1; skipping the tutoring thread is completeness, not an accuracy error here."}',
  },
]


def _build_output_schema(_types: Any) -> Any:
  return _types.Schema(
    type=_types.Type.OBJECT,
    required=["score", "notes"],
    properties={
      "score": _types.Schema(type=_types.Type.INTEGER, description="Integer 1–5."),
      "notes": _types.Schema(type=_types.Type.STRING, description="One short sentence."),
    },
  )


SYSTEM_PROMPT = """Grade **accuracy** only.

You are a **strict rubric grader** for the checker bundle below (XML-style role wrappers; do not require table/column names).

You receive **evidence**, then **`<PROPOSAL>`**, then **`<CALLS>`**.

**Evidence:**

- **`<CONTEXTS>` … `</CONTEXTS>`**: one or more **`<CONTEXT index="N">`** blocks. Each **`<CONTEXT>`** contains **`<RATIONALIZE>` … `</RATIONALIZE>`** (figures, drivers, next steps). Optionally **`<RATIONALIZE_CALLS>` … `</RATIONALIZE_CALLS>`** holds **that rationalize run’s** stored LLM trace — **not** the propose run. Alternatively, the bundle may use a single top-level **`<RATIONALIZE>` … `</RATIONALIZE>`** with no **`<CONTEXTS>`** wrapper (same markdown as inside a context).

**How to read `## Next steps` in `<RATIONALIZE>`:** Treat each bullet as **mandatory for the proposal only if** it reads as a direct instruction to Penny or the user without being framed as optional, hypothetical, or illustrative. **Do not** treat lines as binding when they are clearly **examples** (“example only”, “e.g.”, “eg.”, “for instance”, “illustration”, “if the user decides…”, parenthetical “(Example: …)”), **exploratory** (“consider…”, “think about…”), or **explicitly gated** on **user confirmation** or **ordering** (“after the user confirms…”, “once the user picks…”, “only after that…”, “(1) … (2) …” where **(1)** must precede **(2)**). **Before scoring 5:** scan for those cues; if **`<PROPOSAL>`** turns an **example** pairing, an **eg.** illustration, or a **(2) rule** into **immediate** Penny automation **without** satisfying the prerequisite or **(1)** step, that is **not** faithful reading of **`<CONTEXTS>`** → **typically 4** (axis 5), not 5.

For **grounding**, treat **all** **`<RATIONALIZE>`** bodies as the **combined** evidence. **`<PROPOSAL>`** must not assert facts (merchants, amounts, dates, transaction ids) that are absent from or contradicted by that evidence. **Fabricated** concrete transactions, wires, or ids invented in **`<PROPOSAL>`** or **`<CALLS>`** that never appear in rationalize → **score 1**; do **not** soften invented facts to **3** just because a tool was invoked.

**Numeric `ai_category_id` vs slugs:** `propose_create_categorization_rule` and similar tools often use **`ai_category_id`** (integer). That is **not** an error by itself when it maps to the same leaf as the slug in **`<PROPOSAL>`** (see `categories.py` / `penny/tools/utils.py`, e.g. **4 ↔ `meals_groceries`**, **22 ↔ `shopping_clothing`**, **17 ↔ `shelter_upkeep`**, **45 ↔ `transfers`**). **Do not** mark a bundle inaccurate solely because JSON used an integer ID. **However:** any **slug spelled out in `<PROPOSAL>`** (including in backticks) must still be **word-for-word** one entry from the **Category List** below—IDs do not excuse invented slug strings in the proposal text.

**Then:**

1. **`<PROPOSAL>` … `</PROPOSAL>`** — Markdown from the **propose** outcome (`agent_outcome`): normally **only** the **`# Proposal`** block with **`## Proposed next steps`** and **`## Open items (not addressed)`** (or equivalent **`##`** headings if `# Proposal` was omitted).

2. **`<CALLS>` … `</CALLS>`** — Markdown listing **LLM round-trips** from **this same propose / propose-multi run** (`calls` field): `# Round N`, optional metrics, **`## Invoked tools`** with numbered tools and fenced argument blocks.

**Do not** confuse **`<RATIONALIZE_CALLS>`** inside a **`<CONTEXT>`** with **`<CALLS>`** at the end. Grade tool consistency for the propose run using **`<CALLS>`** only.

Grade **only** what is in the message. Do not invent missing data.

**Out of scope — never lower accuracy solely for these (non-factors; keep score at 5 if nothing else is wrong):**
- Whether every rationalize **Next steps** bullet is reflected in **`<PROPOSAL>`** (completeness).
- Whether **`<PROPOSAL>`** describes Penny automation (rules, goals, retrieve, recategorize) while **`<CALLS>`** lists **no** matching tool invocations (`_No tool calls this round._`, empty rounds, etc.). Missing tools vs proposal text is **completeness**, **not** accuracy—**do not** reduce the score for that gap alone; when grounding and category intent otherwise match rationalize, **prefer score 5** unless another accuracy defect applies.
- Sectioning between **Proposed next steps** vs **Open items** except where it creates a **factual** contradiction with **`<RATIONALIZE>`** text.

**Axis (return ONLY `{score, notes}`):**

**Accuracy**
1. **Grounding:** No invented merchants/amounts/dates/transaction ids; no contradictions vs **any** **`<RATIONALIZE>`** body.
2. **Penny fit:** Proposed steps align with Penny’s capabilities and the **mandatory** (non-example) drivers and next steps.
3. **Tool appropriateness (when `<CALLS>` lists at least one concrete tool invocation):** Tools must support the **Proposed next steps** (right family and sensible args). If **`<PROPOSAL>`** clearly calls for a **categorization rule** (merchant/payee → category automation) but **every** tool in **`<CALLS>`** is from **other** families only (e.g. **only** `propose_retrieve_transactions` + `propose_create_goal`, with **no** `propose_create_categorization_rule`), that mismatch is **always score 3**—do **not** soften to **4** or **5**. **If `<CALLS>` has no tool invocations at all, skip this axis item** (missing tools are completeness); an invalid slug in **`<PROPOSAL>`** alone is then axis **4**, not **3**. When **no** tools are listed, **no** tool-appropriateness penalty.
4. **Invalid categorization (text only):** If the **only** substantive defect is a **string** in **`<PROPOSAL>`** that is **not** one of the Category List tokens **exactly** (character-for-character), e.g. `investments`, `bills_general`, while the narrative is otherwise grounded, **cap** at **4** for one slip, **3** only for multiple invalid labels—**never 1 or 2** for invalid-label-only cases. This rule does **not** apply to valid numeric `ai_category_id` that match the intended leaf. **Anchor (invalid slug only):** If your **only** reason to dip below **5** is a **single** made-up label **not** in the Category List (e.g. `bills_general`, `investments`), choose **4**, not **3**—**3** is for wrong-tool traces or **several** stacked problems. **Do not** use this anchor for confirmation-order / example issues (axis 5).
5. **Illustrative / confirmation-gated next steps:** As in the reading rules above—immediate rule that copies an example or skips required user confirmation → **typically 4**. **Ordered confirm-then-rule:** If rationalize lists **(1) confirm category** then **(2) set categorization rule** and **`<PROPOSAL>`** still schedules **(2)** as immediate Penny rule automation **alongside** **(1)** without stating the rule runs **only after** confirmation, treat that as a **gate / sequencing** error → **4**, not **5**, even when the target slug (e.g. `transfers`) is valid.

**Category List** — **exact tokens only** (lowercase, underscores as shown). A categorization target in **`<PROPOSAL>`** or a string `category` / slug in tools must be **identical** to one line below—no paraphrase, no Title Case, no human labels (“Groceries”, “Service Fees”), no invented compounds (`bills_general`, `investments`, `dining_coffee`). If it is not on this list **character-for-character**, it is invalid for axis 4.

- `meals_dining_out`
- `meals_delivered_food`
- `meals_groceries`
- `leisure_entertainment`
- `leisure_travel`
- `shopping_pets`
- `bills_connectivity`
- `bills_insurance`
- `bills_tax`
- `bills_service_fees`
- `shelter_home`
- `shelter_utilities`
- `shelter_upkeep`
- `education_kids_activities`
- `education_tuition`
- `shopping_clothing`
- `shopping_gadgets`
- `shopping_kids`
- `transportation_car`
- `transportation_public`
- `health_medical_pharmacy`
- `health_gym_wellness`
- `health_personal_care`
- `donations_gifts`
- `miscellaneous`
- `income_salary`
- `income_sidegig`
- `income_business`
- `income_interest`
- `transfers`

**Calibration:** **5** = no meaningful gap on that axis. **4** = one minor gap. **3** = clear but fixable issue. **2** = several problems. **1** = axis largely failed. **3 vs 4:** Do **not** use **3** when the sole defect is one invalid categorization **text** slug (see axis 4 anchor)—that case is **4**.

Return **only** the JSON object matching the schema (`score`, `notes`).
"""


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
    user_msg = (bundled_input or "").strip()
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
    thought_summary = ""
    if hasattr(out, "candidates") and out.candidates:
      for candidate in out.candidates:
        if hasattr(candidate, "content") and candidate.content:
          if hasattr(candidate.content, "parts") and candidate.content.parts:
            for part in candidate.content.parts:
              if hasattr(part, "thought") and part.thought:
                if hasattr(part, "text") and part.text:
                  thought_summary += part.text
    if thought_summary.strip():
      print(f"{'=' * 80}")
      print("THOUGHT SUMMARY:")
      print(thought_summary.strip())
      print("=" * 80)
    text = (out.text or "").strip()
    try:
      return json.loads(text)
    except Exception:
      s = text[text.find("{") : text.rfind("}") + 1] if ("{" in text and "}" in text) else "{}"
      return json.loads(s)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--test", type=str, default=None, help="Test name, index, or 'all'.")
  parser.add_argument("--batch", type=int, default=None, help="Run all tests in batch N.")
  parser.add_argument("--model", type=str, default="gemini-flash-lite-latest")
  parser.add_argument("--max-output-tokens", type=int, default=256)
  parser.add_argument("--thinking-budget", type=int, default=0)
  parser.add_argument(
    "--check",
    action="store_true",
    help="After grading, require integer score to match expected output JSON. Exits non-zero on failure.",
  )
  args = parser.parse_args()

  if args.test is None and args.batch is None:
    print("Available test cases:")
    for i, tc in enumerate(TEST_CASES):
      batch = tc.get("batch")
      batch_s = str(batch) if isinstance(batch, int) else "—"
      print(f"  {i}: {tc.get('name')} (batch {batch_s})")
    return

  opt = ProposeNextStepsAccuracyCheckerOptimizer(
    model_name=args.model,
    max_output_tokens=args.max_output_tokens,
    thinking_budget=args.thinking_budget,
  )

  if args.batch is not None:
    cases = [(i, tc) for i, tc in enumerate(TEST_CASES) if int(tc.get("batch") or 0) == int(args.batch)]
    if not cases:
      raise SystemExit(f"No tests found for batch={args.batch}")
  elif (args.test or "").strip().lower() == "all":
    cases = list(enumerate(TEST_CASES))
  else:
    if (args.test or "").isdigit():
      idx = int(args.test)
      if idx < 0 or idx >= len(TEST_CASES):
        raise SystemExit(f"Test index out of range: {idx}")
      cases = [(idx, TEST_CASES[idx])]
    else:
      idx_tc = next(((i, t) for i, t in enumerate(TEST_CASES) if t.get("name") == args.test), None)
      if idx_tc is None:
        raise SystemExit(f"Unknown test: {args.test!r}")
      cases = [idx_tc]

  failures: list[str] = []

  for run_i, (case_index, tc) in enumerate(cases):
    if run_i:
      print(f"\n{_TEST_SEPARATOR}\n")
    batch = tc.get("batch")
    batch_s = str(batch) if isinstance(batch, int) else "—"
    print(f"# Test: {case_index}  {tc['name']}  (batch {batch_s})\n")
    _print_section_banner("# LLM Checker Input")
    print(tc["input"])
    result = opt.grade(tc["input"])
    _print_section_banner("# LLM Checker Response")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.check:
      exp = _parse_expected_output(tc.get("output"))
      if not isinstance(exp, dict) or "score" not in exp:
        failures.append(f"{tc.get('name')}: invalid expected output JSON")
      else:
        try:
          got = int(result.get("score"))
          want = int(exp["score"])
        except Exception:
          failures.append(f"{tc.get('name')}: non-integer score")
        else:
          if got != want:
            failures.append(f"{tc.get('name')}: score {got} != expected {want}")
    if tc.get("output") is not None:
      _print_section_banner("# Expected Output")
      print(tc["output"])

  print(f"\n{_TEST_SEPARATOR}\n")
  print(f"# Total tests: {len(cases)}\n")
  if args.check and failures:
    print("# CHECK FAILURES\n")
    for line in failures:
      print(line)
    raise SystemExit(1)
  if args.check and not failures:
    print("# CHECK: all cases passed score match.\n")


if __name__ == "__main__":
  main()

