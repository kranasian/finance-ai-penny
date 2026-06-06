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
    "name": "zelle_maria_rule_no_category_dining_out_inappropriate",
    "batch": 1,
    "input": """<RATIONALIZE_1>

# Rationalize What

Explain: Several Zelle payments to Maria are sitting in uncategorized spend this month. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

- **Current Month (Apr 1–30, 2026) — uncategorized Zelle to Maria:** $840.00 across 6 posted transfers in `miscellaneous`.
- **Prior Month (Mar 1–31, 2026) — same payee pattern:** $600.00 across 4 transfers, also uncategorized.

## Drivers

The April activity is concentrated in repeatable peer-to-peer outflows where memos consistently include **"Zelle"** and **"Maria"** (for example **Zelle payment to Maria: $140.00** on April 6 and **Zelle to Maria — thank you: $200.00** on April 19). Nothing in the descriptions suggests merchant card spend or dining; the pattern looks like personal transfers rather than dining or shopping.

## Next steps

1. Set a categorization rule for **Zelle to Maria** (past and future).

</RATIONALIZE_1>

<PROPOSAL>

# Proposal

## Proposed next steps

1. **Create** a categorization rule so merchant substring “zelle to maria” maps to `meals_dining_out` for past and future.

## Open items (not addressed)

1. None.

</PROPOSAL>

<PROPOSAL_TOOL_CALLS>

# Round 1

## Invoked tools

1. **`propose_create_categorization_rule`**

```json
{
  "rule": {
    "name_sub_eq": "zelle to maria"
  },
  "ai_category_id": 2,
  "scope": "future_and_past",
  "rationale": "Encode the Zelle-to-Maria payee pattern as dining out."
}
```

</PROPOSAL_TOOL_CALLS>
""",
    "output": '{"score": 3, "notes": "Rationalize asks for a Zelle-to-Maria rule without a category and Drivers describe generic P2P transfers, so assigning `meals_dining_out` contradicts the evidence."}',
  },
  {
    "name": "zelle_maria_dinner_rule_dining_out_correct",
    "batch": 2,
    "input": """<RATIONALIZE_1>

# Rationalize What

Explain: Several Zelle payments to Maria are sitting in uncategorized spend this month. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

- **Current Month (Apr 1–30, 2026) — uncategorized Zelle to Maria:** $840.00 across 6 posted transfers in `miscellaneous`.
- **Prior Month (Mar 1–31, 2026) — same payee pattern:** $600.00 across 4 transfers, also uncategorized.

## Drivers

April memos mix purposes, but several lines carry an explicit dining cue: **Zelle to Maria: Dinner $85.00** on April 8 and **Zelle payment to Maria — dinner split $62.00** on April 21 read like shared-meal reimbursements. Other lines such as **Zelle to Maria — thank you: $120.00** on April 22 remain general P2P transfers with no dining signal.

## Next steps

1. Set a categorization rule for **Zelle to Maria: Dinner** (past and future).

</RATIONALIZE_1>

<PROPOSAL>

# Proposal

## Proposed next steps

1. **Create** a categorization rule so merchant substring “zelle to maria: dinner” maps to `meals_dining_out` for past and future.

## Open items (not addressed)

1. **Review** remaining generic **Zelle to Maria** lines that lack a dinner memo before applying any broader payee rule.

</PROPOSAL>

<PROPOSAL_TOOL_CALLS>

# Round 1

## Invoked tools

1. **`propose_create_categorization_rule`**

```json
{
  "rule": {
    "name_sub_eq": "zelle to maria: dinner"
  },
  "ai_category_id": 2,
  "scope": "future_and_past",
  "rationale": "Dinner memo lines are clearly shared-meal reimbursements."
}
```

</PROPOSAL_TOOL_CALLS>
""",
    "output": '{"score": 5, "notes": "The Dinner-scoped rule and `meals_dining_out` target match the stated payee purpose and the matching categorization tool."}',
  },
  {
    "name": "walmart_rule_no_category_groceries_acceptable",
    "batch": 3,
    "input": """<RATIONALIZE_1>

# Rationalize What

Explain: Groceries looks low because Walmart runs are still uncategorized this month. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

- **Current Month (Apr 1–30, 2026) — `meals_groceries`:** $318.00.
- **Current Month — uncategorized Walmart-tagged spend:** $186.50 across 7 posted lines flagged on import.
- **Prior Month (Mar 1–31, 2026) — Walmart eventually coded to groceries after review:** $142.00 across 5 lines.

## Drivers

April's uncategorized Walmart charges mostly read like food and pantry purchases (**Walmart Grocery pickup: $64.22** on April 4, **Walmart Supercenter — groceries: $38.15** on April 11, **Walmart.com grocery delivery: $29.88** on April 19). A minority of lines lack item detail, but the dominant signal is grocery-like; Walmart is ~70% likely to be groceries in this household's history.

## Next steps

1. Set a categorization rule for **Walmart** (past and future).

</RATIONALIZE_1>

<PROPOSAL>

# Proposal

## Proposed next steps

1. **Create** a categorization rule so merchant substring “walmart” maps to `meals_groceries` for past and future.

## Open items (not addressed)

1. None.

</PROPOSAL>

<PROPOSAL_TOOL_CALLS>

# Round 1

## Invoked tools

1. **`propose_create_categorization_rule`**

```json
{
  "rule": {
    "name_sub_eq": "walmart"
  },
  "ai_category_id": 4,
  "scope": "future_and_past",
  "rationale": "Walmart memos are predominantly grocery-like in this household."
}
```

</PROPOSAL_TOOL_CALLS>
""",
    "output": '{"score": 5, "notes": "Walmart is a mixed retailer but `meals_groceries` is an acceptable default given the grocery-heavy memo profile Drivers describe."}',
  },
  {
    "name": "duplicate_proposed_steps_non_factor",
    "batch": 4,
    "input": """<RATIONALIZE_1>

# Rationalize What

Explain: Groceries is significantly down this month at $120. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

- Groceries: $120 this month (Apr 1–30, 2026).

## Drivers

Several Costco transactions appear miscategorized as `miscellaneous`.

## Next steps

Create a categorization rule so Costco is always `meals_groceries`.

</RATIONALIZE_1>

<PROPOSAL>

# Proposal

## Proposed next steps

1. **Add** a categorization rule so merchant substring “costco” maps to `meals_groceries` for past and future.
2. **Add** a categorization rule so merchant substring “costco” maps to `meals_groceries` for past and future.

## Open items (not addressed)

1. **Review** whether any non-Costco groceries are still landing in `miscellaneous`.

</PROPOSAL>

<PROPOSAL_TOOL_CALLS>

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

</PROPOSAL_TOOL_CALLS>
""",
    "output": '{"score": 5, "notes": "Grounding, category intent, and the rule tool match rationalize; repeating the same proposed step is out of scope for accuracy."}',
  },
]


def _build_output_schema(_types: Any) -> Any:
  return _types.Schema(
    type=_types.Type.OBJECT,
    required=["score", "notes"],
    properties={
      "score": _types.Schema(
        type=_types.Type.INTEGER,
        description="Accuracy 1–5 per rubric (5 = no meaningful gap).",
      ),
      "notes": _types.Schema(
        type=_types.Type.STRING,
        description=(
          "Semicolon-separated list of every distinct accuracy issue found, regardless of severity; "
          "cite slug tokens in backticks when relevant. Score 5: one brief affirmation only. Never omit "
          "a rubric violation to keep notes short. Never mention duplicates, completeness, or missing tools."
        ),
      ),
    },
  )


SYSTEM_PROMPT = """Grade **accuracy** only. Return `{score, notes}`.

**Input:** `<RATIONALIZE_N>` (figures, drivers, next steps) → `<PROPOSAL>` (`## Proposed next steps`, `## Open items`) → `<PROPOSAL_TOOL_CALLS>` (`# Round N`, `## Invoked tools`). Grade only visible text.

**Notes:** Audit Accuracy items 1–9 below; list **every** distinct violation in `notes`, regardless of severity or whether it alone sets the score. Use semicolon-separated clauses; cite slug tokens in backticks. Score **5** → one brief affirmation only. Never omit a rubric violation to keep notes short—even when other violations already justify a lower score.

**Next steps — binding when** a direct Penny/user instruction, not optional/hypothetical/illustrative (e.g., eg., consider, if user confirms) and not gated on confirmation or order ((1) before (2)). Automating an example or step (2) before prerequisite (1) → typically **4**.

**Grounding:** All `<RATIONALIZE_N>` bodies are combined evidence. Invented merchants, amounts, dates, or transaction ids → **1**.

**Slugs vs IDs:** Slugs in `<PROPOSAL>` and tool `new_category` args must be Category List tokens exactly. Numeric `ai_category_id` in tool calls is always valid—**never** list, penalize, or suggest replacing it with a slug.

**Category choice:** When next steps name a payee/merchant rule but no category, the proposed category must follow Drivers/Figures—not stereotypes. Contradicting evidence → **3**. Explicit purpose in the rule scope or a dominant memo signal (including mixed merchants) → **5** when otherwise aligned. Parent rollups and leaves are both valid—pick the level Drivers support (see item 9).

**Out of scope (never lower score, never list in notes):** uncovered next steps; **missing** tool calls for proposed steps; numeric `ai_category_id` in tool calls; Proposed/Open sectioning unless factual contradiction; **duplicate identical proposed bullets—ignore entirely, score 5 if otherwise sound**.

**Accuracy**
1. Grounding — no contradictions vs any `<RATIONALIZE_N>`.
2. Penny fit — steps match mandatory (non-example) drivers and next steps.
3. Tools — **always check** when calls exist. Families/args must support Proposed steps: `propose_create_categorization_rule` for rule-creation steps; `propose_recategorize_transactions` for one-off recategorize. A proposed "create rule" step with only recategorize tools → flag tool-family mismatch. Wrong family → typically **3–4**. Also flag item 6 separately when that "create rule" step targets a one-off P2P refund/gift from a named counterparty. Invalid slug in tool args → flag per slug rules (separate clause per slug).
4. Invalid proposal slug — one non-list token → cap **4**; several → **3**; never 1–2 for slug-only.
5. Example/gated/ordered violations → typically **4**.
6. Rule scope — **always check** when proposed-step text requests a payee/merchant/P2P rule (e.g. "create rule", "categorization rule", payee + past/future)—even if tools only recategorize one ID. Overbroad when Drivers cite one transaction, a unique memo (gift, refund, reimbursement), or no repeatable payee pattern. Never propose a payee-wide categorization rule for a single named P2P/Venmo/Zelle counterparty on one refund, gift, or reimbursement line—one-off `propose_recategorize_transactions` is the right scope. Flag separately from tool-family errors (item 3). → typically **4**.
7. Over-commitment — **always check** when rationalize next steps use ensure, verify, consider, or either/or. Proposal mandates a settled rule, category, or action → flag. Typically **4**.
8. Internal contradiction — **always check** when multiple proposed steps target the same merchant/transaction. Incompatible category or action → flag. Typically **3–4**.
9. Parent vs leaf — **always check** when taxonomy or Category List exposes leaves under a parent. Both parent rollups and leaves are valid tokens—judge which level fits Drivers. Flag parent rollup when Drivers/name identify a specific leaf (e.g. connectivity charge → `bills_connectivity`, not `bills`). Flag a specific leaf when Drivers show mixed or unknown subtypes and no memo justifies that leaf—parent rollup may be the better fit. Generic opaque memos spanning multiple leaves (e.g. "Bill Payment") → `bills` parent is acceptable when no leaf is evidenced; flag a forced leaf without support. → typically **4**.

**Category List** (exact tokens; no paraphrase or invented compounds; parents roll up leaves—both levels are valid):
`meals`, `meals_dining_out`, `meals_delivered_food`, `meals_groceries`, `leisure`, `leisure_entertainment`, `leisure_travel`, `shopping`, `shopping_pets`, `bills`, `bills_connectivity`, `bills_insurance`, `bills_tax`, `bills_service_fees`, `shelter`, `shelter_home`, `shelter_utilities`, `shelter_upkeep`, `education`, `education_kids_activities`, `education_tuition`, `shopping_clothing`, `shopping_gadgets`, `shopping_kids`, `transportation`, `transportation_car`, `transportation_public`, `health`, `health_medical_pharmacy`, `health_gym_wellness`, `health_personal_care`, `donations_gifts`, `miscellaneous`, `income`, `income_salary`, `income_sidegig`, `income_business`, `income_interest`, `transfers`

**Calibration:** 5 none · 4 minor · 3 clear fixable · 2 several · 1 failed. Sole single invalid slug → **4**, not 3. `score` reflects the worst applicable band across all listed issues.
"""


class ProposeNextStepsAccuracyCheckerOptimizer:
  def __init__(
    self,
    model_name: str = "gemini-flash-lite-latest",
    *,
    max_output_tokens: int = 512,
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
  parser.add_argument("--max-output-tokens", type=int, default=512)
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

