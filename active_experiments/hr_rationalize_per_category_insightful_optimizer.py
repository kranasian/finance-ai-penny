"""
Rationalize rubric checker optimizer: Insightful.

Use this to iterate on the system_prompt that will be stored in `penny_templates`
for the checker template `Chk:RationalizePerCategoryInsightful`.

Run from `finance-ai-penny` repo root:

  python3 active_experiments/hr_rationalize_per_category_insightful_optimizer.py --test all
  python3 active_experiments/hr_rationalize_per_category_insightful_optimizer.py --test high_insight_corrects_narrative_and_prioritizes
  python3 active_experiments/hr_rationalize_per_category_insightful_optimizer.py --test all --model gemini-flash-lite-latest

**Recommended minimal generation settings** (validated `--test all`; scores match `ideal_response` **5 / 1**):

- **model:** `gemini-flash-lite-latest`
- **temperature:** `0` · **top_p:** `0.95`
- **thinking_budget:** `0`
- **max_output_tokens:** `128` (`256` if truncation)
- **response:** `application/json` + **response_schema** for `{score, notes}`

**Input:** a single markdown **`str`**—`# Rationalize What` then `# Rationalize Response` (same shape as `ai_agent_outcomes.agent_outcome` / comprehensive optimizer).
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
from google import genai
from google.genai import types

if load_dotenv is not None:
  load_dotenv()

_TEST_SEPARATOR = "=" * 72
_SECTION_RULE = "-" * 72


def _print_section_banner(title: str) -> None:
  print(f"\n{_SECTION_RULE}\n{title}\n{_SECTION_RULE}\n")


OUTPUT_SCHEMA = types.Schema(
  type=types.Type.OBJECT,
  required=["score", "notes"],
  properties={
    "score": types.Schema(
      type=types.Type.INTEGER,
      description="Integer 1-5 rubric score.",
    ),
    "notes": types.Schema(
      type=types.Type.STRING,
      description="One short sentence.",
    ),
  },
)


SYSTEM_PROMPT = """Grade **insightful** only.

Judge whether **Rationalize Response** **interprets** the situation versus **Rationalize What**—prioritizing what actually drove the change and **why it matters**—not merely echoing figures or the headline story.

**Input:** markdown: `# Rationalize What` then `# Rationalize Response`. Judge **only** this text.

Reward reasoning grounded in **actual amounts and drivers written** in the response (ignore hypothetical forecast tables).

**Scores (integer 1–5)**
- **5** — Adds clear interpretation: reframes or **corrects** the narrative when data warrants it, identifies the **main swing factor**, and conveys **why it matters** to the user.
- **4** — Strong interpretation; one gap (e.g. thinner “why care” or prioritization).
- **3** — Some interpretation but still mostly descriptive; weak prioritization.
- **2** — Lightly interpretive or uneven; hard to see what mattered most.
- **1** — **Restates** figures or the prompt’s framing with little/no interpretation or prioritization.

**`notes`:** One sentence naming strengths or gaps—e.g. premise vs data, swing factor, takeaway—so the score is auditable.

Return **only** the JSON object matching the schema (`score`, `notes`).
"""


TEST_CASES: list[dict[str, Any]] = [
  {
    "name": "high_insight_corrects_narrative_and_prioritizes",
    "batch": 1,
    "output": '{"score": 5, "notes": "Corrects the premise, prioritizes the swing factor, and explains why it matters."}',
    "input": """# Rationalize What

Explain: Home is slightly up this month at $2850. Utilities is significantly down this month at $324. Shelter is thus slightly up this month to $3212. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

* **Shelter:** $3,212.30 (Apr 1–30, 2026) vs $3,339.06 (Mar 1–31, 2026).
* **Shelter Home:** $2,850.00 (Apr 1–30, 2026) vs $2,850.00 (Mar 1–31, 2026).
* **Utilities:** $324.48 (Apr 1–30, 2026) vs $402.84 (Mar 1–31, 2026).

## Drivers

Your summary says shelter is “slightly up,” but the tool-backed Shelter is actually **down** vs March ($3,212.30 vs $3,339.06). The important thing isn’t the $2,850 home line (it’s flat); the swing factor is utilities—Dominion Energy dropped the most this month.

## Next steps

1. Keep shelter_home as a fixed $2,850 budget line; focus optimization on utilities/maintenance variability.
2. Add a rule for Dominion Energy → shelter_utilities so the main driver stays consistently tagged.
""",
  },
  {
    "name": "low_insight_restates_without_interpretation",
    "batch": 1,
    "output": '{"score": 1, "notes": "Mostly restates figures without interpretation or prioritization."}',
    "input": """# Rationalize What

Explain: Home is slightly up this month at $2850. Utilities is significantly down this month at $324. Shelter is thus slightly up this month to $3212. (2026-04-01 to 2026-04-30)

# Rationalize Response

## Figures

* Home is $2,850.
* Utilities is $324.

## Drivers

Home is up and utilities is down. That’s why shelter changed.

## Next steps

1. Keep an eye on it.
""",
  },
]


class CheckerOptimizer:
  def __init__(
    self,
    model_name: str = "gemini-flash-lite-latest",
    *,
    max_output_tokens: int = 128,
    thinking_budget: int = 0,
  ):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set.")
    self.client = genai.Client(api_key=api_key)
    self.model_name = model_name
    self.temperature = 0.0
    self.top_p = 0.95
    self.max_output_tokens = max_output_tokens
    self.thinking_budget = thinking_budget
    self.system_prompt = SYSTEM_PROMPT
    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    ]

  def grade(self, agent_outcome: str) -> Dict[str, Any]:
    user_msg = (agent_outcome or "").strip()
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_msg)])]
    cfg = types.GenerateContentConfig(
      temperature=self.temperature,
      top_p=self.top_p,
      max_output_tokens=self.max_output_tokens,
      safety_settings=self.safety_settings,
      system_instruction=[types.Part.from_text(text=self.system_prompt)],
      thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget, include_thoughts=True),
      response_mime_type="application/json",
      response_schema=OUTPUT_SCHEMA,
    )
    out = self.client.models.generate_content(model=self.model_name, contents=contents, config=cfg)
    text = (out.text or "").strip()
    try:
      return json.loads(text)
    except Exception:
      s = text[text.find("{"): text.rfind("}") + 1] if ("{" in text and "}" in text) else "{}"
      return json.loads(s)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--test", type=str, default=None, help="Test name, index, or 'all'.")
  parser.add_argument("--batch", type=int, default=None, help="Run all tests in batch N.")
  parser.add_argument("--model", type=str, default="gemini-flash-lite-latest")
  parser.add_argument("--max-output-tokens", type=int, default=128)
  parser.add_argument("--thinking-budget", type=int, default=0)
  args = parser.parse_args()

  if args.test is None and args.batch is None:
    print("Available test cases:")
    for i, tc in enumerate(TEST_CASES):
      batch = tc.get("batch")
      batch_s = str(batch) if isinstance(batch, int) else "—"
      print(f"  {i}: {tc.get('name')} (batch {batch_s})")
    return

  opt = CheckerOptimizer(
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
    if tc.get("output") is not None:
      _print_section_banner("# Expected Output")
      print(tc["output"])

  print(f"\n{_TEST_SEPARATOR}\n")
  print(f"# Total tests: {len(cases)}\n")


if __name__ == "__main__":
  main()

