"""
Rationalize rubric checker optimizer: Actionable (for Penny / product, not the user).

Use this to iterate on the system_prompt that will be stored in `penny_templates`
for the checker template `Chk:RationalizePerCategoryActionable`.

Run from `finance-ai-penny` repo root:

  python3 active_experiments/rationalize_checker_actionable_optimizer.py --test all
  python3 active_experiments/rationalize_checker_actionable_optimizer.py --test strong_next_steps_penny_capabilities
  python3 active_experiments/rationalize_checker_actionable_optimizer.py --test all --model gemini-flash-lite-latest

**Recommended minimal generation settings** (validated `python3 active_experiments/rationalize_checker_actionable_optimizer.py --test all`; scores **5 / 1** vs `ideal_response`):

- **model:** `gemini-flash-lite-latest`
- **temperature:** `0` · **top_p:** `0.95`
- **thinking_budget:** `0`
- **max_output_tokens:** `128` (raise to `256` if truncated)
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


def _bundle_rationalize_md(user_task: str, assistant_reply: str) -> str:
  """Same shape as `ai_agent_outcomes.agent_outcome`: Rationalize What + Rationalize Response."""
  ut = (user_task or "").strip()
  ar = (assistant_reply or "").strip()
  return f"# Rationalize What\n\n{ut}\n\n# Rationalize Response\n\n{ar}\n"


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


SYSTEM_PROMPT = """Grade **one axis: actionable** for **rationalize_change**. Judge **only** the **## Next steps** list in **Rationalize Response** (use figures/drivers for context).

**What counts (Penny / product / AI):** Concrete actions the system can implement—e.g. **budget caps**, **goals**, **categorization or merchant→category rules**, **tagging**, surfacing in **app views**—**tied to findings** (amounts, merchants, categories named above).

**What does not count on this axis:** Generic **human lifestyle** advice (save energy, shop for plans, “keep an eye on it”) with **no** product configuration—even if sensible offline.

**Input:** markdown `# Rationalize What` … `# Rationalize Response`.

**Scores (integer 1–5)**
- **5** — Next steps are **specific Penny/product actions** grounded in the drivers/figures.
- **4** — Mostly product-facing; one step vague or weakly tied to findings.
- **3** — Mix of product steps and generic life tips.
- **2** — Mostly generic or thin product linkage.
- **1** — Lifestyle-only, missing **## Next steps**, or nothing usable **for the AI/product**.

**`notes`:** One sentence—Penny-style actions vs user-life advice—so the score is auditable.
"""


TEST_CASES: list[dict[str, Any]] = [
  {
    "name": "strong_next_steps_penny_capabilities",
    "batch": 1,
    "ideal_response": {"score": 5, "notes": "Next steps are specific Penny actions tied to findings."},
    "payload": _bundle_rationalize_md(
      "Explain: Home is slightly up this month at $2850. Utilities is significantly down this month at $324. Shelter is thus slightly up this month to $3212.",
      (
        "## Figures\n\n"
        "* **Shelter utilities:** $324.48 (Apr 1–30, 2026) vs $402.84 (Mar 1–31, 2026).\n\n"
        "## Drivers\n\n"
        "Utilities dropped by about $78 vs March ($324.48 vs $402.84). The biggest driver is the **Dominion Energy** charge, which is materially lower this month than last month; the other utility providers moved less.\n\n"
        "## Next steps\n\n"
        "1. Create a **Utilities** budget cap at **$375/month** (3‑month avg ≈ $367) and surface it in the monthly budget view.\n"
        "2. Create a small monthly **goal** (sinking fund) for utilities variability (e.g. $50/month) so swings don’t disrupt other budgets.\n"
        "3. Add a categorization rule: **Dominion Energy → shelter_utilities** (so future bills are consistently tagged).\n"
      ),
    ),
  },
  {
    "name": "weak_next_steps_user_advice_only",
    "batch": 1,
    "ideal_response": {"score": 1, "notes": "Next steps are user-life advice, not Penny actions like budgets/goals/rules."},
    "payload": _bundle_rationalize_md(
      "Explain: Home is slightly up this month at $2850. Utilities is significantly down this month at $324. Shelter is thus slightly up this month to $3212.",
      (
        "## Figures\n\n"
        "* Utilities is down this month.\n\n"
        "## Drivers\n\n"
        "Your shelter total is being pulled down by the utilities line this month; the home/rent component is likely flat, so the swing is utilities.\n\n"
        "## Next steps\n\n"
        "1. Try to use less electricity.\n"
        "2. Shop around for a better plan."
      ),
    ),
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
    user_msg = (
      "Grade **actionable** only. Input is markdown (plain text):\n\n"
      + (agent_outcome or "")
      + "\n\nRespond with the JSON object only."
    )
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
  parser.add_argument("--test", type=str, default="all")
  parser.add_argument("--model", type=str, default="gemini-flash-lite-latest")
  parser.add_argument("--max-output-tokens", type=int, default=128)
  parser.add_argument("--thinking-budget", type=int, default=0)
  args = parser.parse_args()

  opt = CheckerOptimizer(
    model_name=args.model,
    max_output_tokens=args.max_output_tokens,
    thinking_budget=args.thinking_budget,
  )
  if args.test == "all":
    cases = TEST_CASES
  else:
    tc = next((t for t in TEST_CASES if t["name"] == args.test), None)
    if tc is None:
      raise SystemExit(f"Unknown test: {args.test!r}")
    cases = [tc]

  for i, tc in enumerate(cases):
    if i:
      print(f"\n{_TEST_SEPARATOR}\n")
    print(f"# Test: {tc['name']}\n")
    _print_section_banner("# LLM Checker Input")
    print(tc["payload"])
    result = opt.grade(tc["payload"])
    _print_section_banner("# LLM Checker Response")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if "ideal_response" in tc:
      _print_section_banner("# Ideal Response")
      print(json.dumps(tc["ideal_response"], indent=2, ensure_ascii=False))

  print(f"\n{_TEST_SEPARATOR}\n")
  print(f"# Total tests: {len(cases)}\n")


if __name__ == "__main__":
  main()

