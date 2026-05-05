from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from past_experiments.check_before_recategorize_optimizer import BATCHES
from past_experiments.check_before_recategorize_optimizer import TEST_CASES as SOURCE_TEST_CASES


load_dotenv()


SYSTEM_PROMPT = """You are an AI evaluator for check_before_recategorize outputs.

## Task
Review `REVIEW_NEEDED` against `EVAL_INPUT`.
Return strict JSON only:
{"good_copy": boolean, "info_correct": boolean, "eval_text": string}

## Definitions
- good_copy: true only when REVIEW_NEEDED follows required output format and style:
  - Has key: rules_satisfied (boolean)
  - Has exactly one explanation key: notes OR rationale (single-line string)
  - Explanation length is 8-140 chars
  - Explanation is factual and consistent with rules_satisfied
- info_correct: true only when REVIEW_NEEDED semantically matches IDEAL_OUTPUT verdict and reason quality.
- eval_text: required if either boolean is false. Use bullet lines; concise, actionable.
  - Start each bullet with a fix action.
  - Mention the concrete mismatch from EVAL_INPUT and REVIEW_NEEDED.

Return JSON only, no markdown."""


def _build_checker_test_cases() -> list[dict]:
  test_cases: list[dict] = [
    {
      "name": "approximately_20_with_outlier",
      "eval_input": """# Categorize Request
Identify any transaction approximately $20 and mark it as Income Business rather than Bills Service Fees.
# Transactions
- $19.95 Monthly Server Fee on 2026-05-10.
- $20.05 Service Maintenance on 2026-05-12.
- $20.00 Processing Fee on 2026-05-14.
- $120.00 Annual Membership on 2026-05-15.
- $20.10 Cloud Storage Bill on 2026-05-18.""",
      "review_needed": {
        "rationale": "The request requires identifying transactions approximately $20; however, the provided transactions are all Bills or Service Fees, not Income Business.",
        "rules_satisfied": False,
      },
      "ideal_output": {
        "good_copy": True,
        "info_correct": False,
        "eval_text": "$120.00 Annual Membership is too far from the approximately $20 request.",
      },
    }
  ]

  for source_case in SOURCE_TEST_CASES:
    expected_verify_output = source_case["output"]
    test_cases.append(
      {
        "name": f"{source_case['name']}_golden",
        "eval_input": source_case["input"],
        "review_needed": dict(expected_verify_output),
        "ideal_output": {
          "good_copy": True,
          "info_correct": True,
          "eval_text": "",
        },
      }
    )

  if SOURCE_TEST_CASES:
    first_source = SOURCE_TEST_CASES[0]
    wrong_review = dict(first_source["output"])
    wrong_review["rules_satisfied"] = not wrong_review["rules_satisfied"]
    test_cases.append(
      {
        "name": "synthetic_wrong_verdict_against_input",
        "eval_input": first_source["input"],
        "review_needed": wrong_review,
        "ideal_output": {
          "good_copy": True,
          "info_correct": False,
          "eval_text": "rules_satisfied verdict conflicts with the request and listed transactions.",
        },
      }
    )

    malformed_review = {
      "ok": True,
      "message": "everything looks right",
    }
    test_cases.append(
      {
        "name": "synthetic_invalid_shape",
        "eval_input": first_source["input"],
        "review_needed": malformed_review,
        "ideal_output": {
          "good_copy": False,
          "info_correct": False,
          "eval_text": "Missing required keys rules_satisfied and notes or rationale.",
        },
      }
    )
  return test_cases


TEST_CASES = _build_checker_test_cases()


class CheckCheckBeforeRecategorizeOptimizer:
  """Checks check_before_recategorize outputs against ideal outputs."""

  def __init__(self, model_name: str = "gemini-flash-lite-latest"):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
      raise ValueError("GEMINI_API_KEY environment variable is not set.")
    self.client = genai.Client(api_key=api_key)
    self.model_name = model_name or "gemini-flash-lite-latest"
    self.thinking_budget = 0
    self.temperature = 0.0
    self.top_p = 1.0
    self.max_output_tokens = 128
    self.safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    ]
    self.system_prompt = SYSTEM_PROMPT

  def generate_response(self, eval_input: str, review_needed: dict) -> dict:
    request_text_str = f"""<EVAL_INPUT>

{eval_input}

</EVAL_INPUT>

<REVIEW_NEEDED>

{json.dumps(review_needed, indent=2)}

</REVIEW_NEEDED>

Output:"""
    print(request_text_str)
    print(f"\n{'=' * 80}\n")

    request_text = types.Part.from_text(text=request_text_str)
    contents = [types.Content(role="user", parts=[request_text])]
    config = types.GenerateContentConfig(
      temperature=self.temperature,
      top_p=self.top_p,
      max_output_tokens=self.max_output_tokens,
      safety_settings=self.safety_settings,
      system_instruction=[types.Part.from_text(text=self.system_prompt)],
      thinking_config=types.ThinkingConfig(
        thinking_budget=self.thinking_budget,
        include_thoughts=True,
      ),
    )

    output_text = ""
    thought_summary = ""
    for chunk in self.client.models.generate_content_stream(
      model=self.model_name,
      contents=contents,
      config=config,
    ):
      if chunk.text is not None:
        output_text += chunk.text
      if hasattr(chunk, "candidates") and chunk.candidates:
        for candidate in chunk.candidates:
          if hasattr(candidate, "content") and candidate.content:
            if hasattr(candidate.content, "parts") and candidate.content.parts:
              for part in candidate.content.parts:
                if hasattr(part, "thought") and part.thought:
                  if hasattr(part, "text") and part.text:
                    thought_summary += part.text

    if not output_text or not output_text.strip():
      raise ValueError("Empty response from model. Check API key and model availability.")

    response_json = self._parse_json_response(output_text)
    clean_thought_summary = thought_summary.strip()
    if clean_thought_summary:
      print("-" * 80)
      print("THOUGHT SUMMARY:")
      print(clean_thought_summary)
      print("-" * 80)

    return {
      "response": response_json,
      "thought_summary": clean_thought_summary,
    }

  @staticmethod
  def _parse_json_response(output_text: str) -> dict:
    text = output_text.strip()
    if "```json" in text:
      start = text.find("```json") + 7
      end = text.find("```", start)
      if end != -1:
        text = text[start:end].strip()
    elif "```" in text:
      start = text.find("```") + 3
      end = text.find("```", start)
      if end != -1:
        text = text[start:end].strip()

    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
      text = text[start:end]
    return json.loads(text)


def run_test_case(test_case: dict, checker: CheckCheckBeforeRecategorizeOptimizer | None = None):
  if checker is None:
    checker = CheckCheckBeforeRecategorizeOptimizer()

  print(f"\n{'=' * 80}")
  print(f"Running test: {test_case['name']}")
  print(f"{'=' * 80}")
  print("IDEAL OUTPUT:")
  print(json.dumps(test_case["ideal_output"], indent=2))

  try:
    result = checker.generate_response(
      eval_input=test_case["eval_input"],
      review_needed=test_case["review_needed"],
    )
    print("CHECKER OUTPUT:")
    print(json.dumps(result["response"], indent=2))
    print("OUTPUT MATCHES IDEAL:")
    print("YES" if _matches_ideal_output(result["response"], test_case["ideal_output"]) else "NO")
    print(f"{'=' * 80}")
    return result
  except Exception as error:
    print(f"ERROR: {type(error).__name__}: {error}")
    print(f"{'=' * 80}")
    return None


def run_tests(test_names: list[str] | None = None):
  checker = CheckCheckBeforeRecategorizeOptimizer()
  if test_names is None:
    selected = TEST_CASES
  else:
    selected = [tc for tc in TEST_CASES if tc["name"] in test_names]

  for test_case in selected:
    run_test_case(test_case, checker=checker)


def _matches_ideal_output(actual: dict, ideal: dict) -> bool:
  actual_good_copy = actual.get("good_copy")
  actual_info_correct = actual.get("info_correct")
  actual_eval_text = actual.get("eval_text", "")

  if actual_good_copy != ideal.get("good_copy"):
    return False
  if actual_info_correct != ideal.get("info_correct"):
    return False
  if not isinstance(actual_eval_text, str):
    return False
  if bool(actual_eval_text.strip()) != bool(ideal.get("eval_text", "").strip()):
    return False
  return True


def main(test: str | None = None, batch: int | None = None):
  if batch is not None:
    if batch not in BATCHES:
      print(f"Invalid batch number: {batch}. Available batches: {sorted(BATCHES.keys())}")
      print("\nBatch descriptions:")
      for batch_number, info in BATCHES.items():
        names = [SOURCE_TEST_CASES[idx]["name"] for idx in info["tests"]]
        print(f"  Batch {batch_number}: {info['name']} — {', '.join(names)}")
      return
    test_names = [SOURCE_TEST_CASES[idx]["name"] for idx in BATCHES[batch]["tests"]]
    print(f"\nRunning batch {batch}: {BATCHES[batch]['name']}\n")
    run_tests(test_names=test_names)
    return

  if test:
    selected = next((tc for tc in TEST_CASES if tc["name"] == test), None)
    if selected is None and test.isdigit():
      index = int(test)
      selected = TEST_CASES[index] if 0 <= index < len(TEST_CASES) else None
    if selected is None:
      print(f"Test '{test}' not found.")
      return
    run_test_case(selected)
    return

  print("Available checker test cases (with ideal outputs):")
  for i, test_case in enumerate(TEST_CASES):
    print(f"  {i}: {test_case['name']}")


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(
    description="Check check_before_recategorize outputs against ideal outputs."
  )
  parser.add_argument(
    "--test",
    type=str,
    default=None,
    help="Test name or index to run.",
  )
  parser.add_argument(
    "--batch",
    type=int,
    nargs="?",
    const=1,
    default=None,
    metavar="N",
    help="Run batch N using source optimizer batch indices.",
  )
  args = parser.parse_args()
  main(test=args.test, batch=args.batch)
