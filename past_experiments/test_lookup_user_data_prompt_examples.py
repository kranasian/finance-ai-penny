import os
import re
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sandbox
from past_experiments.lookup_user_data_optimizer_v4 import SYSTEM_PROMPT
from past_experiments.lookup_user_data_optimizer_v4 import _get_heavy_data_user_id


def _extract_example_code_blocks(prompt: str) -> List[str]:
  examples_start = prompt.find("<EXAMPLES>")
  if examples_start == -1:
    return []
  examples_text = prompt[examples_start:]
  return re.findall(r"```python\s*(def process_input\(\):[\s\S]*?)```", examples_text)


def run_example_code_validation():
  user_id = _get_heavy_data_user_id()
  code_blocks = _extract_example_code_blocks(SYSTEM_PROMPT)
  if not code_blocks:
    raise RuntimeError("No example code blocks found in SYSTEM_PROMPT.")

  failures = []
  for index, code in enumerate(code_blocks, start=1):
    try:
      success, output_string, logs, goals_list = sandbox.execute_agent_with_tools(code, user_id)
      if not success:
        failures.append(f"Example {index} failed: {output_string}")
    except Exception as error:
      failures.append(f"Example {index} raised exception: {error}")

  if failures:
    raise AssertionError("\n".join(failures))

  print(f"Validated {len(code_blocks)} example code blocks successfully.")


if __name__ == "__main__":
  run_example_code_validation()
