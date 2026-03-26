import os
import sys
from typing import List
from database import Database
from .task import Task, Outcome
from google import genai
from google.genai import types
from dotenv import load_dotenv
import sys
import os
import json
from .prompts import STRATEGIZER_SYSTEM_PROMPT, REFLECTION_SYSTEM_PROMPT

# Add the parent directory to the path so we can import the tools
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

load_dotenv()

class StrategizerEngine:
    """
    The main orchestrator that iterates over a task list, uses LLMs to generate python 
    scripts, executes them via tool_funcs, and records outcomes until the task is complete.
    """
    
    def __init__(self, db: Database = None, model_name: str = "gemini-flash-lite-latest"):
        self.db = db
        self.model_name = model_name
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        self.client = genai.Client(api_key=api_key)
        self.thinking_budget = 4096
        
        self.safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
        ]
        
    def execute_task_list(self, user_id: int, tasks: List[Task]):
        """
        Main loop. Iterates through all tasks for a specific user.
        """
        print(f"Starting Strategizer for User {user_id} with {len(tasks)} tasks.")
        for task in tasks:
            if task.is_terminal():
                continue
                
            task.status = "IN_PROGRESS"
            self._process_single_task(user_id, task)
    
    def _process_single_task(self, user_id: int, task: Task):
        """
        Processes a single task, looping and self-reflecting until terminal.
        """
        print(f"--- Processing Task: {task.description} ---")
        
        max_iterations = 5 # Safety limit to prevent infinite loops
        iteration = 0
        
        while not task.is_terminal() and iteration < max_iterations:
            iteration += 1
            print(f"Iteration {iteration} for Task {task.id}")
            
            # Step 1: Create prompt containing the Task description and all previous outcomes
            prompt = self._build_prompt_for_iteration(task)
            
            # Step 2: Generate python code doing the next step
            code, action_taken = self._generate_step(prompt)
            
            # Step 3: Execute code
            # We will use sandbox to execute the generated code safely
            import sandbox
            try:
                success, execution_result_str, captured_output, logs = sandbox.execute_planner_with_tools(code, user_id)
                execution_result = execution_result_str + f"\nLogs:\n{logs}"
            except Exception as e:
                success = False
                execution_result = f"Error executing code: {str(e)}"
            
            # Step 4: Reflect on execution result
            reflection, next_status, final_summary = self._reflect_on_result(task, code, execution_result)
            
            # Record outcome
            outcome = Outcome(
                action_taken=action_taken,
                code_executed=code,
                execution_result=execution_result,
                success=success,
                reflection=reflection
            )
            task.add_outcome(outcome)
            
            # Update state
            if next_status:
                task.status = next_status
                task.final_summary = final_summary
                print(f"Task reached terminal state: {next_status}")
                break
                
        if not task.is_terminal():
            task.status = "FAILED"
            task.final_summary = "Exceeded maximum iterations without reaching a terminal state."
            
        print(f"--- Task Completed: {task.status} ---")
            
    def _build_prompt_for_iteration(self, task: Task) -> str:
        outcomes_text = "None. This is the first attempt."
        if task.outcomes:
            outcomes_text = "\n".join([f"Outcome {i+1}: Action Taken: {o.action_taken}, Code Executed: {o.code_executed}, Result: {o.execution_result}, Reflection: {o.reflection}" for i, o in enumerate(task.outcomes)])
            
        return f"**Task Description**: {task.description}\n\n**Previous Outcomes**:\n\n{outcomes_text}\n\noutput:"
        
    def _generate_step(self, prompt: str) -> tuple[str, str]:
        request_text = types.Part.from_text(text=prompt)
        contents = [types.Content(role="user", parts=[request_text])]
        
        generate_content_config = types.GenerateContentConfig(
            temperature=0.6,
            top_p=0.95,
            max_output_tokens=4096,
            safety_settings=self.safety_settings,
            system_instruction=[types.Part.from_text(text=STRATEGIZER_SYSTEM_PROMPT)],
            thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget, include_thoughts=True)
        )
        
        response_text = ""
        thought_summary = ""
        for chunk in self.client.models.generate_content_stream(model=self.model_name, contents=contents, config=generate_content_config):
            if chunk.text: response_text += chunk.text
            if hasattr(chunk, "candidates") and chunk.candidates:
              for candidate in chunk.candidates:
                if hasattr(candidate, "content") and candidate.content and hasattr(candidate.content, "parts"):
                  for part in candidate.content.parts:
                    if getattr(part, "thought", False) and getattr(part, "text", None):
                      thought_summary += part.text
        
        code_start = response_text.find("```python")
        if code_start != -1:
            code_start += len("```python")
            code_end = response_text.find("```", code_start)
            code = response_text[code_start:code_end].strip() if code_end != -1 else response_text[code_start:].strip()
            action_taken = response_text[:code_start - len("```python")].strip()
        else:
            code = response_text.strip()
            action_taken = thought_summary.strip()
            
        return code, action_taken
        
    def _reflect_on_result(self, task: Task, code_executed: str, execution_result: str) -> tuple[str, str, str]:
        outcomes_text = "None."
        if task.outcomes:
            outcomes_text = "\n".join([f"Outcome {i+1}: Result: {o.execution_result}" for i, o in enumerate(task.outcomes)])
            
        prompt = f"**Task Description**: {task.description}\n\n**Previous Outcomes**:\n{outcomes_text}\n\n**Code Executed**:\n{code_executed}\n\n**Execution Result**:\n{execution_result}\n\noutput:"
        request_text = types.Part.from_text(text=prompt)
        contents = [types.Content(role="user", parts=[request_text])]
        
        generate_content_config = types.GenerateContentConfig(
            temperature=0.2,
            top_p=0.95,
            max_output_tokens=2048,
            safety_settings=self.safety_settings,
            system_instruction=[types.Part.from_text(text=REFLECTION_SYSTEM_PROMPT)],
            thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget, include_thoughts=True),
            response_mime_type="application/json"
        )
        
        response_text = ""
        for chunk in self.client.models.generate_content_stream(model=self.model_name, contents=contents, config=generate_content_config):
            if chunk.text: response_text += chunk.text
            
        try:
            parsed = json.loads(response_text)
            return parsed.get("reflection", "No reflection generated"), parsed.get("next_status", "IN_PROGRESS"), parsed.get("final_summary")
        except:
            return f"Failed to parse reflection XML/JSON. Raw: {response_text}", "FAILED", "Reflection failed."
