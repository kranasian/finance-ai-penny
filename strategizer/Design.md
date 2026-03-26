# Strategizer Framework

The Strategizer is a proactive, agentic framework designed to continuously monitor and improve a user's financial setup without waiting for explicit user prompts. It utilizes the existing `tool_funcs` (like data lookup and categorization updates) through a flexible, task-driven engine.

## Core Concepts

### 1. The Task List
Instead of hardcoded scripts, the Strategizer operates on a list of high-level objectives called Tasks. A Task represents a single goal, e.g., "Check if the user has a known salary, and if not, try to find it."

### 2. Task & Outcome Models (`task.py`)
- **Task**: Holds the description of the objective, its current status (Pending, In Progress, Completed, Partially Completed, Failed), and a history of previous attempts.
- **Outcome**: For every attempted step to solve a Task, an Outcome is recorded. It contains a summary of what the LLM did, the data it found, and whether the step brought the task closer to completion.

### 3. The Engine (`engine.py`)
The Engine is the central orchestrator that iterates through the Task List.

**The Execution flow:**
1. **Pull Task**: The engine picks up an active, pending, or in-progress Task.
2. **Contextualize**: It builds a prompt containing the Task's original objective along with the history of all previous `Outcome`s for this Task. This ensures the LLM remembers what it tried previously and what the results were.
3. **Generate Code**: The LLM writes a Python script using standard `tool_funcs` (`lookup_user_accounts...`, etc.) to take the next logical step towards solving the task.
4. **Execute**: The engine safely runs the generated Python code in a sandbox.
5. **Analyze & Reflect**: The LLM output, function returns, and execution logs are captured. The engine then reflects on this new data to generate an `Outcome` summary.
6. **Update State**: The engine updates the `Task` with the new `Outcome` and determines if the task is complete, blocked, or requires another iteration.
7. **Loop**: It repeats this loop for the current task until no further progress can be made, then moves to the next task in the list.

## Benefits
- **Resilience**: Because the engine self-reflects, if an initial lookup returns too much data, the next iteration can refine the search based on the previous failure.
- **Modularity**: By relying on the `tool_funcs` code generation, the Strategizer inherits all capabilities of the base `AgentPlanner` without redefining logic.
- **Traceability**: Every step is logged as an Outcome on the Task, making it easy to audit why the Strategizer made a specific change.
