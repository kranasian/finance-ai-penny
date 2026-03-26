from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import uuid

class Outcome(BaseModel):
    """
    Represents the result of a single step or iteration taken towards completing a Task.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # What the engine/LLM decided to do (e.g. generate code to lookup accounts)
    action_taken: str
    
    # The actual code executed
    code_executed: Optional[str] = None
    
    # What was found / result of the code
    execution_result: Optional[str] = None
    
    # Did this step succeed in finding what it wanted?
    success: bool
    
    # LLM synthesized reflection on what this means for the overall task
    reflection: Optional[str] = None


class Task(BaseModel):
    """
    Represents a high level proactive goal for the Strategizer to accomplish.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    
    # The actual instruction, e.g. "Check if there's salary detected..."
    description: str
    
    status: str = Field(default="PENDING", description="Status can be PENDING, IN_PROGRESS, COMPLETED, PARTIALLY_COMPLETED, FAILED")
    
    # History of steps taken
    outcomes: List[Outcome] = Field(default_factory=list)
    
    # The final summary of what happened.
    final_summary: Optional[str] = None
    
    def add_outcome(self, outcome: Outcome):
        self.outcomes.append(outcome)
    
    def is_terminal(self) -> bool:
        return self.status in ["COMPLETED", "PARTIALLY_COMPLETED", "FAILED"]
