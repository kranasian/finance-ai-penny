from typing import Tuple


def add_to_memory(
    memory_request: str,
    input_info: str = None
) -> Tuple[bool, str]:
    """
    Add note-worthy information to memory.
    
    This function records only note-worthy information that cannot be pulled from 
    transactions, accounts, spending, or forecasts. Examples include:
    - User preferences (e.g., "I prefer to save for emergencies first")
    - Personal facts (e.g., "I'm planning to move to a new city next year")
    - Goals and intentions (e.g., "I want to retire early")
    - Important context (e.g., "I'm self-employed and income varies")
    
    DO NOT record:
    - Account balances (can be pulled from accounts)
    - Transaction details (can be pulled from transactions)
    - Spending patterns (can be pulled from spending data)
    - Forecasts (can be pulled from forecasts)
    - Specific amounts or dates that are in transactions
    
    Args:
        memory_request: Description of the note-worthy information to record.
                       Should exclude any details that can be retrieved from 
                       transactions, accounts, spending, or forecasts.
        input_info: Optional input from another skill function
        
    Returns:
        Tuple[bool, str]: (success, output_info)
        - success: True if memory was added successfully, False if an error occurred
        - output_info: Confirmation message or error message
    """
    # This is a dummy implementation that returns sample data
    # In a real implementation, this would save to a database
    
    if not memory_request or not memory_request.strip():
        return False, "Memory request cannot be empty. Please provide note-worthy information to record."
    
    # Extract the key information from memory_request
    # In a real implementation, this would parse and store the information
    confirmation_message = f"Successfully recorded to memory: {memory_request.strip()}"
    
    return True, confirmation_message
