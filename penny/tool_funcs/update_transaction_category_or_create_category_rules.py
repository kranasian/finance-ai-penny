from typing import Tuple, Dict


def update_single_transaction_category(
    transaction_id: int,
    new_category: str
) -> Tuple[bool, str]:
    """
    Update a single transaction's category.
    
    This is a dummy implementation that returns a success message.
    
    Args:
        transaction_id: Transaction ID to update
        new_category: New category name to assign
        
    Returns:
        Tuple[bool, str]: (success, output_info)
    """
    # Dummy implementation - returns success message
    output = f"Successfully updated transaction {transaction_id} to category '{new_category}'"
    return True, output


def create_category_rules(
    rules_dict: Dict,
    new_category: str
) -> Tuple[bool, str]:
    """
    Create category rules to automatically update future transactions matching the rules.
    
    This is a dummy implementation that returns a success message.
    
    Args:
        rules_dict: Dictionary of matching rules (e.g., {'name_contains': 'costco', 'amount_less_than_or_equal_to': 50})
        new_category: Category name to assign to matching transactions
        
    Returns:
        Tuple[bool, str]: (success, output_info)
    """
    # Dummy implementation - returns success message
    rules_summary = ", ".join([f"{k}={v}" for k, v in rules_dict.items()])
    output = f"Successfully created category rule: {rules_summary} -> '{new_category}'"
    return True, output


def update_multiple_transaction_categories_matching_rules(
    rules_dict: Dict,
    new_category: str
) -> Tuple[bool, str]:
    """
    Update multiple existing transactions that match the given rules.
    
    This is a dummy implementation that returns a success message.
    
    Args:
        rules_dict: Dictionary of matching rules (e.g., {'name_contains': 'costco', 'date_greater_than_or_equal_to': '2025-01-01'})
        new_category: Category name to assign to matching transactions
        
    Returns:
        Tuple[bool, str]: (success, output_info)
    """
    # Dummy implementation - returns success message
    rules_summary = ", ".join([f"{k}={v}" for k, v in rules_dict.items()])
    output = f"Successfully updated transactions matching rules ({rules_summary}) to category '{new_category}'"
    return True, output


def update_transaction_category_or_create_category_rules(
    categorize_request: str, 
    input_info: str = None
) -> Tuple[bool, str]:
    """
    Update transaction category or create category rules.
    
    This is a dummy implementation that returns a success message.
    
    Args:
        categorize_request: A description of the category rule that needs to be created, 
                           or the description of the transaction that needs to be recategorized. 
                           This can be a single transaction, or a group of transactions with a criteria.
        input_info: Optional input from another skill function
        
    Returns:
        Tuple[bool, str]: (success, output_info)
    """
    # Dummy implementation - returns success message
    output = f"Successfully processed categorization request: {categorize_request}"
    if input_info:
        output += f"\n\nUsing input information: {input_info[:100]}..."
    
    return True, output

