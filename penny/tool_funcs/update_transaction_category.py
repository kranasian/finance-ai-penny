from penny.tools.utils import get_params, to_category_id
from penny.tool_funcs.sandbox_logging import log
import pandas as pd
import traceback


def update_transaction_category(df: pd.DataFrame, **kwargs) -> None:
    """Save updated transaction categories to the database.
    
    Args:
        df: DataFrame containing only transactions that were updated.
            Must include columns: transaction_id, category, account_id
        **kwargs: Additional context including database connections (l, dl, db_mconn, user_id, etc.)
    
    Returns:
        None
    """
    try:
        l, dl, debug_arr, db_mconn, db_sconn, templates_df, user_id, _ = get_params(kwargs)
    except (ValueError, KeyError, TypeError):
        # Database connections not available (e.g., called without context)
        l = None
        dl = {}
        debug_arr = []
        db_mconn = None
        user_id = kwargs.get("user_id", 3)
    
    if df.empty:
        log("No transactions to update.")
        return
    
    if db_mconn is None:
        log("Database master connection not available. Cannot save updated categories.")
        return
    
    try:
        cursor = db_mconn.cursor()
        updates_count = 0
        
        for _, row in df.iterrows():
            # Get the category ID from the category name
            category_id = to_category_id(row['category'])
            if category_id is None:
                log(f"Skipping transaction {row.get('transaction_id', 'unknown')}: Invalid category '{row['category']}'")
                continue
            
            transaction_id = row.get('transaction_id')
            account_id = row.get('account_id')
            
            if transaction_id is None:
                log("Skipping row: Missing transaction_id")
                continue
            
            # Update transaction
            update_query = """
            UPDATE transactions 
            SET ai_category_id = %s,
                ai_category_datetime = NOW()
            WHERE transaction_id = %s;
            
            -- Add transaction change record
            INSERT INTO transaction_changes 
            (user_id, account_id, transaction_id, change_datetime, change)
            VALUES (%s, %s, %s, NOW(), 'm');
            """
            
            params = [
                category_id,  # for update
                transaction_id,  # for update
                user_id,  # for insert
                account_id if account_id is not None else -1,  # for insert
                transaction_id  # for insert
            ]
            
            log(f"Updating transaction {transaction_id} with category {row['category']}")
            
            cursor.execute(update_query, params)
            updates_count += 1
        
        db_mconn.commit()
        log(f"Successfully updated {updates_count} transactions with new categories.")
    
    except Exception as e:
        if db_mconn:
            db_mconn.rollback()
        error_msg = f"Error updating transactions: {str(e)}"
        error_msg += f"\nTraceback: {traceback.format_exc()}"
        log(error_msg)

