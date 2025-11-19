from penny.tool_funcs.should_remind import should_remind
from typing import Tuple


def create_reminder(what: str, when: str) -> Tuple[bool, str]:
    """
    Create a reminder based on the what and when parameters.
    
    Args:
        what: What to be reminded about: transaction coming in, subscription getting 
                refunded, account balances, or a clear general task 
                (e.g., "cancel Netflix subscription", "checking account balance drops below $1000").
        when: When the reminder will be relevant: date, condition, or frequency 
               (e.g., "at the end of this year (December 31st)", "immediately when condition is met").
    
    Returns:
        Tuple[bool, str]: (success, message) where success is True if the reminder was created successfully
    """
    print("\n" + "=" * 80, flush=True)
    print("[CREATE_REMINDER] Calling should_remind", flush=True)
    print(f"  what: {what}", flush=True)
    print(f"  when: {when}", flush=True)
    print("=" * 80 + "\n", flush=True)
    
    try:
        message, next_check_date, should_remind_code = should_remind(what, when)
        
        print("\n" + "=" * 80, flush=True)
        print("[CREATE_REMINDER] Generated should_remind_code:", flush=True)
        print("=" * 80, flush=True)
        print(should_remind_code, flush=True)
        print("=" * 80, flush=True)
        print(f"[CREATE_REMINDER] Execution result:", flush=True)
        print(f"  message: {message}", flush=True)
        print(f"  next_check_date: {next_check_date}", flush=True)
        print("=" * 80 + "\n", flush=True)
        
        # save_reminder_request(what, when, should_remind_code, next_check_date)

        if message is None and next_check_date is not None:
            return True, f"Reminder request '{what} {when}' has been saved but no reminder to send now. Next check date: {next_check_date}."
        
        return True, message
    except Exception as e:
        print("\n" + "=" * 80, flush=True)
        print(f"[CREATE_REMINDER] Error: {str(e)}", flush=True)
        import traceback
        print(traceback.format_exc(), flush=True)
        print("=" * 80 + "\n", flush=True)
        return False, f"Error creating reminder: {str(e)}"