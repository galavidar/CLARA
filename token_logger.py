import os
import datetime
from config import TOKEN_LOG_FILE

def log_tokens(model_variant: str = 'gpt-4o-mini', prompt_tokens: int = 0, completion_tokens: int = 0, total_tokens: int = None):
    """
    Logs token usage and estimated cost for gpt-4o.
    
    Parameters:
        model_variant (str): Model variant, e.g., 'mini', 'standard', etc.
            - 'mini' uses a different cost rate.
        prompt_tokens (int): Number of input tokens.
        completion_tokens (int): Number of output tokens.
        total_tokens (int): Total tokens (if only this is available).
    """

    COST_INPUT = 0.15/1000000 if 'mini' in model_variant else 2.5/1000000  # $ per token
    COST_OUTPUT = 0.6/1000000  if 'mini' in model_variant else 10/1000000  # $ per token

    # Ensure log file exists
    if not os.path.exists(TOKEN_LOG_FILE):
        with open(TOKEN_LOG_FILE, "w") as f:
            f.write("timestamp,prompt_tokens,completion_tokens,total_tokens,cost_usd,cumulative_tokens,total_cost_usd\n")

    # Read last total cost if exists
    try:
        with open(TOKEN_LOG_FILE, "r") as f:
            lines = f.readlines()
            if len(lines) > 1:
                last_line = lines[-1].strip().split(",")
                last_total_tokens = int(last_line[-2])  # cumulative_tokens column
                last_total_cost = float(last_line[-1])  # cumulative_cost_usd column
            else:
                last_total_tokens, last_total_cost = 0, 0.0
    except Exception:
        last_total_tokens, last_total_cost = 0, 0.0

    # If we only have total tokens, assume input rate (conservative underestimate)
    if total_tokens is not None and (prompt_tokens == 0 and completion_tokens == 0):
        cost = total_tokens * COST_INPUT
    else:
        cost = (prompt_tokens * COST_INPUT) + (completion_tokens * COST_OUTPUT)
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens

    # Update cumulative sums
    cumulative_tokens = last_total_tokens + total_tokens
    cumulative_cost = last_total_cost + cost

    # Append new log entry
    with open(TOKEN_LOG_FILE, "a") as f:
        f.write(f"{datetime.datetime.now().isoformat()},{prompt_tokens},{completion_tokens},{total_tokens},{cost:.6f},{cumulative_tokens},{cumulative_cost:.6f}\n")

    # Print live update
    print(f"[Token Logger] +{total_tokens} tokens (${cost:.6f}) | Total: {cumulative_tokens} tokens, ${cumulative_cost:.6f}")
