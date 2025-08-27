from datetime import datetime
import numpy as np
import json
from config import EVALUATOR_LOG_FILE, COUNT_TOKENS
from token_logger import log_tokens
from prompts import build_evaluation_prompt
from utils import get_model, normalize_json

def run_agent(chat_model, loan_data, applicant_features, profiles, decision, rate, term, risk_score, user_directives=None, bank_risk_tolerance="medium"):
    """
    Run the Evaluator Agent to validate outputs of other agents.
    """
    prompt = build_evaluation_prompt(loan_data, applicant_features, profiles, rate, term, risk_score, decision, user_directives, bank_risk_tolerance)

    response = chat_model.invoke(prompt)
    resp_dict = normalize_json(response.content)
    # Extract usage if available
    usage = response.response_metadata['token_usage']
    if COUNT_TOKENS:
        log_tokens('evaluation', response.response_metadata['model_name'], prompt_tokens=usage['prompt_tokens'], completion_tokens=usage['completion_tokens'], total_tokens=usage['total_tokens'])

    # Save raw output
    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    with open(EVALUATOR_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n[{timestamp}] Task: Evaluation\n")
        json.dump(resp_dict, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return resp_dict

def evaluate_outputs(loan_data, profiles, user_features, interest_rate, loan_term, decision, risk_score, user_directives=None, risk_tolerance="medium"):
    """
    Wrapper to run the agent that generates a loan report using the specified parameters.
    """
    chat_model = get_model()
    evaluation_result = run_agent(
        chat_model=chat_model,
        loan_data=loan_data,
        profiles=profiles,
        applicant_features=user_features,
        rate=interest_rate,
        term=loan_term,
        decision=decision,
        risk_score=risk_score,
        user_directives=user_directives,
        bank_risk_tolerance=risk_tolerance
    )
    return evaluation_result


def test():

    chat_model = get_model()
    res = run_agent(
        chat_model=chat_model,
        loan_data={
                "loan_amount": 10000,
                "loan_term": 36,
                "job_title": 'Doctor',
                "job_tenure": 10,
                "home_status": 'OWN',
                "annual_income": 120000,
                "loan_purpose": 'car',
                "total_debt": 50000,
                "delinquencies": 'no',
                "credit_score": 750,
                "accounts": 5,
                "bankruptcy": 'no',
            },
        profiles=  {
                    "profiles": {
                        "discretionary_spending_share": 0,
                        "liquidity_stress": 1,
                        "growth_potential": 1, 
                        "income_stability": 1,
                        "expense_volatility": 0,
                        "savings_habit": 0,
                        "debt_dependence": 1,
                        "category_concentration_risk": 1
                    },
                    "reasoning": {
                        "discretionary_spending_share": "Discretionary share is ~26%, which is moderate rather than high.",
                        "liquidity_stress": "No overdrafts but credit-card payment ratio >1 and negative savings rate indicate limited liquid buffers.",
                        "growth_potential": "Income is trending upward, expenses are decreasing, and savings are growing, showing positive momentum.",
                        "income_stability": "Income std is low (≈2 percent of mean) and trend is positive, confirming stable earnings.",
                        "expense_volatility": "Expense std is modest (≈5.6 percent of mean) and category volatility is low, so spending is not highly volatile.",
                        "savings_habit": "Average savings rate is negative (~-18 percent), indicating the user does not consistently save.",
                        "debt_dependence": "Credit-card payment ratio exceeds 1, suggesting reliance on credit relative to income/limit.",
                        "category_concentration_risk": "Top category accounts for ~46 percent of spending, a strong concentration risk."
                    }
                    },
        applicant_features={'income_mean': np.float64(3121.1), 'income_std': np.float64(57.75629056648284), 'expense_mean': np.float64(3686.3133333333335), 'expense_std': np.float64(208.44803413161108), 'savings_rate_mean': np.float64(-0.18159607641009798), 'overdraft_frequency': np.float64(0.0), 'discretionary_share': np.float64(0.26089677115754273), 'top_category_share': np.float64(0.46020332558401744), 'category_volatility': np.float64(0.05283480965030089), 'cc_payment_ratio_mean': np.float64(1.137476855584842), 'last3_category_shares': {'Clothing': 0.21064161977993687, 'Dining': 0.041776037167873085, 'Electronics': 0.3741130322684745, 'Entertainment': 0.0043969970596531086, 'Groceries': 0.019116141288624964, 'Health': 0.04176493121022646, 'Home': 0.06402692285363044, 'Travel': 0.24416431837158062}, 'income_trend': np.float64(12.104999999999979), 'expense_trend': np.float64(-207.85999999999976), 'savings_trend': np.float64(219.96500000000015)},
        rate=0.12,
        term=36,
        decision="approved",
        risk_score=0.4,
        user_directives=None
    )

    print(res)

if __name__ == "__main__":
    test()