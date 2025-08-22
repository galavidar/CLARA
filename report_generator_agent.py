import os
import numpy as np
from datetime import datetime
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from token_logger import log_tokens
from langchain_openai import AzureChatOpenAI
from config import AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, API_VERSION, CHAT_DEPLOYMENT, HF_API_KEY
from prompts import build_loan_report_prompt


def run_agent(chat_model, loan_data, profiles, user_features, interest_rate, loan_term, decision, risk_score, user_directives=None, token_counts=False):
    """
    Generate a JSON-structured loan decision report.
    """
    prompt = build_loan_report_prompt(
        loan_data=loan_data,
        profiles=profiles,
        features=user_features,
        interest_rate=interest_rate,
        loan_term=loan_term,
        decision=decision,
        risk_score=risk_score,
        user_directives=user_directives
    )

    response = chat_model.invoke(prompt)

    if token_counts:
        log_tokens(response)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('./outputs/reports_agent_responses.txt', "a", encoding="utf-8") as f:
        f.write(f"\n[{timestamp}] Task: Report Generation\n")
        f.write(response.content.strip())  # remove leading/trailing whitespace
        f.write("\n")
    
    return response.content

def generate_loan_report(loan_data, profiles, user_features, interest_rate, loan_term, decision, risk_score, user_directives=None):
    chat_model = AzureChatOpenAI(
        azure_deployment=CHAT_DEPLOYMENT,
        azure_endpoint = AZURE_OPENAI_ENDPOINT,
        api_key = AZURE_OPENAI_API_KEY,
        openai_api_type = "azure",
        openai_api_version = API_VERSION,
        model = 'gpt-4o-mini'
    )
    report = run_agent(
        chat_model=chat_model,
        loan_data=loan_data,
        profiles=profiles,
        user_features=user_features,
        interest_rate=interest_rate,
        loan_term=loan_term,
        decision=decision,
        risk_score=risk_score,
        user_directives=user_directives,
        token_counts=True
    )
    return report


    
def test():
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_API_KEY

    llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-120b",   # free OSS model
        task="text-generation",
        max_new_tokens=2056,
        do_sample=False,
        repetition_penalty=1.03,
        provider="auto",
    )
    chat_model = ChatHuggingFace(llm=llm)
    report = run_agent(
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
                        "expense_volatility": 0,
                        "savings_habit": 0,
                        "debt_dependence": 1,
                        "category_concentration_risk": 1
                    },
                    "reasoning": {
                        "liquidity_stress": "No overdrafts but credit-card payment ratio >1 and negative savings rate indicate limited liquid buffers.",
                        "growth_potential": "Income is trending upward, expenses are decreasing, and savings are growing, showing positive momentum.",
                        "income_stability": "Income std is low (≈2 percent of mean) and trend is positive, confirming stable earnings.",
                        "expense_volatility": "Expense std is modest (≈5.6 percent of mean) and category volatility is low, so spending is not highly volatile.",
                        "savings_habit": "Average savings rate is negative (~-18 percent), indicating the user does not consistently save.",
                        "debt_dependence": "Credit-card payment ratio exceeds 1, suggesting reliance on credit relative to income/limit.",
                        "category_concentration_risk": "Top category accounts for ~46 percent of spending, a strong concentration risk."
                    }
                    },
        user_features={'income_mean': np.float64(3121.1), 'income_std': np.float64(57.75629056648284), 'expense_mean': np.float64(3686.3133333333335), 'expense_std': np.float64(208.44803413161108), 'savings_rate_mean': np.float64(-0.18159607641009798), 'overdraft_frequency': np.float64(0.0), 'discretionary_share': np.float64(0.26089677115754273), 'top_category_share': np.float64(0.46020332558401744), 'category_volatility': np.float64(0.05283480965030089), 'cc_payment_ratio_mean': np.float64(1.137476855584842), 'last3_category_shares': {'Clothing': 0.21064161977993687, 'Dining': 0.041776037167873085, 'Electronics': 0.3741130322684745, 'Entertainment': 0.0043969970596531086, 'Groceries': 0.019116141288624964, 'Health': 0.04176493121022646, 'Home': 0.06402692285363044, 'Travel': 0.24416431837158062}, 'income_trend': np.float64(12.104999999999979), 'expense_trend': np.float64(-207.85999999999976), 'savings_trend': np.float64(219.96500000000015)},
        interest_rate=0.12,
        loan_term=36,
        decision="approved",
        risk_score=0.4,
        user_directives=None
    )
    print(report)

if __name__ == "__main__":
    # test()
    COST_OUTPUT = 0.6/1000000 
    COST_INPUT = 0.15/1000000 
    cost1 = 682*COST_INPUT + 323*COST_OUTPUT
    print(f"Cost 1: {cost1}")
    cost2 = 682*COST_INPUT + 309*COST_OUTPUT
    print(f"Cost 2: {cost2}")
    print(f"Total cost: {cost1+cost2}")
