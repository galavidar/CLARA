from datetime import datetime
import json
import numpy as np
from typing import Dict
from config import COUNT_TOKENS, DECISIONS_LOG_FILE, VECTOR_DB_URL, VECTOR_DB_API_KEY, RAG_EVAL_LOG_FILE
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.http.models import Filter
from time import time
from token_logger import log_tokens
from risk_agent import risk_assesment
from utils import normalize_json, get_model
from prompts import build_decision_prompt, build_rag_eval_prompt
from langchain.evaluation import load_evaluator

def retrieve_similar_cases(client: QdrantClient, collection_name: str, query_vector: np.ndarray, top_k: int = 10, filters: Filter = None):
    """
    Retrieves similar loan cases from Qdrant using the given vector.
    Returns a list of (score, payload) dicts.
    """
    try:
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
            query_filter=filters,
        )
        return [
            {
                "case_id": res.id,
                "score": res.score,
                "payload": res.payload
            }
            for res in search_results
        ]
    except Exception as e:
        print(f"⚠️ Retrieval error: {e}")
        return []
    
def run_evaluation(eval_model, query: str, prediction: str, reference: str = "", criteria: str = "faithfulness"):
    """
    Run an evaluation of the model output against the given criteria.
    Counts tokens explicitly (same pattern as run_agent).
    """
    eval_prompt = build_rag_eval_prompt(criteria, query, prediction, reference)
    response = eval_model.invoke(eval_prompt)
    eval_dict = normalize_json(response.content)

    # Extract usage metadata
    usage = response.response_metadata.get("token_usage", {})
    if COUNT_TOKENS:
        log_tokens(
            "rag_eval",
            response.response_metadata.get("model_name", "unknown"),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )
    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    with open(RAG_EVAL_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n[{timestamp}] Task: RAG {criteria} Evaluation \n")
        json.dump(eval_dict, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return eval_dict


def evaluate_decision_with_langchain(eval_model, loan_data: dict, decision: dict, retrieved_cases: list):
    """
    Evaluate a decision object for multiple criteria: faithfulness, relevance, correctness.
    Uses explicit eval_model calls (token counting enabled).
    """
    query = loan_data
    prediction = decision
    reference = retrieved_cases

    eval_results = {
        "faithfulness": run_evaluation(eval_model, query, prediction, reference, criteria="faithfulness"),
        "relevance": run_evaluation(eval_model, query, prediction, reference, criteria="relevance"),
        "correctness": run_evaluation(eval_model, query, prediction, reference, criteria="correctness"),
    }

    return eval_results


def run_agent(chat_model, loan_data, user_features, profiles, risk_score, interest_rate, loan_term, evaluator_comments=None, retrieved_cases=None):
    """
    Runs the decision-making agent
    """
    prompt = build_decision_prompt(loan_data, user_features, profiles, interest_rate, loan_term, risk_score, retrieved_cases, evaluator_comments)
    response = chat_model.invoke(prompt)
    resp_dict = normalize_json(response.content)
    # Extract usage if available
    usage = response.response_metadata['token_usage']
    if COUNT_TOKENS:
        log_tokens('decisions', response.response_metadata['model_name'], prompt_tokens=usage['prompt_tokens'], completion_tokens=usage['completion_tokens'], total_tokens=usage['total_tokens'])

    # Save raw output
    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    with open(DECISIONS_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n[{timestamp}] Task: Decsision\n")
        json.dump(resp_dict, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return resp_dict


def loan_to_text(loan_data: Dict) -> str:
    """
    Convert loan_data dict into a meaningful text representation for embedding.
    """
    text_repr = (
        f"Loan request: {loan_data['loan_amount']} USD over {loan_data['loan_term']} months. "
        f"Applicant: {loan_data['job_title']} with {loan_data['job_tenure']} years tenure. "
        f"Home: {loan_data['home_status']}. Annual income: {loan_data['annual_income']}. "
        f"Purpose: {loan_data['loan_purpose']}. Current debt: {loan_data['monthly_debt']}. "
        f"Delinquencies: {loan_data['delinquencies']}. Credit score: {loan_data['credit_score']}. "
        f"Accounts: {loan_data['accounts']}. Bankruptcy history: {loan_data['bankruptcy']}."
    )
    return text_repr

def create_embeddings(loan_data: Dict) -> np.ndarray:
    """
    Generate embedding for loan_data.
    """
    text_input = loan_to_text(loan_data)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(text_input)
    return np.array(emb)

def get_rag_params(loan_data):
    client = QdrantClient(
        url=VECTOR_DB_URL,
        api_key=VECTOR_DB_API_KEY,
        timeout=360,
        prefer_grpc=True
    )
    query_vec = create_embeddings(loan_data)
    return client, query_vec


def decide(loan_data, user_features, behavioural_profiles, evaluator_comments=None):
    """
    Wrapper to obtain the risk and run the agent
    """
    retrieved_cases = []
    try:
        client, query_vec = get_rag_params(loan_data)
        retrieved_cases = retrieve_similar_cases(
            client=client,
            collection_name="lending_club_loans",
            query_vector=query_vec,
            top_k=10
            )
    except Exception as e:
        print(f"⚠️ Could not retrieve cases: {e}")
    
    int_rate, risk_sc = risk_assesment(loan_data)
    interest = int_rate[0]
    risk_score = risk_sc[0]
    chat_model = get_model(open_ai_model="gpt-4o")
    decision = run_agent(
        chat_model=chat_model,
        loan_data=loan_data,
        user_features=user_features,
        profiles=behavioural_profiles,
        risk_score=risk_score,
        interest_rate=interest,
        loan_term=loan_data['loan_term'],
        evaluator_comments=evaluator_comments,
        retrieved_cases=retrieved_cases
    )
    decision['risk_score'] = risk_score
    eval_model = get_model()
    eval_results = evaluate_decision_with_langchain(eval_model, loan_data, decision, retrieved_cases)
    decision['evaluation'] = eval_results

    return decision

def test_rag():
    loan_data={"loan_amount": 10000,
                "loan_term": 36,
                "job_title": 'Doctor',
                "job_tenure": 10,
                "home_status": 'OWN',
                "annual_income": 120000,
                "loan_purpose": 'car',
                "monthly_debt": 50000,
                "delinquencies": 'no',
                "credit_score": 750,
                "accounts": 5,
                "bankruptcy": 'no',}
    client, query_vec = get_rag_params(loan_data)
    rag_res = retrieve_similar_cases(client, 'lending_club_loans', query_vec)
    print(rag_res)

def test():
    loan_data={
                "loan_amount": 10000,
                "loan_term": 36,
                "job_title": 'Doctor',
                "job_tenure": 10,
                "home_status": 'OWN',
                "annual_income": 120000,
                "loan_purpose": 'car',
                "monthly_debt": 5000,
                "delinquencies": 'no',
                "credit_score": 750,
                "accounts": 5,
                "bankruptcy": 'no',
            }
    decision = decide(loan_data=loan_data, 
        behavioural_profiles=  {
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
        user_features={'income_mean': np.float64(3121.1), 'income_std': np.float64(57.75629056648284), 'expense_mean': np.float64(3686.3133333333335), 'expense_std': np.float64(208.44803413161108), 'savings_rate_mean': np.float64(-0.18159607641009798), 'overdraft_frequency': np.float64(0.0), 'discretionary_share': np.float64(0.26089677115754273), 'top_category_share': np.float64(0.46020332558401744), 'category_volatility': np.float64(0.05283480965030089), 'cc_payment_ratio_mean': np.float64(1.137476855584842), 'last3_category_shares': {'Clothing': 0.21064161977993687, 'Dining': 0.041776037167873085, 'Electronics': 0.3741130322684745, 'Entertainment': 0.0043969970596531086, 'Groceries': 0.019116141288624964, 'Health': 0.04176493121022646, 'Home': 0.06402692285363044, 'Travel': 0.24416431837158062}, 'income_trend': np.float64(12.104999999999979), 'expense_trend': np.float64(-207.85999999999976), 'savings_trend': np.float64(219.96500000000015)},
        evaluator_comments=None
    )
    print(decision)

if __name__ == "__main__":
    test()