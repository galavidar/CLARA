import pandas as pd
import numpy as np
from typing import Dict
import json
from datetime import datetime
from token_logger import log_tokens
from pydantic import BaseModel, Field
from config import BEHAVIOURAL_LOG_FILE, COUNT_TOKENS
from agents.prompts import build_behavioural_json_prompt
from utils import get_model, normalize_json

class ProfileOutput(BaseModel):
    """
    Enforce a structure for the profile output, currently not used.
    """
    profiles: Dict[str, int] = Field(..., description="Mapping of profile name to 0/1 decision")
    reasoning: Dict[str, str] = Field(..., description="Reasoning for each profile")

def preprocess_user(bank_df: pd.DataFrame, card_df: pd.DataFrame) -> Dict:
    """
    Preprocess and aggregate user data from bank and card DataFrames, create meaningful metrics for analysis.
    """
    bank_df['date'] = pd.to_datetime(bank_df['date'])
    card_df['date'] = pd.to_datetime(card_df['date'])
    bank_df["month"] = bank_df["date"].dt.to_period("M")
    month_counts = bank_df.groupby("month").size().sort_values(ascending=False)
    # Pick the top 3 months with most transactions
    top_months = month_counts.head(3).index
    bank_df = bank_df[bank_df["month"].isin(top_months)].copy()


    bank_monthly = bank_df.groupby(pd.Grouper(key="date", freq="MS")).agg({
        "income": "sum",
        "expense": "sum",
        "balance": "last"
    }).reset_index()

    # aggreagate and add card transactions to bank data
    card_monthly = card_df.groupby([pd.Grouper(key="date", freq="MS"), "category"])["amount_paid"].sum().unstack(fill_value=0)
    card_monthly["total_card_spend"] = card_monthly.sum(axis=1)

    df = bank_monthly.merge(card_monthly, on="date", how="left").fillna(0)
    df["net_savings"] = df["income"] - df["expense"]
    df["savings_rate"] = df["net_savings"] / (df["income"].replace(0, np.nan))
    df["overdraft"] = (df["balance"] < 0).astype(int)

    # calculate category shares
    categories = [c for c in df.columns if c not in ["date","income","expense","balance","net_savings","savings_rate","overdraft","total_card_spend"]]
    for cat in categories:
        df[f"{cat}_share"] = df[cat] / df["total_card_spend"].replace(0, np.nan)

    cc_payment_mask = bank_df["description"].str.contains("Credit Card Payment", case=False, na=False)
    monthly_cc_payment = bank_df.loc[cc_payment_mask].groupby(pd.Grouper(key="date", freq="MS"))["expense"].sum()
    df = df.merge(monthly_cc_payment.rename("cc_payment"), on="date", how="left").fillna(0)
    df["cc_payment_ratio"] = df["cc_payment"] / (df["total_card_spend"].replace(0, np.nan))

    monthly_totals = card_df.groupby([bank_df['date'].dt.to_period("M"), "category"])["amount_paid"].sum().unstack(fill_value=0)
    monthly_shares = monthly_totals.div(monthly_totals.sum(axis=1), axis=0)

    # Take average share across last 3 months
    if len(monthly_shares) >= 3:
        last3 = monthly_shares.tail(3).mean().to_dict()
    else:
        last3 = monthly_shares.mean().to_dict()

    features = {
        "income_mean": df["income"].mean(),
        "income_std": df["income"].std(), 
        "expense_mean": df["expense"].mean(),
        "expense_std": df["expense"].std(),
        "savings_rate_mean": df["savings_rate"].mean(),
        "overdraft_frequency": df["overdraft"].mean(),
        "discretionary_share": df[[c for c in df.columns if "Entertainment" in c or "Travel" in c or "Dining" in c]].sum().sum() / df["total_card_spend"].sum() if "total_card_spend" in df else 0,
        "top_category_share": df[categories].sum().max() / df["total_card_spend"].sum() if len(categories) > 0 else 0,
        "category_volatility": df[categories].div(df["total_card_spend"], axis=0).std().mean() if len(categories) > 0 else 0,
        "cc_payment_ratio_mean": df["cc_payment_ratio"].mean(),
         "last3_category_shares": last3,
    }

    if len(df) > 1:
        t = np.arange(len(df))
        features["income_trend"] = np.polyfit(t, df["income"].values, 1)[0]
        features["expense_trend"] = np.polyfit(t, df["expense"].values, 1)[0]
        features["savings_trend"] = np.polyfit(t, df["net_savings"].values, 1)[0]
    else:
        features["income_trend"] = features["expense_trend"] = features["savings_trend"] = 0

    return features


def infer_rule_based_profiles(features: Dict) -> Dict[str, int]:
    """
    Infer rule-based profiles from user features.
    """
    profiles = {}
    profiles["income_stability"] = int(features["income_std"] / (features["income_mean"]+1e-6) < 0.1)
    profiles["expense_volatility"] = int(features["expense_std"] / (features["expense_mean"]+1e-6) > 0.3)
    profiles["savings_habit"] = int(features["savings_rate_mean"] > 0.1)
    profiles["category_concentration_risk"] = int(features["top_category_share"] > 0.4)
    return profiles

def run_agent(chat_model, user_features, rule_profiles, supervisor_comments=None):
    """
    Run the LLM agent for profile inference.
    Logs tokens and returns the JSON result.
    """
    prompt = build_behavioural_json_prompt(user_features, rule_profiles, supervisor_comments)

    #structured_model = chat_model.with_structured_output(ProfileOutput)
    resp = chat_model.invoke(prompt)
    response_dict = normalize_json(resp.content)
    usage = resp.response_metadata['token_usage']

    if COUNT_TOKENS:
        log_tokens('behavioural_features', resp.response_metadata['model_name'], prompt_tokens=usage['prompt_tokens'], completion_tokens=usage['completion_tokens'], total_tokens=usage['total_tokens'])
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(BEHAVIOURAL_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n[{timestamp}] Task: Behavioural Profiling\n")
        json.dump(response_dict, f, indent=2, ensure_ascii=False)
        f.write("\n")
    
    return response_dict


def extract_behavioural_features(bank, card, supervisor_comments=None):
    """
    Wrapper to run the behavioural profiling agent
    """
    user_features = preprocess_user(bank, card)
    profiles = infer_rule_based_profiles(user_features)

    chat_model = get_model()

    response = run_agent(chat_model, user_features, profiles, supervisor_comments)
    return response, user_features

def test():
    bank = pd.read_csv('./dev/data/synthetic_users/bank_user_0001.csv')
    card = pd.read_csv('./dev/data/synthetic_users/card_user_0001.csv')
    out_df = preprocess_user(bank, card)
    out_prof = infer_rule_based_profiles(out_df)
    chat_model = get_model()
    r = run_agent(chat_model, out_df, out_prof)
    print(r)

if __name__ == "__main__":
    test()