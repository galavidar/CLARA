import pandas as pd
import numpy as np
import json
from typing import Dict
from openai import OpenAI
from token_logger import log_tokens
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

PROFILE_DEFS = {
    "discretionary_spending_share": "High share of spending on non-essential categories (entertainment, shopping, leisure).",
    "liquidity_stress": "Signs of low liquidity such as frequent overdrafts, high CC utilization, or low savings.",
    "growth_potential": "Evidence that income and savings are trending upwards.",
    "income_stability": "Consistency of income levels month-to-month.",
    "expense_volatility": "Fluctuations in monthly spending patterns.",
    "savings_habit": "Tendency to consistently save from income.",
    "debt_dependence": "Reliance on credit card spending relative to income or limit.",
    "category_concentration_risk": "Spending heavily concentrated in one or two categories."
}

class ProfileOutput(BaseModel):
    profiles: Dict[str, int] = Field(..., description="Mapping of profile name to 0/1 decision")
    reasoning: Dict[str, str] = Field(..., description="Reasoning for each profile")

def preprocess_user(bank_df: pd.DataFrame, card_df: pd.DataFrame) -> Dict:
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

    card_monthly = card_df.groupby([pd.Grouper(key="date", freq="MS"), "category"])["amount_paid"].sum().unstack(fill_value=0)
    card_monthly["total_card_spend"] = card_monthly.sum(axis=1)

    df = bank_monthly.merge(card_monthly, on="date", how="left").fillna(0)
    df["net_savings"] = df["income"] - df["expense"]
    df["savings_rate"] = df["net_savings"] / (df["income"].replace(0, np.nan))
    df["overdraft"] = (df["balance"] < 0).astype(int)

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
        "discretionary_share": df[[c for c in df.columns if "entertainment" in c or "shopping" in c]].sum().sum() / df["total_card_spend"].sum() if "total_card_spend" in df else 0,
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
    profiles = {}
    profiles["income_stability"] = int(features["income_std"] / (features["income_mean"]+1e-6) < 0.1)
    profiles["expense_volatility"] = int(features["expense_std"] / (features["expense_mean"]+1e-6) > 0.3)
    profiles["savings_habit"] = int(features["savings_rate_mean"] > 0.1)
    profiles["category_concentration_risk"] = int(features["top_category_share"] > 0.4)
    return profiles

def query_llm_for_profiles(client, user_features: Dict, rule_profiles: Dict) -> Dict:
    prompt = f"""
    You are a financial analyst agent. 
    Your task is to infer and validate financial behavioural profiles for a loan applicant.

    Profiles and their definitions:
    {json.dumps(PROFILE_DEFS, indent=2)}

    You are given:
    - Structured user features (summary stats + last 3 months)
    - Rule-based profile decisions (0=No, 1=Yes)

    Instructions:
    - For each profile, validate the rule-based decision if provided, overriding if necessary.
    - For profiles not rule-based, infer them directly.
    - Provide reasoning for each profile.

    Data: {json.dumps(user_features, indent=2)}
    Rule-based profiles: {json.dumps(rule_profiles, indent=2)}

    Return valid JSON with:
      - "profiles": mapping profile->0/1
      - "reasoning": mapping profile->short explanation
    """

    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    return json.loads(resp.choices[0].message.content)


def analyze_user(bank_path: str, card_path: str) -> Dict:
    bank_df = pd.read_csv(bank_path)
    card_df = pd.read_csv(card_path)

    features = preprocess_user(bank_df, card_df)
    rule_profiles = infer_rule_based_profiles(features["summary"])
    llm_profiles = query_llm_for_profiles(features, rule_profiles)

    return {
        "features": features,
        "rule_based_profiles": rule_profiles,
        "llm_profiles": llm_profiles,
        "merged_profiles": llm_profiles["profiles"],
        "reasoning": llm_profiles["reasoning"]
    }

def langchain_profile_agent():
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)
    structured_llm = llm.with_structured_output(ProfileOutput)

    return RunnableMap({
        "analysis": RunnableLambda(lambda x: analyze_user(x["bank_path"], x["card_path"])),
        "structured_profiles": structured_llm
    })

def test():
    bank = pd.read_csv('./backend/data/synthetic_users/bank_user_0001.csv')
    card = pd.read_csv('./backend/data/synthetic_users/card_user_0001.csv')
    out = preprocess_user(bank, card)
    out2 = infer_rule_based_profiles(out)
    print(out2) 

if __name__ == "__main__":
    test()