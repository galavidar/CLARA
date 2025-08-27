import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("new_form_data.csv")


interest_model = joblib.load("interest_model.pkl")
risk_model = joblib.load("risk_model.pkl")
tfidf_emp = joblib.load("tfidf_emp.pkl")
tfidf_purpose = joblib.load("tfidf_purpose.pkl")
home_ownership_cols = joblib.load("home_ownership_categories.pkl")  # list of one-hot columns used during training

def parse_emp_length(val):
    if pd.isna(val): return np.nan
    val = str(val)
    if "<" in val: return 0.5
    if "10+" in val: return 10
    digits = ''.join(filter(str.isdigit, val))
    return float(digits) if digits else np.nan

def parse_term(val):
    if pd.isna(val): return np.nan
    s = str(val).lower().replace("months", "").strip()
    digits = ''.join(filter(str.isdigit, s))
    return int(digits) if digits else np.nan


df["Employment Length"] = df["Employment Length"].apply(parse_emp_length)
df["term"] = df["term"].apply(parse_term)


numeric_cols = ["Amount Requested", "Employment Length", "annual_inc", "Debt-To-Income Ratio",
                "delinq_2yrs", "num_actv_bc_tl", "pub_rec_bankruptcies", "term"]
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
df_encoded = pd.get_dummies(df[["home_ownership"]], drop_first=True)

for col in home_ownership_cols:
    if col not in df_encoded:
        df_encoded[col] = 0
df_encoded = df_encoded[home_ownership_cols]  # enforce column order


emp_tfidf = tfidf_emp.transform(df["emp_title"].fillna(""))
emp_df = pd.DataFrame(emp_tfidf.toarray(), columns=[f"emp_title_tfidf_{i}" for i in range(emp_tfidf.shape[1])])

purpose_tfidf = tfidf_purpose.transform(df["purpose"].fillna(""))
purpose_df = pd.DataFrame(purpose_tfidf.toarray(), columns=[f"purpose_tfidf_{i}" for i in range(purpose_tfidf.shape[1])])


X_interest = pd.concat([
    df[numeric_cols].reset_index(drop=True),
    df_encoded.reset_index(drop=True),
    emp_df.reset_index(drop=True),
    purpose_df.reset_index(drop=True)
], axis=1)


df["clara_predicted_interest"] = interest_model.predict(X_interest)

X_risk = X_interest.copy()
X_risk["clara_predicted_interest"] = df["clara_predicted_interest"]

df["clara_risk_score"] = risk_model.predict_proba(X_risk)[:, 1]

# === 9. Done — Output Results ===
print(df[["Amount Requested", "clara_predicted_interest", "clara_risk_score"]])
#df.to_csv("scored_new_applicants.csv", index=False)
