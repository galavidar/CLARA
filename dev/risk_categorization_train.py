import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv("form_with_interest.csv", low_memory=False)
def label_loan_status(status):
    if pd.isna(status): return np.nan
    status = status.lower()
    bad_statuses = ["charged off", "default", "late", "grace", "does not meet the credit policy", "in grace period"]
    if any(term in status for term in bad_statuses):
        return 1
    elif "fully paid" in status:
        return 0
    return np.nan


df["label"] = df["loan_status"].apply(label_loan_status)


form_features = [
    "Amount Requested", "Loan Title", "emp_title", "Employment Length",
    "home_ownership", "annual_inc", "purpose", "Debt-To-Income Ratio",
    "delinq_2yrs", "num_actv_bc_tl", "pub_rec_bankruptcies","term"
]
existing_features = [f for f in form_features if f in df.columns]

df_model = df.dropna(subset=existing_features + ["label"]).copy()

def parse_emp_length(val):
    if pd.isna(val): return np.nan
    val = str(val)
    if "<" in val: return 0.5
    if "10+" in val: return 10
    digits = ''.join(filter(str.isdigit, val))
    return float(digits) if digits else np.nan

df_model.loc[:, "Employment Length"] = df_model["Employment Length"].apply(parse_emp_length)

def parse_term(val):
    if pd.isna(val): return np.nan
    val = str(val).lower().replace("months", "").strip()
    digits = ''.join(filter(str.isdigit, val))
    return float(digits) if digits else np.nan

df_model["term"] = df_model["term"].apply(parse_term)

# Fill missing numerics with median
numeric_cols = df_model[existing_features].select_dtypes(include=[np.number]).columns
df_model.loc[:, numeric_cols] = df_model[numeric_cols].fillna(df_model[numeric_cols].median())

df_encoded = pd.get_dummies(df_model[["home_ownership"]], drop_first=True)
home_ownership_categories = df_encoded.columns.tolist()

tfidf_emp = TfidfVectorizer(max_features=30)
emp_tfidf = tfidf_emp.fit_transform(df_model["emp_title"].fillna(""))
emp_df = pd.DataFrame(emp_tfidf.toarray(), columns=[f"emp_title_tfidf_{i}" for i in range(emp_tfidf.shape[1])])

tfidf_purpose = TfidfVectorizer(max_features=20)
purpose_tfidf = tfidf_purpose.fit_transform(df_model["purpose"].fillna(""))
purpose_df = pd.DataFrame(purpose_tfidf.toarray(), columns=[f"purpose_tfidf_{i}" for i in range(purpose_tfidf.shape[1])])

X = pd.concat([
    df_model[["Amount Requested", "Employment Length", "annual_inc", "Debt-To-Income Ratio", "delinq_2yrs", "num_actv_bc_tl", "pub_rec_bankruptcies","term"]].reset_index(drop=True),
    df_encoded.reset_index(drop=True),
    emp_df.reset_index(drop=True),
    purpose_df.reset_index(drop=True),
    df_model[["clara_predicted_interest"]].reset_index(drop=True)
], axis=1)

y = df_model["label"].values


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

joblib.dump(model, "risk_model.pkl")
joblib.dump(tfidf_emp, "tfidf_emp_risk.pkl")
joblib.dump(tfidf_purpose, "tfidf_purpose_risk.pkl")
joblib.dump(home_ownership_categories, "home_ownership_categories_risk.pkl")


y_pred_proba = model.predict_proba(X_test)[:, 1]
df_full = df.dropna(subset=existing_features).copy()
df_full.loc[:, "Employment Length"] = df_full["Employment Length"].apply(parse_emp_length)
df_full.loc[:, numeric_cols] = df_full[numeric_cols].fillna(df_model[numeric_cols].median())
df_full["term"] = df_full["term"].apply(parse_term)

encoded_full = pd.get_dummies(df_full[["home_ownership"]], drop_first=True)
emp_full_tfidf = tfidf_emp.transform(df_full["emp_title"].fillna(""))
emp_full_df = pd.DataFrame(emp_full_tfidf.toarray(), columns=emp_df.columns)
purpose_full_tfidf = tfidf_purpose.transform(df_full["purpose"].fillna(""))
purpose_full_df = pd.DataFrame(purpose_full_tfidf.toarray(), columns=purpose_df.columns)

X_all = pd.concat([
    df_full[["Amount Requested", "Employment Length", "annual_inc", "Debt-To-Income Ratio", "delinq_2yrs", "num_actv_bc_tl", "pub_rec_bankruptcies","term"]].reset_index(drop=True),
    encoded_full.reset_index(drop=True),
    emp_full_df.reset_index(drop=True),
    purpose_full_df.reset_index(drop=True),
    df_full[["clara_predicted_interest"]].reset_index(drop=True)
], axis=1)

X_all = X_all.reindex(columns=X.columns, fill_value=0)

p_default_all = model.predict_proba(X_all)[:, 1]
df_full.loc[:, "clara_risk_score"] = p_default_all

#df_full.to_csv("lendingclub_with_clara_risk_scores_form_only.csv", index=False)
