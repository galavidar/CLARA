import pandas as pd
import numpy as np
import joblib
import zipfile

def load_weights():
    with zipfile.ZipFile("./weights/interest_model.zip", 'r') as zip_file:
        with zip_file.open("interest_model.pkl") as pkl_file:
            interest_model = joblib.load(pkl_file)
    risk_model = joblib.load("./weights/risk_model.pkl")
    tfidf_emp = joblib.load("./weights/tfidf_emp.pkl")
    tfidf_purpose = joblib.load("./weights/tfidf_purpose.pkl")
    home_ownership_cols = joblib.load("./weights/home_ownership_categories.pkl")  # list of one-hot columns used during training
    return interest_model, risk_model, tfidf_emp, tfidf_purpose, home_ownership_cols

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

def rename_columns(df):
    df = df.rename(columns={
        "loan_amount": "Amount Requested",
        "loan_term": "term",
        "job_title": "emp_title",
        "job_tenure": "Employment Length",
        "home_status": "home_ownership",
        "annual_income": "annual_inc",
        "loan_purpose": "purpose",
        "delinquencies": "delinq_2yrs",
        "credit_score": "credit_score",
        "accounts": "num_actv_bc_tl",
        "bankruptcy": "pub_rec_bankruptcies",
    })
    return df

def risk_assesment(form_data):
    interest_model, risk_model, tfidf_emp, tfidf_purpose, home_ownership_cols = load_weights()

    df = pd.DataFrame([form_data])
    print(df)
    df = df.drop('credit_score', axis=1)
    df['bankruptcy'] = df['bankruptcy'].map({"no": 0, "yes": 1})
    df["delinquencies"] = df["delinquencies"].map({"no": 0, "yes": 1})
    df['Debt-To-Income Ratio'] = (df['monthly_debt'] / df['annual_income'])
    df = rename_columns(df)

    df.replace({"n/a": np.nan, "N/A": np.nan, "NA": np.nan, "": np.nan}, inplace=True)
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

    return df["clara_predicted_interest"]/100, df["clara_risk_score"]
 

def test():
    sample_data = {
                "loan_amount": 10000,
                "loan_term": 36,
                "job_title": 'Doctor',
                "job_tenure": 10,
                "home_status": 'OWN',
                "annual_income": 120000,
                "loan_purpose": 'car',
                "monthly_debt": 4000,
                "delinquencies": 'no',
                "credit_score": 750,
                "accounts": 5,
                "bankruptcy": 'no',
            }
    interest, risk = risk_assesment(sample_data)
    print(f"Predicted Interest Rate: {interest.values[0]:.4f}")
    print(f"Risk Score: {risk.values[0]:.4f}")

if __name__ == "__main__":
    test()