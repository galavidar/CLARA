import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("form_data.csv", low_memory=False)

form_features = [
    "Amount Requested", "Loan Title", "emp_title", "Employment Length",
    "home_ownership", "annual_inc", "purpose", "Debt-To-Income Ratio",
    "delinq_2yrs", "num_actv_bc_tl", "pub_rec_bankruptcies", "term"
]
required_cols = form_features + ["int_rate"]+["loan_status"]
df_model = df.dropna(subset=required_cols).copy()

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

df_model["Employment Length"] = df_model["Employment Length"].apply(parse_emp_length)
df_model["term"] = df_model["term"].apply(parse_term)

numeric_cols = df_model[form_features].select_dtypes(include=[np.number]).columns
df_model[numeric_cols] = df_model[numeric_cols].fillna(df_model[numeric_cols].median())

df_encoded = pd.get_dummies(df_model[["home_ownership"]], drop_first=True)
home_ownership_categories = df_encoded.columns.tolist()

tfidf_emp = TfidfVectorizer(max_features=30)
emp_tfidf = tfidf_emp.fit_transform(df_model["emp_title"].fillna(""))
emp_df = pd.DataFrame(emp_tfidf.toarray(), columns=[f"emp_title_tfidf_{i}" for i in range(emp_tfidf.shape[1])])

tfidf_purpose = TfidfVectorizer(max_features=20)
purpose_tfidf = tfidf_purpose.fit_transform(df_model["purpose"].fillna(""))
purpose_df = pd.DataFrame(purpose_tfidf.toarray(), columns=[f"purpose_tfidf_{i}" for i in range(purpose_tfidf.shape[1])])


X = pd.concat([
    df_model[[
        "Amount Requested", "Employment Length", "annual_inc",
        "Debt-To-Income Ratio", "delinq_2yrs", "num_actv_bc_tl",
        "pub_rec_bankruptcies", "term"
    ]].reset_index(drop=True),
    df_encoded.reset_index(drop=True),
    emp_df.reset_index(drop=True),
    purpose_df.reset_index(drop=True)
], axis=1)

y = df_model["int_rate"].astype(float).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "interest_model.pkl")
joblib.dump(tfidf_emp, "tfidf_emp.pkl")
joblib.dump(tfidf_purpose, "tfidf_purpose.pkl")
joblib.dump(home_ownership_categories, "home_ownership_categories.pkl")

y_pred_all = model.predict(X)
df_model.loc[:, "clara_predicted_interest"] = y_pred_all
df_model.to_csv("form_with_interest.csv", index=False)

