import streamlit as st
import time
import json
import pandas as pd


# PAGE CONTROL
if "page" not in st.session_state:
    st.session_state.page = "welcome"

def go_to(page_name):
    st.session_state.page = page_name


# WELCOME PAGE
if st.session_state.page == "welcome":
    st.title("CLARA")
    st.markdown("""
    Welcome to **CLARA**, your intelligent loan application assistant.  
    CLARA will analyze a loan applicant's financial data and help you decide whether to approve or deny the loan.
    CLARA will also help you draft your report regarding the decision and provide similar cases that back up your decision.
                
    Before we begin, you will need to input the applicant's request data, and to upload their bank and credit card statements as .csv files, so make sure to have them handy!

    Click **Continue** to get started.
    """)
    if st.button("Continue ➡️"):
        go_to("form")


# INPUT FORM
elif st.session_state.page == "form":
    st.title("📋 Loan Application Form")

    with st.form("loan_form"):
        loan_amount = st.number_input("Loan amount requested ($)", min_value=0)
        loan_term = st.number_input("Term of the loan (months)", min_value=1)

        job_title = st.text_input("Job title")
        job_time = st.number_input("Years in current job", min_value=0)

        home_status = st.selectbox(
            "Home ownership status",
            ["RENT", "MORTGAGE", "OWN", "NONE"]
        )

        annual_income = st.number_input("Annual income ($)", min_value=0)

        loan_purpose = st.selectbox(
            "Purpose of the loan",
            ["car", "credit card", "debt consolidation", "education",
             "home improvement", "house", "major purchase", "medical",
             "moving", "other", "small business", "vacation", "wedding"]
        )

        total_debt = st.number_input("Total debt ($)", min_value=0)

        delinquencies = st.selectbox(
            "Any financial delinquencies in the last 2 years?",
            ["no", "yes"]
        )

        credit_score = st.number_input("Credit score", min_value=0, max_value=850)

        accounts = st.number_input("Number of credit cards & bank accounts", min_value=0)

        bankruptcy = st.selectbox(
            "Have you ever declared bankruptcy?",
            ["no", "yes"]
        )

        # File uploads
        st.markdown("### Upload Documents")
        bank_file = st.file_uploader("Upload your bank statements CSV", type="csv")
        cc_file = st.file_uploader("Upload your credit card transactions CSV", type="csv")

        submitted = st.form_submit_button("Submit Application")

    if submitted:
        if not bank_file or not cc_file:
            st.warning("⚠️ Please upload both bank and credit card statements.")
        else:
            st.session_state.loan_data = {
                "loan_amount": loan_amount,
                "loan_term": loan_term,
                "job_title": job_title,
                "job_tenure": job_time,
                "home_status": home_status,
                "annual_income": annual_income,
                "loan_purpose": loan_purpose,
                "total_debt": total_debt,
                "delinquencies": delinquencies,
                "credit_score": credit_score,
                "accounts": accounts,
                "bankruptcy": bankruptcy,
            }
            st.session_state.bank_df = pd.read_csv(bank_file)
            st.session_state.cc_df = pd.read_csv(cc_file)
            go_to("banker_comments")

elif st.session_state.page == "banker_comments":
    st.title("📝 Banker Comments")
    st.markdown("Please provide your comments regarding the loan application. " \
    "This is your opportunity to add any insights or considerations that may not be captured in the application data.")

    with st.form("comments_form"):
        comments = st.text_area("Comments", height=200)
        risk_level = st.selectbox(
            "Please select the bank's current risk tolerance",
            ["None", "Low", "Medium", "High", "N/A"]
        )
        submitted = st.form_submit_button("Submit Comments")

    if submitted:
        st.session_state.banker_comments = comments
        st.session_state.risk_level = risk_level
        go_to("processing")

# PROCESSING PAGE
elif st.session_state.page == "processing":
    st.title("⚙️ Processing Your Request...")
    st.markdown("Our agents are reviewing the data and making a decision. Please wait...")

    with st.spinner("Analyzing profiles..."):
        time.sleep(3)  # simulate processing
        # >>> Here you would call your pipeline, e.g.
        # decision = run_multiagent_pipeline(st.session_state.loan_data, st.session_state.bank_df, st.session_state.cc_df)
        # For now, mock it:
        decision = {"approved": True, "reason": "Stable income and good credit score."}

    st.session_state.decision = decision
    go_to("result")

# RESULT PAGE
elif st.session_state.page == "result":
    st.title("✅ Loan Decision")
    decision = st.session_state.decision

    if decision["approved"]:
        st.success("🎉 Your loan has been **Approved**!")
    else:
        st.error("❌ Your loan has been **Denied**.")

    # Here we need to print the report generated from the previous step. 
    # There must also be a comments box allowing for an ammended report, and an option to accept/download/email the report.
    st.subheader("Reasoning")
    st.write(decision["reason"])
    user_directives = st.text_area("Add your comments or directives:")


    if st.button("🔄 Start Over"):
        st.session_state.clear()
        st.session_state.page = "welcome"
