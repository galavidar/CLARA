import streamlit as st
import time
import json
import pandas as pd
import uuid
from clara_agents_pipeline import LoanEligibilityChain

if "applications" not in st.session_state:
    st.session_state.applications = []

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
    CLARA will also draft your report and provide similar cases that back up your decision.
                
    Before we begin, you will need to input the applicant's request data, and upload their bank and credit card statements as .csv files, so make sure to have them handy!

    Click **Continue** to get started.
    """)
    if st.button("Continue ➡️"):
        go_to("form")


# INPUT FORM
elif st.session_state.page == "form":
    st.title("📋 Application Form")

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

        monthly_debt = st.number_input("Monthly debt ($)", min_value=0)

        delinquencies = st.selectbox(
            "Any financial delinquencies in the last 2 years?",
            ["no", "yes"]
        )

        credit_score = st.number_input("Credit score", min_value=0, max_value=850)

        accounts = st.number_input("Number of credit cards & bank accounts", min_value=0)

        bankruptcy = st.selectbox(
            "Has the applicant ever declared bankruptcy?",
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
                "monthly_debt": monthly_debt,
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
    st.markdown("CLARA is reviewing the data and making a decision. Please wait...")

    with st.spinner("Analyzing user data and loan history..."):
        chain = LoanEligibilityChain(max_retries=5)
        inputs = {
            "input_data": st.session_state.loan_data,
            "bank_csv": st.session_state.bank_df,
            "card_csv": st.session_state.cc_df,
        }

        # Run the full chain
        result = chain(inputs)

        # Save full state to allow report regeneration
        st.session_state.chain_state = result
        decision = json.loads(result["decision"])
        outcome = decision['decision']
        st.session_state.generated_report = result["final_report"]

    user_id = str(uuid.uuid4())  # unique ID for each applicant
    application_record = {
        "user_id": user_id,
        "loan_data": st.session_state.loan_data,
        "decision": outcome,
        "LLM_motivation": decision["reason"]
    }
    st.session_state.applications.append(application_record)

    with open("./outputs/application_decisions_history.json", "a", encoding="utf-8") as f:
        json.dump(st.session_state.applications, f, indent=2, ensure_ascii=False)

    st.session_state.decision = decision
    go_to("result")

elif st.session_state.page == "result":
    st.title("✅ Loan Decision")
    decision = st.session_state.decision

    if decision["approved"]:
        st.success("🎉 The loan has been **Approved**!")
        st.write(f"- **Interest Rate**: {decision['interest_rate']}%")
        st.write(f"- **Loan Term**: {decision['loan_term']} months")
    else:
        st.error("❌ The loan has been **Denied**.")

    st.subheader("Reasoning")
    st.write(decision['reason'])

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Start Over"):
            st.session_state.clear()
            st.session_state.page = "welcome"

    with col2:
        if st.button("📄 Generate Report"):
            go_to("report")


# --- REPORT PAGE ---
elif st.session_state.page == "report":
    st.title("📑 Loan Report")
    st.subheader("Generated Report")
    st.write(st.session_state.generated_report)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Finish & Start New Application"):
            st.session_state.clear()
            st.session_state.page = "welcome"

    with col2:
        if st.button("✏️ Edit & Regenerate Report"):
            go_to("report_edit")


# --- REPORT EDIT PAGE ---
elif st.session_state.page == "report_edit":
    st.title("✏️ Edit & Regenerate Report")
    st.subheader("Existing Report")
    st.write(st.session_state.generated_report)

    st.subheader("Add Your Comments")
    st.session_state.user_comments = st.text_area(
        "Comments to adjust the report:",
        value=st.session_state.get("user_comments", ""),
        height=200
    )

    if st.button("🔄 Regenerate Report"):
        # Regenerate report using only the report_agent and saved chain state
        chain_state = st.session_state.chain_state
        # update user directives in the state for regeneration
        chain_state["user_directives"] = st.session_state.user_comments
        # Reinvoke only report agent
        chain = LoanEligibilityChain()  # reinstantiate to access report_agent
        report_output = chain.report_agent.invoke(chain_state)
        st.session_state.generated_report = report_output["final_report"]
        st.session_state.user_comments = ""  # clear after use
        go_to("report")