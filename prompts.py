import json
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

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

def build_behavioural_json_prompt(user_features, rule_profiles):
    """
    Builds a LangChain chat prompt that instructs the model to validate or infer
    financial profiles, using structured JSON input.
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a financial profile inference agent. "
            "Your task is to analyze user financial data and infer profiles based on provided features and rule-based profiles."
            "Your response should be a valid JSON object with the following schema:\n"
            "- profiles: mapping of profile -> 0/1\n"
            "- reasoning: mapping of profile -> short explanation\n"
        ),
        HumanMessagePromptTemplate.from_template(
            """Task: classify_financial_profiles

                Profiles definitions:
                {profiles_definitions}

                User data:
                {user_features}

                Rule-based profiles:
                {rule_profiles}

                Instructions:
                - Validate each rule-based profile decision (0=No, 1=Yes), overriding if necessary.
                - Infer non-rule-based profiles directly.
                - Provide reasoning for each profile.
                - Respond ONLY with valid JSON in the specified schema.

                Output schema:
                - profiles: mapping of profile -> 0/1
                - reasoning: mapping of profile -> short explanation
                """
        )
    ])

    return prompt.format(
        profiles_definitions=json.dumps(PROFILE_DEFS, indent=2),
        user_features=json.dumps(user_features, indent=2),
        rule_profiles=json.dumps(rule_profiles, indent=2)
    )


def build_loan_report_prompt(loan_data, profiles, features, interest_rate, loan_term, decision, risk_score, user_directives=None):
    """
    Build a LangChain ChatPromptTemplate for loan decision report generation.
    """

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("You are a financial reporting agent. You generate professional, bank-internal style documentation for loan applications." 
                                                  "Your task is to generate a report justifying the loan decision."),
        HumanMessagePromptTemplate.from_template(
            """
        Task: Generate a loan decision report.

         Inputs:
         - Loan data (from applicant): {loan_data}
         - Summary of key financial metrics of applicant: {features}
         - Behavioural profiles: {profiles}
         - Reason for profile assignments: {profile_reasons}
         - Interest rate: {interest_rate}
         - Loan term: {loan_term}
         - Risk score: {risk_score}
         - Final decision: {decision}
         - User directives (if any): {user_directives}

         Instructions:
         1. Summarize the applicant request.
         2. Integrate behavioural profile validation + reasoning.
         3. Include financial analysis with interest rate, risk score, and debt-to-income ratio.
         4. Provide a final decision with motivation.
         5. Generate a full professional report suitable for bank internal documentation.
         6. If provided, pay attention to the user directives when creating the report. 
         """
        )
    ])

    return prompt.format(
        loan_data=json.dumps(loan_data, indent=2),
        features=json.dumps(features.pop("last3_category_shares", {}), indent=2),
        profiles=json.dumps(profiles.get("profiles", {}), indent=2),
        profile_reasons=json.dumps(profiles.get("reasoning", {}), indent=2),
        interest_rate=interest_rate,
        loan_term=loan_term,
        decision=json.dumps(decision, indent=2),
        risk_score=risk_score,
        user_directives=user_directives or "No additional directions provided.",
    )

