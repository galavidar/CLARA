# Prompt templates for various tasks

import json
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate

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

def build_behavioural_json_prompt(user_features, rule_profiles, supervisor_comments=None):
    """
    Builds a LangChain chat prompt that instructs the model to validate or infer
    financial profiles, using structured JSON input.
    """
    messages = [
        SystemMessagePromptTemplate.from_template(
            "You are a financial profile inference agent. "
            "Your task is to analyze an applicant's financial data and infer profiles based on provided features and rule-based profiles."
            "Your response should be a valid dictionary object with the following schema:\n"
            "- profiles: mapping of profile -> 0/1\n"
            "- reasoning: mapping of profile -> short explanation\n"
        ),
        HumanMessagePromptTemplate.from_template(
            """Task: classify_financial_profiles

                Profiles definitions:
                {profiles_definitions}

                Applicant data:
                {user_features}

                Rule-based profiles:
                {rule_profiles}

                Instructions:
                - Validate each rule-based profile decision (0=No, 1=Yes), overriding if necessary.
                - Infer non-rule-based profiles directly.
                - Provide reasoning for each profile.
                - Respond ONLY with valid dictionary in the specified schema.

                Output schema:
                - profiles: mapping of profile -> 0/1
                - reasoning: mapping of profile -> short explanation
                """
        )]
    
    if supervisor_comments:
        messages.append(
            AIMessagePromptTemplate.from_template(
                "\n\n{previous_response}"
            )
        )
        messages.append(
            HumanMessagePromptTemplate.from_template(
                "Supervisor feedback:\n\nAction: {action}\nComments: {comments}\n\n"
                "Revise your response accordingly, keeping strictly to the JSON schema."
            )
        )
    prompt = ChatPromptTemplate.from_messages(messages)

    kwargs = {
        "profiles_definitions": json.dumps(PROFILE_DEFS, indent=2),
        "user_features": json.dumps(user_features, indent=2),
        "rule_profiles": json.dumps(rule_profiles, indent=2),
    }

    if supervisor_comments:
        kwargs.update({
            "previous_response": json.dumps(supervisor_comments.get("prev_response_behavioral", {}), indent=2),
            "action": supervisor_comments.get("action", ""),
            "comments": supervisor_comments.get("comments", "")
        })

    return prompt.format(**kwargs)

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
         - Binary behavioural profiles: {profiles}
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
         5. Calculate the required monthly installments (with interest) to include in the report.
         6. Generate a full professional report suitable for bank internal documentation.
         7. If provided, pay attention to the user directives when creating the report.
         """
        )
    ])
    return prompt.format(
        loan_data=json.dumps(loan_data, indent=2),
        features=json.dumps({k: v for k, v in features.items() if k != "last3_category_shares"}, indent=2),
        profiles=json.dumps(profiles.get("profiles", {}), indent=2),
        profile_reasons=json.dumps(profiles.get("reasoning", {}), indent=2),
        interest_rate=interest_rate,
        loan_term=loan_term,
        decision=json.dumps(decision, indent=2),
        risk_score=risk_score,
        user_directives=user_directives or "No additional directions provided.",
    )

def build_evaluation_prompt(loan_data, features, profiles, rate, term, risk_score, decision, user_directives=None, bank_risk_tolerance="medium"):
    """
    Build a LangChain ChatPromptTemplate for the evaluator agent.
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are the Evaluator Agent (Bank Manager). "
            "You oversee all other agents. "
            "Your job is to ensure the behavioral profiling is logical, "
            "and the loan decision is fair, consistent with the bank's interests, "
            "and aligned with the bank's stated goals and risk tolerance."
        ),
        HumanMessagePromptTemplate.from_template(
            """
            Task: Assess agent outputs for consistency and quality.

            Inputs:
            - Loan application data: {loan_data}
            - Summary of key financial metrics of applicant: {features}
            - Binary behavioural profiles: {profiles}
            - Reason for profile assignments: {profile_reasons}
            - Interest rate: {interest_rate}
            - Loan term: {loan_term}
            - Risk score: {risk_score}
            - Loan decision + justification + retrieval evaluation: {decision}
            - Bank risk tolerance: {bank_risk_tolerance}
            - User directives (if any): {user_directives}

            Instructions:
            1. Check if the behavioural profiling is logical and consistent with applicant data.
            2. Check if the loan decision is fair and aligns with both the profiles and risk tolerance.
            3. Check the evaluation of the retrieval portion of the decision. Consider the scores and reasons of each metric.
            4. Check if the altering the interest rate or term would be beneficial.
            5. If profiling is problematic → return: {{"action": "revise_profiles", "comments": "..."}}
            6. If the decision is problematic → return: {{"action": "revise_decision", "comments": "..."}}
            7. If the decision is acceptable but the interest rate or term should be changed → return: {{"action": "revise_terms", "comments": "..."}}
            8. If all are acceptable → return: {{"action": "approve", "comments": "... (short justification)"}}

            Output schema: Always respond in a dictionary with keys: action, comments.
            """

        )
    ])

    return prompt.format(
        loan_data=json.dumps(loan_data, indent=2),
        features=json.dumps({k: v for k, v in features.items() if k != "last3_category_shares"}, indent=2),
        profiles=json.dumps(profiles.get("profiles", {}), indent=2),
        profile_reasons=json.dumps(profiles.get("reasoning", {}), indent=2),
        interest_rate=rate,
        loan_term=term,
        risk_score=risk_score,
        decision=json.dumps(decision, indent=2),
        bank_risk_tolerance=bank_risk_tolerance,
        user_directives=user_directives or "No additional directives provided."
    )

def build_decision_prompt(loan_data, features, profiles, rate, term, risk_score, retrieved_cases=None, supervisor_comments=None):
    """
    Builds a LangChain chat prompt that instructs the model to decide on a loan application.
    """
    messages = [
        SystemMessagePromptTemplate.from_template(
            "You are a financial decision-making agent."
            "Your task is to analyze the applicant's financial data and profiling and make a decision on the loan application. You are provided with a 0-1 risk score from a machine learning model where 0 is risk-free."
            "Additionally, you are provided with a predicted interest rate and a applicant-requested loan-term. You may change these values if needed."
            "You must also reference similar past loan cases retrieved from a database, including the relevant case IDs. "
            "Your response should be a valid dictionary object with the following schema:\n"
            "- decision: a final decision -> accepted/rejected \n"
            "- reason: motivation for decision -> short explanation including an analysis or aggregation of relevant past cases\n"
            "- interest_rate: the final interest rate -> float\n"
            "- loan_term: the final loan term -> int\n"
        ),
        HumanMessagePromptTemplate.from_template(
            """Task: decide on loan application

                Inputs:
                - Loan application data: {loan_data}
                - Summary of key financial metrics of applicant: {features}
                - Binary behavioural profiles: {profiles}
                - Reason for profile assignments: {profile_reasons}
                - Interest rate (predicted, allowed to change): {interest_rate}
                - Loan term (by applicant request): {loan_term}
                - Risk score (0-1, ML model output): {risk_score}
                - Retrieved similar cases (for reference): {retrieved_cases}

                Instructions:
                - Analyse the applicant's data, the provided profiles and the risk score to make a decision.
                - Consider retrieved past cases as precedent when motivating your decision.
                - When referencing past cases, include the relevant case IDs in square brackets.
                - Consider the potential impact of the loan terms on the applicant's financial situation.
                - Evaluate if the predicted interest rate aligns with the applicant's risk profile.
                - Evaluate if the term length is appropriate given the applicant's financial situation.
                - Recommend adjustments to the loan terms if they do not align with the applicant's risk profile or financial situation.

                Output schema:
                "- decision: a final decision -> accepted/rejected \n"
                "- reason: motivation for decision -> short explanation including references to relevant past cases\n"
                "- interest_rate: the final interest rate -> float\n"
                "- loan_term: the final loan term -> int\n"
                """
        )]
    
    if supervisor_comments:
        messages.append(
            AIMessagePromptTemplate.from_template(
                "\n\n{previous_response}"
            )
        )
        messages.append(
            HumanMessagePromptTemplate.from_template(
                "Supervisor feedback:\n\nAction: {action}\nComments: {comments}\n\n"
                "Revise your response accordingly, keeping strictly to the JSON schema."
            )
        )
    prompt = ChatPromptTemplate.from_messages(messages)

    kwargs = {
        "loan_data": json.dumps(loan_data, indent=2),
        "features": json.dumps({k: v for k, v in features.items() if k != "last3_category_shares"}, indent=2),
        "profiles": json.dumps(profiles.get("profiles", {}), indent=2),
        "profile_reasons": json.dumps(profiles.get("reasoning", {}), indent=2),
        "interest_rate": rate,
        "loan_term": term,
        "risk_score": risk_score,
        "retrieved_cases": json.dumps(retrieved_cases or [], indent=2)
    }

    if supervisor_comments:
        kwargs.update({
            "previous_response": json.dumps(supervisor_comments.get("previous_response_decision", {}), indent=2),
            "action": supervisor_comments.get("action", ""),
            "comments": supervisor_comments.get("comments", "")
        })

    return prompt.format(**kwargs)

def build_rag_eval_prompt(criteria: str, query: str, prediction: str, reference: str = ""):
    """
    Build a structured prompt for evaluation with explicit criteria.
    """
    messages = [SystemMessagePromptTemplate.from_template(
        """
        You are an evaluation agent. Your task is to check whether a given model's response 
        satisfies the criterion: {criteria}, relative to data retrieved from a database.\n
        Note that some fields in the model's response (risk score, interest rate), are calculated using an algorithm,
        and are given directly to the model. They may not match the database records exactly.
        "Respond with a JSON object of the form:\n"
        "score": 0 to 1, "reasoning": "short explanation"
        """), 
        HumanMessagePromptTemplate.from_template(
        """
        Query: {query}

        Retrieved Context (Important: these fields do not contain a risk score): {reference}

        Model Response: {prediction}
        """
    )]
    prompt = ChatPromptTemplate.from_messages(messages)

    return prompt.format(**{
        "criteria": criteria,
        "query": query,
        "reference": reference,
        "prediction": prediction
    })
