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

