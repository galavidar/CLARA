from typing import Dict, Any, List
from pydantic import Field
import logging, json
from langchain.chains.base import Chain
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnableLambda
from agents.behavioural_agent import extract_behavioural_features
from agents.evaluator_agent import evaluate_outputs
from agents.report_generator_agent import generate_loan_report
from agents.decision_agent import decide
import pandas as pd

logging.basicConfig(filename="./outputs/loan_chain.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")

class LoanEligibilityChain(Chain):
    """
    LangChain Chain that orchestrates inline micro-agents 
    with shared memory and evaluator-driven flow control.
    """
    max_retries: int = Field(default=3, description="Maximum number of retries before fallback")
    behavioral_agent: RunnableLambda = Field(default=None)
    decision_agent: RunnableLambda = Field(default=None)
    evaluator_agent: RunnableLambda = Field(default=None)
    report_agent: RunnableLambda = Field(default=None)
    verbose: bool = Field(default=False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Shared memory across agents
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="input_data",
            output_key="final_report"
        )

        self.verbose = kwargs.get("verbose", True)

        # Build microagents
        self._build_agents()

    @property
    def input_keys(self) -> List[str]:
        return ["input_data", "bank_csv", "card_csv"]

    @property
    def output_keys(self) -> List[str]:
        return ["final_report", "decision"]

    @property
    def memory_variables(self) -> List[str]:
        return ["chat_history"]

    def _debug(self, label: str, state: dict):
        """Log current pipeline state, excluding bulky objects like memory/history."""
        if self.verbose:
            filtered = {
                k: v for k, v in state.items()
                if k not in ["chat_history", "memory", "bank_csv", "card_csv"]
            }
        chat_history_str = ""
        if "chat_history" in state and state["chat_history"]:
            messages = []
            for msg in state["chat_history"]:
                sender = getattr(msg, "type", "unknown")  # 'human' or 'ai'
                content = getattr(msg, "content", str(msg))
                messages.append(f"[{sender}] {content}")
            chat_history_str = "\n".join(messages)
            logging.info(f"\n--- {label} ---\n{json.dumps(filtered, indent=2)}\n")
            if chat_history_str:
                logging.info(f"--- Chat History ---\n{chat_history_str}\n")


    # ---- Main call method ----
    def _call(self, inputs: Dict[str, Any], run_manager=None) -> Dict[str, Any]:
        """Execute evaluator-driven pipeline with shared memory."""

        # Load shared history
        memory_vars = self.memory.load_memory_variables({})
        self.memory.chat_memory.add_user_message(
            f"Loan application: {json.dumps(inputs['input_data'], indent=2)}"
        )

        # Initialize shared state
        state = {
            "input_data": inputs["input_data"],
            "bank_csv": inputs["bank_csv"],
            "card_csv": inputs["card_csv"],
            "retry_count": 0,
            "current_step": "behavioral",
            "memory": self.memory,
            "chat_history": memory_vars.get("chat_history", [])
        }
        self._debug("Initial State", state)
        while state["retry_count"] <= self.max_retries:
            if state["current_step"] == "behavioral":
                state = self.behavioral_agent.invoke(state)
                self._debug("After Behavioral Agent", state)
                self.memory.chat_memory.add_ai_message(
                    f"Behavioral Analysis completed: {len(state['behavioral_profiles'])} profiles"
                )
                state["current_step"] = "decision"

            elif state["current_step"] == "decision":
                state = self.decision_agent.invoke(state)
                self._debug("After Decision Agent", state)
                self.memory.chat_memory.add_ai_message(
                    f"Decision outcome: {json.dumps(state['decision'], indent=2)}"
                )
                state["current_step"] = "evaluator"

            elif state["current_step"] == "evaluator":
                state = self.evaluator_agent.invoke(state)
                self._debug("After Evaluator Agent", state)
                eval_action = state["evaluation_result"]["action"]

                self.memory.chat_memory.add_ai_message(
                    f"Evaluator: {eval_action} - {state['evaluation_result'].get('feedback', '')}"
                )

                if eval_action == "approve":
                    final = self.report_agent.invoke(state)
                    self.memory.save_context(
                        {"input_data": json.dumps(inputs["input_data"], indent=2)},
                        {"final_report": json.dumps(final["final_report"], indent=2)}
                    )
                    return {
                        "final_report": final.get("final_report"),
                        "decision": final.get("decision", state.get("decision"))['decision']
                        }

                elif eval_action == "revise_profiles":
                    state["current_step"] = "behavioral"
                    state["retry_count"] += 1
                    continue

                elif eval_action == "revise_decision":
                    state["current_step"] = "decision"
                    state["retry_count"] += 1
                    continue
                
                elif eval_action == "revise_terms":
                    state["current_step"] = "decision"
                    state["retry_count"] += 1
                    continue

        final = self.report_agent.invoke(state)
        self._debug("Final Report", final)
        self.memory.save_context(
            {"input_data": json.dumps(inputs["input_data"], indent=2)},
            {"final_report": json.dumps(final["final_report"], indent=2)}
        )
        return {
        "final_report": final.get("final_report"),
        "decision": final.get("decision", state.get("decision"))['decision']
    }

    # ---- Build inline agents ----
    def _build_agents(self):
        self.behavioral_agent = RunnableLambda(
            lambda state: (
                lambda res: {**state, "behavioral_profiles": res[0], "user_features": res[1]}
            )(extract_behavioural_features(
                bank=state["bank_csv"],
                card=state["card_csv"],
                supervisor_comments=state.get("evaluation_result", {}).get("comments")
                if state.get("retry_count", 0) > 0 else None
            ))
        )

        self.decision_agent = RunnableLambda(
            lambda state: {
                **state,
                "decision": decide(
                    loan_data=state["input_data"],
                    user_features=state["user_features"],
                    behavioural_profiles=state["behavioral_profiles"],
                    evaluator_comments=state.get("evaluation_result", {}).get("comments")
                )
            }
        )

        self.evaluator_agent = RunnableLambda(
            lambda state: {
                **state,
                "evaluation_result": evaluate_outputs(
                    loan_data=state["input_data"],
                    user_features=state["user_features"],
                    profiles=state["behavioral_profiles"],
                    interest_rate=state["decision"]["interest_rate"],
                    loan_term=state["decision"]["term"],
                    decision=state["decision"],
                    risk_score=state["decision"]["risk_score"],
                    user_directives=state.get("user_directives", ''),
                    risk_tolerance=state.get("risk_tolerance", "medium")
                )
            }
        )

        self.report_agent = RunnableLambda(
            lambda state: {
                "final_report": generate_loan_report(
                    loan_data=state["input_data"],
                    profiles=state["behavioral_profiles"],
                    user_features=state["user_features"],
                    interest_rate=state["decision"]["interest_rate"],
                    loan_term=state["decision"]["term"],
                    decision=state["decision"],
                    risk_score=state["decision"]["risk_score"],
                    user_directives=state.get("user_directives", ''),
                )
            }
        )

def test():
    bank_data = pd.read_csv("./dev/data/synthetic_users/bank_user_0001.csv")
    card_data = pd.read_csv("./dev/data/synthetic_users/card_user_0001.csv")
    loan_data={
                "loan_amount": 10000,
                "loan_term": 36,
                "job_title": 'Doctor',
                "job_tenure": 10,
                "home_status": 'OWN',
                "annual_income": 120000,
                "loan_purpose": 'car',
                "total_debt": 50000,
                "delinquencies": 'no',
                "credit_score": 750,
                "accounts": 5,
                "bankruptcy": 'no',
            }
    chain = LoanEligibilityChain(max_retries=3)

    result = chain.invoke({
        "input_data": loan_data,
        "bank_csv": bank_data,
        "card_csv": card_data,
        "verbose": True
    })

    print("=== Final Report ===")
    print(result["final_report"])
    print("=== Decision ===")
    print(result["decision"])



# Usage Examples
if __name__ == "__main__":
    test()