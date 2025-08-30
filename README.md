# CLARA – Credit & Loan Automated Risk Assessment

CLARA (**C**redit & **L**oan **A**ssessment with **R**isk **A**gents) is a modular multi-agent system for analyzing loan applications, predicting interest rates, and categorizing borrower risk.  
It leverages synthetic and real-world financial data to train ML-driven agents that mimic decision-making pipelines used in financial institutions.

---

## 🚀 Agents pipeline

- **Interest Rate Prediction Agent** – predicts fair loan interest rate using ML models.
- **Risk Categorization Agent** – assigns a credit risk score based on loan application data.
- **Behavioural Agent** – analyzes user financial behavior (e.g., spending & repayment habits).
- **Decision Agent** – evaluates loan approval/rejection based on risk and financial profile.
- **Report Generator Agent** – creates human-readable financial risk assessment reports.
- **Evaluator Agent** – validates and compares decisions across agents.

---

## 📂 Repository Structure

```
CLARA-main/
│── app.py                     # Main entry point
│── clara_agents_pipeline.py   # Orchestration of agents
│── behavioural_agent.py       # Behavioural analysis agent
│── decision_agent.py          # Loan decision agent
│── risk_agent.py              # Risk categorization agent
│── evaluator_agent.py         # Evaluates agent outputs
│── report_generator_agent.py  # Generates financial reports
│── prompts.py                 # LLM prompt templates
│── config.py                  # Configuration file
│── utils.py                   # Utility functions
│── requirements.txt           # Dependencies
│
├── dev/
│   ├── Data preperation.ipynb
│   ├── data_generator.py
│   ├── interest_calculator.py
│   ├── risk_categorization_train.py
│   ├── risk_score.ipynb
│   └── data/
│       └── synthetic_users/   # Synthetic bank data (CSV)
│
└── README.md
```

---



## ▶️ Usage

Run the CLARA pipeline:


## 📊 Data

Synthetic user bank statements are included under:

```
dev/data/synthetic_users/
```

Each CSV file corresponds to a single user’s monthly financial behavior.

---

[Lending club dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

Several millions real loan data and outcomes


