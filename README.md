# 🤖 CLARA – Credit & Loan Automated Risk Assessment

CLARA (**C**redit & **L**oan **A**ssessment with **R**isk **A**gents) is a **modular multi-agent system** designed to aid financial institutions in their assessment of loan applications, driving decision-making and reducing cumbersome paperwork.

It combines **LLM-driven agents** and **ML models** to:
- Predict fair loan interest rates
- Categorize borrower risk
- Analyze behavioral spending and repayment patterns
- Generate transparent reports to explain credit decisions

CLARA can run either with **paid LLMs** or with **free open-source models** via Hugging Face.

## 🧩 Multi-Agent System Architecture

The pipeline is designed as a **sequence of specialized agents**, each responsible for one part of the loan assessment:

- **💳 Behavioral Agent**  
  Analyzes spending, savings, repayment patterns and income stability from bank and card transaction data to build a behavioural financial profile of the applicant. Compares profiling to those of synthetically generated users to match behvaioural profiles to similar applicants (via a mini RAG-style comparison).

- **📈 Interest Rate Prediction Agent**  
  Predicts fair interest rates based on loan amount, term, and borrower profile, using a ML model trained on a dataset of over 2.5 million loan applications.

- **⚖️ Risk Categorization Agent**  
  Assigns a risk score to the applicant using a ML model trained on the same dataset as above. 

- **✅ Decision Agent**  
  Decides whether to approve, reject, or modify loan terms based on risk and behavior. Utilizes RAG to aid decision making by comparing to similar cases in a Vector DB with over 3GB of historical loan data.

- **🕵️‍♂️ RAG Validation Agent**
  Validates the decision agent's conlcusions and assigns scores for the answer's faithfulness, relevance and correctness, relative to the retrieved data.  

- **🔍 Evaluator Agent**  
Double-checks the profiling, decisions, and reasoning requests revisions to any part of the pipeline, if needed, ensuring consistency and alignment with the financial instution's specific requests and risk tolerance.

- **📝 Report Generator Agent**  
  Produces a **human-readable summary report** of the full analysis, reducing the paperwork overhead required in processing loan applications. Allows for the user to request ammendments to the report. 

The agents communicate through a **LangChain pipeline**, making the process iterative and auditable, and each agent's outputs are separately tracked and saved. The entire system is wrapped in a streamlit app to make interaction simple, fast and user-friendly while the application and decision histories are also saved locally. Tokens are counted for each API call (but this can be turned off in the config.py file).

<sup>If you are here from the 096290 course, we outlined the revisions made from our proposal in [this document](https://technionmail-my.sharepoint.com/:b:/g/personal/galavidar_campus_technion_ac_il/EXbRPNkNtvxIjIGEGIglOAUBHyEq2vXzapObPMAS9wOOzA?e=YknCBm). </sup>

## ⚙️ Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/galavidar/CLARA.git
   cd CLARA
   ```

2. **Install dependencies**  
   Make sure you are using Python 3.11.
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys**  
   Create a `.env` file inside the `agents/` directory:
   ```ini
   API_KEY=your-llm-api-key
   ```
   
   For OpenAIAzure → use `API_KEY`.
   
   For the free version → instead provide:
   ```ini
   HF_API=your-huggingface-token
   ```
   and set `USE_HF_MODELS = True` in `config.py`.

## ▶️ Running CLARA

Run the Streamlit app:
```bash
streamlit run clara_app.py
```

Then follow the on-screen prompts to upload a loan application and view the results.

## 📊 Data

Synthetic financial data (CSV bank & card statements) is included in:
```bash
dev_data/data/synthetic_users/
```

Each file simulates a user's financial behavior.  
Real loan data is available from the (Lending Club dataset)[https://www.kaggle.com/datasets/wordsforthewise/lending-club] and was uploaded to our Vector DB. The data itself was too large to upload to the repo.
