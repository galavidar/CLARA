# loads environment variables from .env file and concentrates hard-coded configuration parameters for API

from dotenv import load_dotenv
import os

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("API_KEY")
CHAT_DEPLOYMENT = "team7-gpt4o"
EMBEDDING_DEPLOYMENT = "team7-embedding"
AZURE_OPENAI_ENDPOINT = "https://096290-oai.openai.azure.com"
API_VERSION = "2023-05-15"

VECTOR_DB_URL = "https://your-vector-db-url.com"

HF_API_KEY = os.getenv("HF_API")
USE_HF_MODELS = True # change this to use Hugging Face models
COUNT_TOKENS = False # change this to disbale token counting


# paths to log files
TOKEN_LOG_FILE = "./outputs/total_tokens.txt"
BEHAVIOURAL_LOG_FILE = "./outputs/behavioural_agent_responses.txt"
REPORTS_DIR = "./outputs/reports/"
EVALUATOR_LOG_FILE = "./outputs/evaluator_agent_responses.txt"
DECISIONS_LOG_FILE = "./outputs/decision_agent_responses.txt"