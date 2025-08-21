from dotenv import load_dotenv
import os

load_dotenv()

HF_API_KEY = os.getenv("HF_API")
AZURE_OPENAI_API_KEY = os.getenv("API_KEY")
CHAT_DEPLOYMENT = "team7-gpt4o"
EMBEDDING_DEPLOYMENT = "team7-embedding"
AZURE_OPENAI_ENDPOINT = "https://096290-oai.openai.azure.com"
API_VERSION = "2023-05-15"

VECTOR_DB_URL = "https://your-vector-db-url.com"


TOKEN_LOG_FILE = "./tokens_count/total_tokens.txt"