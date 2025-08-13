from dotenv import load_dotenv
import os

load_dotenv()


API_KEY = os.getenv("API_KEY")

VECTOR_DB_URL = "https://your-vector-db-url.com"
CHAT_DEPLOYMENT = "team7-gpt4o"
EMBEDDING_DEPLOYMENT = "team7-embedding"
