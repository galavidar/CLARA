from datetime import datetime
import numpy as np
import os
import json
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import AzureChatOpenAI
from config import AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, API_VERSION, CHAT_DEPLOYMENT, HF_API_KEY, USE_HF_MODELS, COUNT_TOKENS
from token_logger import log_tokens

def run_agent():
    pass

def decide(loan_data, user_features, behavioural_profiles, evaluator_comments=None):
    # placeholder sample 
    return {'decision': 'approved', 'interest_rate': 0.12, 'term': 36, 'reason': '7 similar examples in the DB that the loans were approved and they did not default', 'risk_score': 0.3}

def get_model():
    """
    Determines and initiates the model to be used
    """
    if USE_HF_MODELS:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_API_KEY
        llm = HuggingFaceEndpoint(
            repo_id="openai/gpt-oss-120b",  # free OSS model
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
            provider="auto",
        )
        chat_model = ChatHuggingFace(llm=llm)
    
    else:
        chat_model = AzureChatOpenAI(
        azure_deployment=CHAT_DEPLOYMENT,
        azure_endpoint = AZURE_OPENAI_ENDPOINT,
        api_key = AZURE_OPENAI_API_KEY,
        openai_api_type = "azure",
        openai_api_version = API_VERSION,
        model = 'gpt-4o-mini'
        )
    
    return chat_model

def test():
    pass

if __name__ == "__main__":
    test()