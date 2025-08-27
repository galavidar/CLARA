import json
from config import AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, API_VERSION, CHAT_DEPLOYMENT, HF_API_KEY, USE_HF_MODELS
from langchain_openai import AzureChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os

def normalize_json(obj):
    """
    Take either a JSON string or a dict-like object and return a proper dict.
    Always safe for downstream use.
    """
    # Case 1: Already a dict (or list)
    if isinstance(obj, (dict, list)):
        return obj

    # Case 2: It's a string
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except json.JSONDecodeError:
            # String but not valid JSON → wrap in dict
            return {"raw_text": obj}

    # Case 3: Unexpected type → serialize then load back
    try:
        return json.loads(json.dumps(obj))
    except Exception:
        return {"raw_text": str(obj)}

def get_model():
    """
    Determines and initiates the model to be used
    """
    if USE_HF_MODELS:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_API_KEY
        llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-120b",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        provider="auto",  # let Hugging Face choose the best provider for you
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