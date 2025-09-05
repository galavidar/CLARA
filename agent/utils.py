import json
from config import AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, API_VERSION, CHAT_DEPLOYMENT, HF_API_KEY, USE_HF_MODELS
from langchain_openai import AzureChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os
import re

def normalize_json(obj):
    """
    Take either a JSON string or a dict-like object and return a proper dict/list.
    - If obj is a dict or list: return as-is.
    - If obj is a string:
        * Try json.loads directly.
        * If that fails, try to extract and parse JSON from fenced code blocks like ```json ... ```.
        * If that fails, try to extract the first balanced JSON object/array substring and parse it.
        * If all parsing fails, return {"raw_text": original_string} (unchanged fallback).
    - For other types: serialize with json.dumps then json.loads (unchanged), else {"raw_text": str(obj)}.
    """

    # Case 1: Already a dict (or list) → unchanged behavior
    if isinstance(obj, (dict, list)):
        return obj

    # Helper: try to parse the first valid JSON found in fenced code blocks
    def _parse_from_fences(s: str):
        # Capture contents inside ```...``` or ```json ...```
        blocks = re.findall(
            r"```(?:json|javascript|js|ts|python)?\s*(.*?)\s*```",
            s,
            flags=re.IGNORECASE | re.DOTALL,
        )
        for block in blocks:
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue
        return None

    # Helper: try to extract the first balanced {...} or [...] region and parse it
    def _parse_first_balanced_json(s: str):
        s = s.strip()
        # Find the earliest '{' or '['
        starts = [p for p in (s.find("{"), s.find("[")) if p != -1]
        if not starts:
            return None
        start = min(starts)
        opening = s[start]
        closing = "}" if opening == "{" else "]"

        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == opening:
                    depth += 1
                elif ch == closing:
                    depth -= 1
                    if depth == 0:
                        candidate = s[start : i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            return None
        return None

    # Case 2: It's a string
    if isinstance(obj, str):
        s = obj.strip()

        # 2a) direct parse
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass

        # 2b) parse from fenced code blocks
        parsed = _parse_from_fences(s)
        if parsed is not None:
            return parsed

        # 2c) parse first balanced JSON object/array found in the text
        parsed = _parse_first_balanced_json(s)
        if parsed is not None:
            return parsed

        # 2d) fallback (unchanged)
        return {"raw_text": obj}

    # Case 3: Unexpected type → serialize then load back (unchanged)
    try:
        return json.loads(json.dumps(obj))
    except Exception:
        return {"raw_text": str(obj)}


def get_model(open_ai_model:str="gpt-4o-mini"):
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
        model = open_ai_model
        )
    
    return chat_model

def normalize_markdown(obj):
    """
    Normalize input into a Markdown string safe for Streamlit's st.markdown().
    
    Handles:
    - Already a string: returned as-is if valid Markdown.
    - Strings with fenced code blocks (```markdown ... ```): unwraps the block.
    - Dicts with keys like {"type": "markdown", "content": "..."}: extracts content.
    - Lists of strings/dicts: joins them with newlines.
    - Any other object: coerces to string.
    """

    # Case 1: Already a string
    if isinstance(obj, str):
        s = obj.strip()

        # If string looks like ```markdown ... ```
        match = re.match(r"```(?:markdown)?\s*(.*?)\s*```", s, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return s

    # Case 2: Dict-like input
    if isinstance(obj, dict):
        # Handle JSON-style "markdown objects"
        if "content" in obj:
            return str(obj["content"]).strip()
        if "markdown" in obj:
            return str(obj["markdown"]).strip()
        # Fallback: pretty-print dict as JSON inside a code block
        return "```json\n" + json.dumps(obj, indent=2, ensure_ascii=False) + "\n```"

    # Case 3: List-like input
    if isinstance(obj, (list, tuple)):
        parts = [normalize_markdown(x) for x in obj]
        return "\n\n".join(parts)

    # Case 4: Unexpected type → just string
    return str(obj)
