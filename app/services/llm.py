import requests
from pathlib import Path
# from langchain_huggingface import HuggingFaceEmbeddings
from ..config import GROQ_API_KEY, GROQ_MODEL
import os
markdown_context = os.getenv("output_document.md")

def generate_from_groq(prompt: str, markdown_context: str = "", max_tokens: int = 512) -> str:
    """
    Calls Groq Cloud API for text generation.
    Automatically includes markdown context in the prompt.
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing in .env file")

    # Defensive: ensure markdown_context is a string (caller may pass None)
    if markdown_context is None:
        markdown_context = ""

    # Prepend markdown context to prompt if available
    try:
        has_context = bool(markdown_context.strip())
    except Exception:
        has_context = False

    if has_context:
        full_prompt = f"""SYSTEM: You are Deepankar's portfolio assistant. Use the following document context when answering questions try to answer properly from the given context. If the answer is not found in the context, say 'I don't know'.:

{markdown_context}

User Query: {prompt}"""
    else:
        full_prompt = prompt

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": full_prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0
    }

    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        r = requests.post(url, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()

        if "choices" in data and len(data["choices"]) > 0:
            content = data["choices"][0]["message"]["content"]
            return content.strip()
        
        raise RuntimeError(f"Unexpected Groq response: {data}")

    except requests.exceptions.Timeout:
        raise RuntimeError("Groq generation request timed out") from None
    except requests.exceptions.RequestException as e:
        msg = str(e)
        if e.response is not None:
             msg += f" | Body: {e.response.text}"
        raise RuntimeError(f"Groq request failed: {msg}") from e
