import os
from dotenv import load_dotenv
from pathlib import Path

# Loads values from .env into os.environ
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",") if os.getenv("ALLOWED_ORIGINS") else ["*"]

# Markdown context file path
MARKDOWN_CONTEXT_FILE = os.getenv("output_document.md")
