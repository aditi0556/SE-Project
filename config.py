"""
Model configuration for the SDLC multi-agent pipeline.

Set USE_LOCAL_MODEL to True to use local Ollama (qwen3.5:2b).
Set USE_LOCAL_MODEL to False to use Gemini (gemini-3.8-flash) via Google API.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# TOGGLE THIS FLAG
#   True  → Ollama local model (qwen3.5:2b)
#   False → Gemini API (gemini-3.8-flash)
# ──────────────────────────────────────────────
USE_LOCAL_MODEL = False


def get_model():
    """Return the appropriate model instance based on the USE_LOCAL_MODEL flag."""
    if USE_LOCAL_MODEL:
        from google.adk.models.lite_llm import LiteLlm

        return LiteLlm(
            model="ollama_chat/qwen3.5:2b",
            extra_body={"think": False},
        )
    else:
        return "gemini-3.8-flash"
