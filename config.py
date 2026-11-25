"""
Model configuration for the SDLC multi-agent pipeline.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# MODEL PROVIDER CONFIGURATION
# Options: "gemini", "ollama", "openrouter", "groq"
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "gemini")


def get_model():
    """Return the appropriate model instance based on MODEL_PROVIDER."""
    if MODEL_PROVIDER == "gemini":
        return "gemini-3.1-flash-lite"
    elif MODEL_PROVIDER == "ollama":
        from google.adk.models.lite_llm import LiteLlm

        return LiteLlm(
            model="ollama_chat/qwen3.5:2b",
            extra_body={"think": False},
        )
    elif MODEL_PROVIDER == "openrouter":
        from google.adk.models.lite_llm import LiteLlm

        return LiteLlm(
            model="openrouter/qwen/qwen-2.5-72b-instruct",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
    elif MODEL_PROVIDER == "groq":
        from google.adk.models.lite_llm import LiteLlm

        return LiteLlm(
            model="groq/qwen/qwen3.8-27b",
            api_key=os.environ.get("GROQ_API_KEY"),
        )
    else:
        return "gemini-3.1-flash-lite"
