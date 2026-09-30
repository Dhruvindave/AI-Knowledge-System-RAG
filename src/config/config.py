# src/config.py

import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)

GEMINI_FALLBACK_MODELS = tuple(
    model.strip()
    for model in os.getenv(
        "GEMINI_FALLBACK_MODELS",
        "gemini-2.0-flash,gemini-1.5-flash"
    ).split(",")
    if model.strip()
)

MAX_LLM_RETRIES = int(os.getenv("MAX_LLM_RETRIES", 3))

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2"
)


CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", 5))