import os
from dotenv import load_dotenv
from openai import OpenAI

# Load the values from the .env file
load_dotenv()

# Read the API key
api_key = os.getenv("OPENROUTER_API_KEY")

# Stop if the API key is missing
if not api_key:
    raise ValueError(
        "OPENROUTER_API_KEY is missing from your .env file"
    )

# Connect to OpenRouter
client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

# Read the model name from .env
MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "openai/gpt-4o-mini"
) 
