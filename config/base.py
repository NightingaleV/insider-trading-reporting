import os
from config.utils import load_env_file

ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")

if ENVIRONMENT == "dev":
    load_env_file("env/dev.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")
HUGGING_FACE_API_KEY = os.getenv("HUGGING_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
# STOCK_DATA_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")