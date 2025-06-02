import os
import pytz
from dotenv import load_dotenv
from pathlib import Path

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEVELOPMENT_MODE = True if ENVIRONMENT == "development" else False

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PACKAGE_DIR = Path(__file__).resolve().parent.parent

# Load Environment Variables
if DEVELOPMENT_MODE:
    env_file = ROOT_DIR / 'env' / 'development.env'
else:
    env_file = ROOT_DIR / 'env' / 'production.env'
load_dotenv(dotenv_path=env_file)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")
HUGGING_FACE_API_KEY = os.getenv("HUGGING_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
# STOCK_DATA_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")