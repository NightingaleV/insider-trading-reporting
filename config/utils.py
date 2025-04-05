from pathlib import Path
import os
from .paths import ROOT_DIR

def load_env_file(file_path: str) -> None:
    """Manually loads environment variables from a .env file using pathlib."""
    env_path = ROOT_DIR / file_path

    if not env_path.is_file():
        raise FileNotFoundError(f"Environment file '{file_path}' not found.")

    with env_path.open("r") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):  # Ignore empty lines and comments
                continue

            key, sep, value = line.partition("=")
            key, value = key.strip(), value.strip()

            if not key or not sep:  # Ensure it's a valid key-value pair
                continue

            os.environ[key] = value  # Store as string (os.environ only supports str)