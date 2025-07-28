import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "./data")
MODEL_DIR = os.getenv("MODEL_DIR", "./models")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
