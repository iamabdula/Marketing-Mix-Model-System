import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "./data")
MODEL_DIR = os.getenv("MODEL_DIR", "./models")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Default parameters for feature engineering
DEFAULT_DECAY_FACTORS = {
    "tv_spend": 0.5,
    "radio_spend": 0.4,
    "social_media_spend": 0.3,
    "search_spend": 0.4,
    "print_spend": 0.2,
    "outdoor_spend": 0.3,
}

DEFAULT_ALPHA_PARAMS = {
    "tv_spend": 1.0,
    "radio_spend": 1.0,
    "social_media_spend": 1.0,
    "search_spend": 1.0,
    "print_spend": 1.0,
    "outdoor_spend": 1.0,
}

DEFAULT_GAMMA_PARAMS = {
    "tv_spend": 1.0,
    "radio_spend": 1.0,
    "social_media_spend": 1.0,
    "search_spend": 1.0,
    "print_spend": 1.0,
    "outdoor_spend": 1.0,
}
