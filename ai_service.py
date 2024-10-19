import os
import sys
import logging
import time
import json
from typing import List, Optional, Dict
import requests
import openai

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

class AIService:
    """Performs AI-based sentiment analysis and predictions using AI"""

    def __init__(self, user=None, debug: bool = False):
        self.debug = debug
        if self.debug:
            logger.info("Initializing AIService")
        # Load environment variables
        self.load_environment_variables()
        self.max_tokens = 1000  # Adjusted to a reasonable number
        # Initialize variables for vector database
        self.index_name = None
        self.dimension = 768  # Default dimension for embeddings
        self.initialize_openai_client()

    def load_environment_variables(self):
        """
        Loads environment variables required for the application.
        """
        if self.debug:
            logger.info("Loading environment variables...")
        self.CLOUDFLARE_BASE_URL = "https://api.cloudflare.com/client/v4/accounts"
        self.CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        self.CLOUDFLARE_BASE_API_URL = (
            f"{self.CLOUDFLARE_BASE_URL}/{self.CLOUDFLARE_ACCOUNT_ID}"
        )
        self.CLOUDFLARE_USER_EMAIL = os.getenv("CLOUDFLARE_USER_EMAIL")
        self.CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
        self.CLOUDFLARE_API_KEY = os.getenv("CLOUDFLARE_API_KEY")
        self.CLOUDFLARE_MODEL_NAME = os.getenv(
            "CLOUDFLARE_MODEL_NAME", "@cf/meta/llama-3.2-1b-instruct"
        )
        self.CLOUDFLARE_EMBED_MODEL_NAME = "@cf/baai/bge-base-en-v1.5"

        if not all([self.CLOUDFLARE_API_TOKEN, self.CLOUDFLARE_ACCOUNT_ID]):
            logger.error("Missing required environment variables.")
            raise ValueError("Missing required environment variables.")
        if self.debug:
            logger.info("Environment variables loaded.")

    def initialize_openai_client(self):
        """
        Initializes the OpenAI client with Cloudflare Workers AI settings.
        """
        if self.debug:
            logger.info(
                "Initializing OpenAI client with Cloudflare Workers AI settings..."
            )
        openai.api_key = self.CLOUDFLARE_API_TOKEN
        openai.api_base = f"{self.CLOUDFLARE_BASE_API_URL}/ai/v1"
        if self.debug:
            logger.info(f"OpenAI API Base: {openai.api_base}")
            logger.info("OpenAI client initialized.")

    def make_cf_request(self, method, endpoint, data=None, json_data=None, headers=None, params=None, files=None):
        """
        Makes a request to the Cloudflare API with proper authentication.
        """
        if headers is None:
            headers = {}

        if self.CLOUDFLARE_API_KEY and self.CLOUDFLARE_USER_EMAIL:
            # Use API key and email
            headers["X-Auth-Key"] = self.CLOUDFLARE_API_KEY
            headers["X-Auth-Email"] = self.CLOUDFLARE_USER_EMAIL
        else:
            logger.error("No valid Cloudflare API credentials provided.")
            raise Exception("Authentication credentials missing.")

        # Set Content-Type if not set
        if "Content-Type" not in headers and (data or json_data):
            headers["Content-Type"] = "application/json"

        url = f"{self.CLOUDFLARE_BASE_API_URL}{endpoint}"

        if json_data:
            data = json.dumps(json_data)

        try:
            response = requests.request(
                method, url, params=params, data=data, headers=headers, files=files
            )
            response.raise_for_status()
            if self.debug:
                logger.debug(f"Request URL: {url}")
                logger.debug(f"Request Headers: {headers}")
                logger.debug(f"Response Status Code: {response.status_code}")
                logger.debug(f"Response Text: {response.text}")
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Cloudflare request failed: {e}")
            raise

    def _get_valid_json_response(self, system_content: str, user_content: str, expected_keys: List[str], max_tokens: int, temperature: float = 0, retries: int = 10, model: Optional[str] = None) -> Optional[Dict]:
        """
        Helper function to get a valid JSON response from the AI.

        Args:
            system_content (str): The system message defining the AI's role and response format.
            user_content (str): The user message containing the task.
            expected_keys (list): List of keys expected in the JSON response.
            max_tokens (int): Maximum number of tokens for the AI response.
            temperature (float): Sampling temperature for the AI.
            retries (int): Number of retry attempts.
            model (str): AI model to use.

        Returns:
            dict or None: Parsed JSON response if successful, else None.
        """
        system_message = {"role": "system", "content": system_content}
        user_message = {"role": "user", "content": user_content}

        for attempt in range(1, retries + 1):
            try:
                selected_model = model or self.CLOUDFLARE_MODEL_NAME

                if self.debug:
                    logger.debug(f"Using model: {selected_model}")

                response = openai.ChatCompletion.create(
                    model=selected_model,
                    messages=[system_message, user_message],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                ai_response = response["choices"][0]["message"]["content"].strip()
                if self.debug:
                    logger.debug(f"AI Response: {ai_response}")

                # Attempt to parse the response as JSON
                parsed_response = json.loads(ai_response)

                # Validate that all expected keys are present
                if all(key in parsed_response for key in expected_keys):
                    logger.debug("Valid JSON response received from AI.")
                    return parsed_response
                else:
                    missing_keys = [
                        key for key in expected_keys if key not in parsed_response
                    ]
                    logger.warning(
                        f"Attempt {attempt}: Missing keys {missing_keys} in response: {parsed_response}. Retrying..."
                    )

            except json.JSONDecodeError:
                logger.warning(
                    f"Attempt {attempt}: Failed to parse JSON response. Retrying..."
                )
            except Exception as e:
                logger.error(f"Attempt {attempt}: AI request failed: {e}. Retrying...")

            time.sleep(1)  # Brief pause before retrying

        logger.error(f"Failed to get a valid JSON response after {retries} attempts.")
        return None
