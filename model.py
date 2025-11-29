import json
import requests
import os
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv


# Configuration class
class YuppConfig:
    """Yupp API configuration management"""

    def __init__(self):
        self.base_url = "https://yupp.ai"
        self.api_url = f"{self.base_url}/api/trpc/model.getModelInfoList,scribble.getScribbleByLabel?batch=1&input=%7B%220%22%3A%7B%22json%22%3Anull%2C%22meta%22%3A%7B%22values%22%3A%5B%22undefined%22%5D%7D%7D%2C%221%22%3A%7B%22json%22%3A%7B%22label%22%3A%22homepage_banner%22%7D%7D%7D"

    def get_headers(self) -> Dict[str, str]:
        """Get required HTTP headers"""
        return {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": f"{self.base_url}/",
            "Origin": self.base_url,
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
        }

    def get_cookies(self) -> Dict[str, str]:
        """Retrieve session token from environment variables and build cookies"""
        # Retrieve YUPP_TOKENS from environment
        env_tokens = os.getenv("YUPP_TOKENS")
        if not env_tokens:
            print("Warning: YUPP_TOKENS environment variable not set")
            return {}

        try:
            # Support multiple tokens separated by commas; use the first one
            tokens = [token.strip() for token in env_tokens.split(",") if token.strip()]
            if not tokens:
                print("Warning: No valid tokens found")
                return {}

            # Use the first valid token
            token = tokens[0]
            return {"__Secure-yupp.session-token": token}

        except Exception as e:
            print(f"Warning: Failed to parse YUPP_TOKENS environment variable: {e}")
            return {}


# Initialize configuration
config = YuppConfig()


def fetch_model_data() -> Optional[List[Dict[str, Any]]]:
    """Fetch model data from Yupp"""
    try:
        # Create session and set cookies
        session = requests.Session()
        cookies = config.get_cookies()
        headers = config.get_headers()

        # Apply cookies
        for key, value in cookies.items():
            session.cookies.set(key, value)

        print(f"Requesting: {config.api_url}")
        response = session.get(config.api_url, headers=headers, timeout=30)

        print(f"Response status code: {response.status_code}")

        if response.status_code != 200:
            print(f"Request failed with status: {response.status_code}")
            return None

        # Parse JSON payload
        response_data = response.json()
        print("Successfully retrieved and parsed JSON data")

        # Extract model list data
        if isinstance(response_data, list) and len(response_data) > 0:
            return response_data[0]["result"]["data"]["json"]
        else:
            print("Unexpected response structure")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Network request failed: {e}")
        return None
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Failed to parse JSON: {e}")
        if "response" in locals():
            print(f"Response content: {response.text[:200]}")
        return None
    except KeyError as e:
        print(f"Failed to parse response structure: {e}")
        return None


def load_fallback_data() -> List[Dict[str, Any]]:
    """Load fallback data from local file"""
    try:
        with open("models.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("models.json fallback file not found")
        return []
    except json.JSONDecodeError as e:
        print(f"Failed to parse fallback file JSON: {e}")
        return []


def generate_model_tags(item: Dict[str, Any]) -> List[str]:
    """Generate model tags (emojis)"""
    tags = []
    tag_mapping = {
        "isPro": "☀️",
        "isMax": "🔥",
        "isNew": "🆕",
        "isLive": "🎤",
        "isAgent": "🤖",
        "isFast": "🚀",
        "isReasoning": "🧠",
        "isImageGeneration": "🎨",
    }

    for key, emoji in tag_mapping.items():
        if item.get(key, False):
            tags.append(emoji)

    # Check if attachments are supported
    if (
        item.get("supportedAttachmentMimeTypes")
        and len(item["supportedAttachmentMimeTypes"]) > 0
    ):
        tags.append("📎")

    return tags


def filter_and_process_models(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter and process model data"""
    # Supported model families
    supported_families = [
        "GPT",
        "Claude",
        "Gemini",
        "Qwen",
        "DeepSeek",
        "Perplexity",
        "Kimi",
    ]

    processed_models = []

    for item in data:
        # Filter criteria: supported families or special capabilities
        if (
            item.get("family") in supported_families
            or item.get("isImageGeneration")
            or item.get("isAgent")
            or item.get("isLive")
        ):

            # Generate tags
            tags = generate_model_tags(item)

            # Build display label
            label = item.get("label", "")
            if tags:
                label += "\n" + " | ".join(tags)

            # Build processed model data
            processed_item = {
                "id": item.get("id"),
                "name": item.get("name"),
                "label": label,
                "shortLabel": item.get("shortLabel"),
                "publisher": item.get("publisher"),
                "family": item.get("family"),
                "isPro": item.get("isPro", False),
                "isInternal": item.get("isInternal", False),
                "isMax": item.get("isMax", False),
                "isLive": item.get("isLive", False),
                "isNew": item.get("isNew", False),
                "isImageGeneration": item.get("isImageGeneration", False),
                "isAgent": item.get("isAgent", False),
                "isReasoning": item.get("isReasoning", False),
                "isFast": item.get("isFast", False),
            }
            processed_models.append(processed_item)

    return processed_models


def save_models_to_file(
    models: List[Dict[str, Any]], filename: str = "model.json"
) -> bool:
    """Persist model data to disk"""
    try:
        # Ensure parent directory exists
        dir_name = os.path.dirname(filename)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        # Create file if it does not exist
        if not os.path.exists(filename):
            try:
                with open(filename, "x", encoding="utf-8") as _:
                    pass
                print(f"Created file: {filename}")
            except FileExistsError:
                # File may have been created concurrently; ignore
                pass

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(models, f, indent=4, ensure_ascii=False)
        print(f"Saved {len(models)} models to {filename}")
        return True
    except Exception as e:
        print(f"Failed to save file: {e}")
        return False


def fetch_and_save_models(filename: str = "model.json") -> bool:
    """Fetch models and save them to the specified file"""
    # Load environment variables
    load_dotenv()

    print("=== Automatically fetching Yupp model data ===")

    # Ensure required environment variables are present
    if not os.getenv("YUPP_TOKENS"):
        print("Warning: YUPP_TOKENS environment variable is not set. Cannot fetch models automatically.")
        return False

    # Fetch model data
    data = fetch_model_data()
    if not data:
        print("API request failed; attempting to load local fallback data...")
        data = load_fallback_data()

    # Process model data
    if data:
        print(f"Processing {len(data)} model entries...")
        processed_models = filter_and_process_models(data)
        return save_models_to_file(processed_models, filename)
    else:
        print("No model data available")
        return False


def main():
    """Main entry point"""
    # Load environment variables
    load_dotenv()

    print("=== Yupp model data fetch tool ===")

    # Ensure required environment variables are present
    if not os.getenv("YUPP_TOKENS"):
        print("Warning: YUPP_TOKENS environment variable is not set")
        print("Please set YUPP_TOKENS, for example:")
        print("export YUPP_TOKENS='your_token_here'")
        return False

    # Fetch model data
    data = fetch_model_data()
    if not data:
        print("API request failed; attempting to load local fallback data...")
        data = load_fallback_data()

    # Process model data
    if data:
        print(f"Processing {len(data)} model entries...")
        processed_models = filter_and_process_models(data)
        save_models_to_file(processed_models)
    else:
        print("No model data available")
        return False

    return True


if __name__ == "__main__":
    main()
