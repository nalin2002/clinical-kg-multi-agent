"""Utility functions for OpenRouter API interactions."""

import os
from typing import Any, Dict

import httpx

# API Configuration
BASE_URL = "https://openrouter.ai/api/v1"


def check_credits(api_key: str, base_url: str = BASE_URL) -> Dict[str, Any]:
    """Check remaining OpenRouter credits for this specific API key."""
    url = f"{base_url}/key"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=30) as client:
        try:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            return {
                "error": f"HTTP {e.response.status_code}",
                "detail": e.response.text[:200]
            }
        except Exception as e:
            return {"error": str(e)}


def print_remaining_credits(api_key: str, base_url: str = BASE_URL) -> None:
    """Print remaining OpenRouter credits in a formatted way."""
    credits_data = check_credits(api_key, base_url)
    if "error" in credits_data:
        print("⚠️  Error checking credits:", credits_data)
    else:
        data = credits_data.get("data", {})

        # Key-specific information
        limit = data.get("limit", 0)  # Credit limit set for this key
        usage = data.get("usage", 0)  # Usage by this key
        remaining = limit - usage if limit else "No limit set"

        print(f"💳 API Key Credit Balance:")
        print(f"   Key limit:    ${limit:.2f}" if isinstance(limit, (int, float)) else f"   Key limit:    {limit}")
        print(f"   Key usage:    ${usage:.2f}")
        print(f"   Remaining:    ${remaining:.2f}" if isinstance(remaining, (int, float)) else f"   Remaining:    {remaining}")


def main():
    api_key = os.getenv("API_KEY")
    print_remaining_credits(api_key)


if __name__ == "__main__":
    main()