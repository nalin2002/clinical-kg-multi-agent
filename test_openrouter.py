import os
import requests

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "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")

def test_openrouter_key(api_key: str) -> None:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 10,
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        data = response.json()
        message = data["choices"][0]["message"]["content"]
        print(f"API key is valid. Response: {message}")
    elif response.status_code == 401:
        print("Invalid API key (401 Unauthorized).")
    elif response.status_code == 402:
        print("API key valid but insufficient credits (402 Payment Required).")
    else:
        print(f"Unexpected status {response.status_code}: {response.text}")


if __name__ == "__main__":
    test_openrouter_key(OPENROUTER_API_KEY)
