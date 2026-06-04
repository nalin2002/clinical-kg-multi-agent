"""LLM API clients for OpenRouter and Anthropic."""

from __future__ import annotations

import time

from .constants import MAX_COMPLETION_TOKENS, MAX_RETRIES, REQUEST_TIMEOUT_SECONDS


class OpenRouterClient:
    """OpenAI-compatible OpenRouter client with low-temperature JSON calls."""

    def __init__(self, api_key: str) -> None:
        from openai import OpenAI

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=90,
        )

    def generate(self, prompt: str, model: str) -> tuple[str, dict]:
        for attempt in range(MAX_RETRIES):
            try:
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                content = completion.choices[0].message.content or ""
                usage = {}
                if completion.usage:
                    usage = {
                        "prompt_tokens": completion.usage.prompt_tokens or 0,
                        "completion_tokens": completion.usage.completion_tokens or 0,
                    }
                if content.strip():
                    return content, usage
            except Exception as exc:
                print(f"      retry {attempt + 1}/{MAX_RETRIES}: {exc}", flush=True)
                time.sleep(2**attempt)
        return "", {}


class AnthropicClient:
    """Direct Anthropic API client, interface-compatible with OpenRouterClient."""

    def __init__(self, api_key: str) -> None:
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key, timeout=90.0)

    def generate(self, prompt: str, model: str) -> tuple[str, dict]:
        for attempt in range(MAX_RETRIES):
            try:
                message = self.client.messages.create(
                    model=model,
                    max_tokens=MAX_COMPLETION_TOKENS,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = message.content[0].text if message.content else ""
                usage = {
                    "prompt_tokens": message.usage.input_tokens,
                    "completion_tokens": message.usage.output_tokens,
                }
                if content.strip():
                    return content, usage
            except Exception as exc:
                print(f"      retry {attempt + 1}/{MAX_RETRIES}: {exc}", flush=True)
                time.sleep(2**attempt)
        return "", {}

    def batch_generate(
        self, requests: list[dict]
    ) -> dict[str, tuple[str, dict]]:
        """Submit requests via Anthropic Message Batches API (50% cost discount).

        Args:
            requests: list of {"custom_id": str, "model": str, "prompt": str}

        Returns:
            dict mapping custom_id -> (content_str, usage_dict)
        """
        if not requests:
            return {}

        batch_requests = [
            {
                "custom_id": req["custom_id"],
                "params": {
                    "model": req["model"],
                    "max_tokens": MAX_COMPLETION_TOKENS,
                    "temperature": 0.1,
                    "messages": [{"role": "user", "content": req["prompt"]}],
                },
            }
            for req in requests
        ]

        batch = self.client.messages.batches.create(requests=batch_requests)
        print(
            f"    Batch {batch.id} submitted ({len(requests)} requests), "
            "polling for results...",
            flush=True,
        )

        poll_interval = 10
        while True:
            batch = self.client.messages.batches.retrieve(batch.id)
            if batch.processing_status == "ended":
                break
            counts = batch.request_counts
            print(
                f"      {counts.succeeded}/{len(requests)} done, "
                f"{counts.processing} processing, {counts.errored} errors",
                flush=True,
            )
            time.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.5, 60)

        results: dict[str, tuple[str, dict]] = {}
        for result in self.client.messages.batches.results(batch.id):
            cid = result.custom_id
            if result.result.type == "succeeded":
                msg = result.result.message
                content = msg.content[0].text if msg.content else ""
                usage = {
                    "prompt_tokens": msg.usage.input_tokens,
                    "completion_tokens": msg.usage.output_tokens,
                }
                results[cid] = (content, usage)
            else:
                print(f"      Batch request {cid} failed: {result.result.type}", flush=True)
                results[cid] = ("", {})

        print(f"    Batch complete: {len(results)} results", flush=True)
        return results
