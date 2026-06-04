"""Active LLM model names (set at CLI startup from provider)."""

from __future__ import annotations

from .constants import PROVIDER_MODELS

MODEL_RECALL = PROVIDER_MODELS["openrouter"]["recall"]
MODEL_FILTER = PROVIDER_MODELS["openrouter"]["filter"]
MODEL_NEGATION = PROVIDER_MODELS["openrouter"]["negation"]
MODEL_RELATION = PROVIDER_MODELS["openrouter"]["relation"]
MODEL_CANONICALIZE = PROVIDER_MODELS["openrouter"]["canonicalize"]


def configure_models(provider: str) -> None:
    """Set module-level MODEL_* variables from the provider config."""
    global MODEL_RECALL, MODEL_FILTER, MODEL_NEGATION, MODEL_RELATION, MODEL_CANONICALIZE
    models = PROVIDER_MODELS[provider]
    MODEL_RECALL = models["recall"]
    MODEL_FILTER = models["filter"]
    MODEL_NEGATION = models["negation"]
    MODEL_RELATION = models["relation"]
    MODEL_CANONICALIZE = models["canonicalize"]
