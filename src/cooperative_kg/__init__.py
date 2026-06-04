"""
Cooperative multi-agent clinical KG extraction pipeline.

Package layout:
- ``constants`` — schema types, aliases, enrichment patterns
- ``paths`` — default transcript directory
- ``models`` — active LLM model names (set at CLI startup)
- ``clients`` — OpenRouter and Anthropic API wrappers
- ``io`` — transcript / note file I/O
- ``text_utils`` — JSON parsing and canonicalization
- ``prompts`` — agent prompt templates
- ``agents`` — LLM agent stages 1–5
- ``validation`` — deterministic validator and enrichment
- ``pipeline`` — run_pipeline, batch processing
- ``cli`` — argparse entry point
"""

from .cli import main, load_client
from .constants import OUTPUT_SUFFIX, PROVIDER_MODELS, VALID_EDGE_TYPES, VALID_NODE_TYPES
from .paths import TRANSCRIPT_DIR, default_transcripts_dir
from .pipeline import process_one, run_all_batch, run_pipeline

__all__ = [
    "OUTPUT_SUFFIX",
    "PROVIDER_MODELS",
    "TRANSCRIPT_DIR",
    "VALID_EDGE_TYPES",
    "VALID_NODE_TYPES",
    "default_transcripts_dir",
    "load_client",
    "main",
    "process_one",
    "run_all_batch",
    "run_pipeline",
]
