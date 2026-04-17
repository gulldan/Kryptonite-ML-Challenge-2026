"""Helpers for Hugging Face AudioXVector model outputs."""

from __future__ import annotations

from typing import Any


def extract_xvector_embeddings(outputs: Any) -> Any:
    """Return embeddings from a Transformers AudioXVector output object."""
    embeddings = getattr(outputs, "embeddings", None)
    if embeddings is not None:
        return embeddings
    if isinstance(outputs, dict) and outputs.get("embeddings") is not None:
        return outputs["embeddings"]
    raise RuntimeError("AudioXVector model output did not contain `embeddings`.")


__all__ = ["extract_xvector_embeddings"]
