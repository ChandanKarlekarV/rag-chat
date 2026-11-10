"""Choose where the local models run (GPU when available) and cap CPU threads."""

import logging
from functools import lru_cache

import torch

log = logging.getLogger("rag-chat.devices")


@lru_cache
def resolve_device(setting: str = "auto") -> str:
    """'auto' → 'cuda' when an NVIDIA GPU is usable, else 'cpu'. Explicit values are passed through."""
    if setting != "auto":
        return setting
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        log.info("using GPU", extra={"gpu": torch.cuda.get_device_name(0)})
    return device


def limit_cpu_threads(threads: int) -> None:
    """Cap PyTorch's CPU threads (0 = leave PyTorch's default, which uses every core)."""
    if threads > 0:
        torch.set_num_threads(threads)
