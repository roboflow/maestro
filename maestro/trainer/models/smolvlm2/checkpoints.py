import os
from typing import Optional

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor


def save_checkpoint(
    model: AutoModelForVision2Seq,
    processor: AutoProcessor,
    path: str,
    metadata: Optional[dict] = None
) -> None:
    """
    Save model checkpoint.

    Args:
        model: Model to save
        processor: Processor to save
        path: Path to save checkpoint
        metadata: Optional metadata to save
    """
    os.makedirs(path, exist_ok=True)

    # Save model
    model.save_pretrained(path)

    # Save processor
    processor.save_pretrained(path)

    # Save metadata if provided
    if metadata is not None:
        torch.save(metadata, os.path.join(path, "metadata.pt"))

def load_checkpoint(
    path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> dict:
    """
    Load model checkpoint.

    Args:
        path: Path to checkpoint
        device: Device to load model on

    Returns:
        Dictionary containing model, processor, and metadata
    """
    # Load model
    model = AutoModelForVision2Seq.from_pretrained(path)
    model.to(device)

    # Load processor
    processor = AutoProcessor.from_pretrained(path)

    # Load metadata if exists
    metadata_path = os.path.join(path, "metadata.pt")
    metadata = torch.load(metadata_path) if os.path.exists(metadata_path) else None

    return {
        "model": model,
        "processor": processor,
        "metadata": metadata
    }
