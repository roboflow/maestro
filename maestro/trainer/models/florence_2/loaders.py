from typing import List, Tuple, Dict, Any

import torch
from PIL import Image
from transformers import AutoProcessor


def process_batch(
    batch: List[Tuple[Image.Image, Dict[str, Any]]],
    processor: AutoProcessor,
    device: torch.device,
) -> Tuple[torch.Tensor, List[str], List[str], List[Image.Image]]:
    images, data = zip(*batch)
    prefixes = [entry["prefix"] for entry in data]
    suffixes = [entry["suffix"] for entry in data]
    inputs = processor(text=list(prefixes), images=list(images), return_tensors="pt", padding=True).to(device)
    return inputs, prefixes, suffixes, images
