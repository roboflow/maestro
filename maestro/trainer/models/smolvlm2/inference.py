from typing import Optional, Union
from PIL import Image

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from maestro.trainer.common.utils.device import parse_device_spec


def predict_with_inputs(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    inputs: torch.Tensor,
    device: Union[str, torch.device],
    max_new_tokens: int = 512,
    **kwargs,
) -> list[str]:
    """
    Generate text predictions using the model.

    Args:
        model: The SmolVLM2 model
        processor: The model's processor
        input_ids: Input token IDs
        pixel_values: Input image pixel values
        device: Device to run inference on
        max_new_tokens: Maximum number of tokens to generate
        **kwargs: Additional generation parameters

    Returns:
        List of generated text strings
    """
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs.to(device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return processor.batch_decode(generated_ids, skip_special_tokens=True)

