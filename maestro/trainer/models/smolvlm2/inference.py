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

def predict(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    image: str | bytes | Image.Image,
    prefix: str,
    device: str | torch.device = "auto",
    max_new_tokens: int = 1024,
) -> str:
    """Generate a text prediction for a single image and text prefix.

    Args:
        model (AutoModelForImageTextToText): The PaliGemma model for generation.
        processor (AutoProcessor): Tokenizer and processor for model inputs/outputs.
        image (str | bytes | Image.Image): Input image as a file path, bytes, or PIL Image.
        prefix (str): Text prefix to condition the generation.
        device (str | torch.device): Device to run inference on.
        max_new_tokens (int): Maximum number of new tokens to generate.

    Returns:
        str: Generated text prediction.
    """
    device = parse_device_spec(device)
    text = "<image>" + prefix
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
    return predict_with_inputs(
        inputs = inputs, model=model, processor=processor, device=device, max_new_tokens=max_new_tokens
    )[0]