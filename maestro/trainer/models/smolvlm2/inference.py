from typing import Optional, Union
from PIL import Image

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from maestro.trainer.common.utils.device import parse_device_spec


def predict_with_inputs(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
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
            input_ids=input_ids.to(device),
            pixel_values=pixel_values.to(device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=3,
        )
    return processor.batch_decode(generated_ids, skip_special_tokens=False)


def predict(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    image: Image.Image,
    prefix: str,
    device: str | torch.device = "auto",
    max_new_tokens: int = 1024,
) -> str:
    """Generate a text prediction for a single image and text prefix.

    Args:
        model (AutoModelForCausalLM): The Florence-2 model for conditional text generation.
        processor (AutoProcessor): Processor for model inputs and outputs, handling tokenization and decoding.
        image (str | bytes | Image.Image): Input image as a file path, raw bytes, or a PIL Image.
        prefix (str): Text prefix to condition the generated output.
        device (str | torch.device): Device on which to run inference (e.g., "auto", "cpu", "cuda").
        max_new_tokens (int): Maximum number of tokens to generate.

    Returns:
        str: The generated text prediction.
    """
    device = parse_device_spec(device)
    inputs = processor(text=prefix, images=image, return_tensors="pt", padding=True)
    return predict_with_inputs(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        model=model,
        processor=processor,
        device=device,
        max_new_tokens=max_new_tokens,
    )[0]

# def predict_with_images(
#     model: AutoModelForImageTextToText,
#     processor: AutoProcessor,
#     images: Union[str, list[str]],
#     prompt: Optional[str] = None,
#     device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
#     max_new_tokens: int = 512,
#     **kwargs,
# ) -> list[str]:
#     """
#     Generate text predictions from images.

#     Args:
#         model: The SmolVLM2 model
#         processor: The model's processor
#         images: Path(s) to image(s)
#         prompt: Optional prompt to guide generation
#         device: Device to run inference on
#         max_new_tokens: Maximum number of tokens to generate
#         **kwargs: Additional generation parameters

#     Returns:
#         List of generated text strings
#     """
#     if isinstance(images, str):
#         images = [images]

#     inputs = processor(images=images, text=prompt if prompt else "", return_tensors="pt")

#     return predict_with_inputs(
#         model=model,
#         processor=processor,
#         input_ids=inputs["input_ids"],
#         pixel_values=inputs["pixel_values"],
#         device=device,
#         max_new_tokens=max_new_tokens,
#         **kwargs,
#     )
