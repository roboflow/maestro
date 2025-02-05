import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from maestro.trainer.common.utils.device import parse_device_spec


def predict_with_inputs(
    model: AutoModelForCausalLM,
    processor: AutoProcessor,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    device: torch.device,
    max_new_tokens: int = 1024,
) -> list[str]:
    """Generate text predictions from preprocessed model inputs.

    Args:
        model (AutoModelForCausalLM): The Florence-2 model for conditional text generation.
        processor (AutoProcessor): Processor for model inputs and outputs, handling tokenization and decoding.
        input_ids (torch.Tensor): Tensor of input token IDs representing the text prompt.
        pixel_values (torch.Tensor): Processed image tensor.
        device (torch.device): Device on which to run inference.
        max_new_tokens (int): Maximum number of tokens to generate.

    Returns:
        list[str]: A list of generated text predictions.
    """
    generated_ids = model.generate(
        input_ids=input_ids.to(device),
        pixel_values=pixel_values.to(device),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=3,
    )
    return processor.batch_decode(generated_ids, skip_special_tokens=False)


def predict(
    model: AutoModelForCausalLM,
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
