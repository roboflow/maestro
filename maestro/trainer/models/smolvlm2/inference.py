from typing import Optional, Union

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor


class SmolVLM2Inference:
    """Inference interface for SmolVLM2 model."""

    def __init__(
        self,
        model_name: str = "smol-ai/smolvlm2-500m",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ):
        """Initialize inference interface."""
        self.model = AutoModelForVision2Seq.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.device = device

    def generate(
        self, images: Union[str, list[str]], prompt: Optional[str] = None, max_new_tokens: int = 512, **kwargs
    ) -> dict:
        """
        Generate text from images.

        Args:
            images: Path(s) to image(s)
            prompt: Optional prompt to guide generation
            max_new_tokens: Maximum number of tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            Dictionary containing generated text and other outputs
        """
        # Process inputs
        inputs = self.processor(images=images, text=prompt if prompt else "", return_tensors="pt")

        # Generate
        outputs = self.model.generate(
            input_ids=inputs["input_ids"].to(self.device),
            pixel_values=inputs["pixel_values"].to(self.device),
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

        # Decode outputs
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)

        return {"generated_text": generated_text, "model_outputs": outputs}


def predict_with_inputs(
    model: AutoModelForVision2Seq,
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
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids.to(device),
            pixel_values=pixel_values.to(device),
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
    return processor.batch_decode(outputs, skip_special_tokens=True)


def predict_with_images(
    model: AutoModelForVision2Seq,
    processor: AutoProcessor,
    images: Union[str, list[str]],
    prompt: Optional[str] = None,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    max_new_tokens: int = 512,
    **kwargs,
) -> list[str]:
    """
    Generate text predictions from images.

    Args:
        model: The SmolVLM2 model
        processor: The model's processor
        images: Path(s) to image(s)
        prompt: Optional prompt to guide generation
        device: Device to run inference on
        max_new_tokens: Maximum number of tokens to generate
        **kwargs: Additional generation parameters

    Returns:
        List of generated text strings
    """
    if isinstance(images, str):
        images = [images]

    inputs = processor(images=images, text=prompt if prompt else "", return_tensors="pt")

    return predict_with_inputs(
        model=model,
        processor=processor,
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        device=device,
        max_new_tokens=max_new_tokens,
        **kwargs,
    )
