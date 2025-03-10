from typing import Optional

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from maestro.trainer.common.utils.device import parse_device_spec
from maestro.trainer.models.phi_4.checkpoints import filter_audio_components


def predict_with_inputs(
    model: AutoModelForCausalLM,
    processor: AutoProcessor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    input_image_embeds: torch.Tensor,
    image_sizes: torch.Tensor,
    image_attention_mask: torch.Tensor,
    input_mode: torch.Tensor,
    device: torch.device,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> list[str]:
    """
    Generates predictions from the Phi-4 model using textual and optional image inputs.

    Args:
        model (AutoModelForCausalLM):
            A Phi-4 model capable of conditional generation with visual context.
        processor (AutoProcessor):
            Processor for handling inputs and outputs for the Phi-4 model.
        input_ids (torch.Tensor):
            Tokenized input text IDs.
        attention_mask (torch.Tensor):
            Attention mask corresponding to the tokenized input.
        device (torch.device):
            Device on which to run inference.
        max_new_tokens (int):
            Maximum number of tokens to generate.
        temperature (float):
            Sampling temperature for controlling randomness in generation.
        top_p (float):
            Top-p sampling parameter for nucleus sampling.
        input_image_embeds (torch.Tensor, optional):
            Pre-processed image embeddings for the model.
        image_sizes (torch.Tensor, optional):
            Sizes of the input images.
        image_attention_mask (torch.Tensor, optional):
            Attention mask for the image inputs.
        pixel_values (torch.Tensor, optional):
            Preprocessed image data for visual inputs.
        input_mode (torch.Tensor):
            Tensor specifying if Phi4 works in vision mode (1) or speech mode (0).

    Returns:
        list[str]: A list of decoded strings corresponding to the generated sequences.
    """

    input_len = input_ids.size(1)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            input_image_embeds=input_image_embeds.to(device),
            image_sizes=image_sizes.to(device),
            image_attention_mask=image_attention_mask.to(device),
            max_new_tokens=max_new_tokens,
            input_mode=input_mode.to(device),
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

        generated_ids = outputs[:, input_len:]
        return processor.batch_decode(generated_ids, skip_special_tokens=True)


def predict(
    model: AutoModelForCausalLM,
    processor: AutoProcessor,
    prompt: Optional[str] = None,
    system_message: Optional[str] = None,
    image: str | bytes | Image.Image = None,
    device: str | torch.device = "auto",
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Generates a prediction from the Phi-4 model given a text prompt and optional image.

    Args:
        model (Phi4ForConditionalGeneration):
            A Phi-4 model capable of conditional generation with visual context.
        processor (AutoProcessor):
            Processor for handling inputs and outputs for the Phi-4 model.
        prompt  (str, optional):
            Text prompt for the model to complete.
        image (str | bytes | Image.Image, optional):
            Optional image input for multimodal capabilities.
        system_message (str, optional):
            Optional system message to add context or instructions.
        device (str | torch.device):
            Device on which to run inference.
        max_new_tokens (int):
            Maximum number of tokens to generate.
        temperature (float):
            Sampling temperature for controlling randomness in generation.
        top_p (float):
            Top-p sampling parameter for nucleus sampling.

    Returns:
        str: The decoded string representing the model's generated response.
    """
    device = parse_device_spec(device)

    formatted_prompt = ""

    if system_message:
        formatted_prompt += f"<|system|>{system_message}<|end|>"

    formatted_prompt += "<|user|>"

    if image is not None:
        formatted_prompt += "<|image_1|>"

    formatted_prompt += f"{prompt}<|end|><|assistant|>"

    if image is not None:
        inputs = processor(text=formatted_prompt, images=image, return_tensors="pt")
    else:
        inputs = processor(text=formatted_prompt, return_tensors="pt")

    inputs = filter_audio_components(inputs)
    model = model.to(device)

    return predict_with_inputs(
        model=model,
        processor=processor,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        **inputs,
    )[0]
