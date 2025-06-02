from typing import Optional, Union
from PIL import Image

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from maestro.trainer.common.utils.device import parse_device_spec

def predict_with_inputs(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    pixel_attention_mask: torch.Tensor,
    max_new_tokens: int = 64,
) -> list[str]:
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_attention_mask=pixel_attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
        prefix_length = input_ids.shape[-1]
        generated_ids = generated_ids[:, prefix_length:]
        return processor.batch_decode(generated_ids, skip_special_tokens=True)


def predict(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    image: str | bytes | Image.Image,
    prefix: str,
    device: str | torch.device = "auto",
    max_new_tokens: int = 64,
) -> str:
    device = parse_device_spec(device)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prefix},
            ]
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device, dtype=torch.bfloat16)
    return predict_with_inputs(
        **inputs, model=model, processor=processor, max_new_tokens=max_new_tokens
    )[0]