from typing import Any

from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLProcessor


def format_conversation(
    image: str | bytes | Image.Image, prefix: str, suffix: str | None = None, system_message: str | None = None
) -> list[dict]:
    messages = []

    if system_message is not None:
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            }
        )

    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": prefix,
                },
            ],
        }
    )

    if suffix is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": suffix}],
            }
        )

    return messages


def train_collate_fn(
    batch: list[tuple[Image.Image, dict[str, Any]]], processor: Qwen2_5_VLProcessor, system_message: str | None = None
):
    images, data = zip(*batch)
    conversations = [
        format_conversation(image, entry["prefix"], entry["suffix"], system_message)
        for image, entry in zip(images, data)
    ]

    texts = [processor.apply_chat_template(conversation=conversation, tokenize=False) for conversation in conversations]
    image_inputs = [process_vision_info(conversation)[0] for conversation in conversations]
    model_inputs = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    labels = model_inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    image_tokens = [151652, 151653, 151655]
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100

    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]
    image_grid_thw = model_inputs["image_grid_thw"]

    return input_ids, attention_mask, pixel_values, image_grid_thw, labels


def evaluation_collate_fn(
    batch: list[tuple[Image.Image, dict[str, Any]]], processor: Qwen2_5_VLProcessor, system_message: str | None = None
):
    images, data = zip(*batch)
    prefixes = [entry["prefix"] for entry in data]
    suffixes = [entry["suffix"] for entry in data]
    conversations = [
        format_conversation(image, entry["prefix"], system_message=system_message) for image, entry in zip(images, data)
    ]

    texts = [processor.apply_chat_template(conversation=conversation, tokenize=False) for conversation in conversations]
    image_inputs = [process_vision_info(conversation)[0] for conversation in conversations]
    model_inputs = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]
    image_grid_thw = model_inputs["image_grid_thw"]

    return input_ids, attention_mask, pixel_values, image_grid_thw, prefixes, suffixes
