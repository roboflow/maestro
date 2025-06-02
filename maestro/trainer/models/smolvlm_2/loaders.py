from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor

from maestro.trainer.common.utils.device import parse_device_spec


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
    batch: list[tuple[Image.Image, dict[str, Any]]],
    processor: AutoProcessor,
    system_message: str | None = None,
    device: str | torch.device = "auto",
):
    device = parse_device_spec(device)
    images, data = zip(*batch)
    conversations = [
        format_conversation(image, entry["prefix"], entry["suffix"], system_message)
        for image, entry in zip(images, data)
    ]
    texts = [
        processor.apply_chat_template(conversation=conversation, add_generation_prompt=False).strip()
        for conversation in conversations
    ]
    user_conversations = [
        format_conversation(image, entry["prefix"], system_message) for image, entry in zip(images, data)
    ]
    user_texts = [
        processor.apply_chat_template(conversation=user_conversation, add_generation_prompt=False).strip()
        for user_conversation in user_conversations
    ]
    image_lists = [[image] for image in images]
    model_inputs = processor(text=texts, images=image_lists, return_tensors="pt", padding=True).to(
        device, dtype=torch.bfloat16
    )
    user_model_inputs = processor(text=user_texts, images=image_lists, return_tensors="pt", padding=True).to(
        device, dtype=torch.bfloat16
    )

    labels = model_inputs["input_ids"].clone()
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]
    pixel_attention_mask = model_inputs["pixel_attention_mask"]
    user_input_ids = user_model_inputs["input_ids"]

    for index, user_input_id in enumerate(user_input_ids):
        user_input_length = user_input_id.shape[0]
        labels[index, :user_input_length] = -100

    return input_ids, attention_mask, pixel_values, pixel_attention_mask, labels


def evaluation_collate_fn(
    batch: list[tuple[Image.Image, dict[str, Any]]],
    processor: AutoProcessor,
    system_message: str | None = None,
    device: str | torch.device = "auto",
):
    device = parse_device_spec(device)
    images, data = zip(*batch)
    prefixes = [entry["prefix"] for entry in data]
    suffixes = [entry["suffix"] for entry in data]
    user_conversations = [
        format_conversation(image, entry["prefix"], system_message) for image, entry in zip(images, data)
    ]
    user_texts = [
        processor.apply_chat_template(conversation=user_conversation, add_generation_prompt=False).strip()
        for user_conversation in user_conversations
    ]
    image_lists = [[image] for image in images]
    user_model_inputs = processor(text=user_texts, images=image_lists, return_tensors="pt", padding=True).to(
        device, dtype=torch.bfloat16
    )

    user_input_ids = user_model_inputs["input_ids"]
    user_attention_mask = user_model_inputs["attention_mask"]
    user_pixel_values = user_model_inputs["pixel_values"]
    user_pixel_attention_mask = user_model_inputs["pixel_attention_mask"]

    return (
        user_input_ids,
        user_attention_mask,
        user_pixel_values,
        user_pixel_attention_mask,
        image_lists,
        prefixes,
        suffixes,
    )
