from typing import Any

from PIL import Image
from transformers import  AutoProcessor

def format_data(image, prefix, suffix):
    return [

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
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": suffix}],
        },
    ]

def train_collate_fn(
    batch: list[tuple[Image.Image, dict[str, Any]]],
      processor: AutoProcessor ):
    images, data = zip(*batch)

    messages = [
        format_data(image, entry["prefix"], entry["suffix"])
        for image, entry in zip(images, data)
    ]
    
    inputs = processor.apply_chat_template(messages, tokenize=True,return_dict=True,
            return_tensors="pt",padding = True)

    # Clone input_ids to labels and mask out everything except suffix
    labels = inputs.input_ids.clone()

    # Mask pad tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Mask <image> tokens
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    labels[labels == image_token_id] = -100
    return inputs, labels

def evaluation_collate_fn(
    batch: list[tuple[Image.Image, dict[str, Any]]],
    processor: AutoProcessor
):
    images, data = zip(*batch)

    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": entry["prefix"]},
                ],
            }
        ]
        for image, entry in zip(images, data)
    ]
    inputs = processor.apply_chat_template(messages, tokenize=True,return_dict=True,
            return_tensors="pt",padding = True)


    prefixes = ["<image>" + entry["prefix"] for entry in data]
    suffixes = [entry["suffix"] for entry in data]
    return inputs, prefixes, suffixes

