from typing import Any

from PIL import Image
from transformers import  AutoProcessor
import supervision as sv
from torch.nn.utils.rnn import pad_sequence
import torch

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
    suffixes = [entry["suffix"] for entry in data]
    
    inputs = processor.apply_chat_template(messages, tokenize=True,return_dict=True,
            return_tensors="pt",padding = True)

    # Clone input_ids to labels and mask out everything except suffix
    labels = inputs.input_ids.clone()

    # Mask pad tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Mask <image> tokens
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    labels[labels == image_token_id] = -100

    # Mask prefix tokens: keep only suffix as target
    for i, suffix in enumerate(suffixes):
        suffix_ids = processor.tokenizer(suffix, add_special_tokens=False).input_ids
        # Try to find the start index of the suffix tokens in the full input sequence
        sequence = inputs.input_ids[i].tolist()
        for j in range(len(sequence) - len(suffix_ids) + 1):
            if sequence[j:j + len(suffix_ids)] == suffix_ids:
                print("here")
                labels[i, :j] = -100
                labels[i, j + len(suffix_ids):] = -100
                break
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

