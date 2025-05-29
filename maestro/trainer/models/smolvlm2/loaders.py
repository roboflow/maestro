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
    
    # Apply chat template WITHOUT tokenization
    #texts = [processor.apply_chat_template(m, tokenize=False) for m in messages]
    batch_enc = processor.apply_chat_template(messages, tokenize=True,return_dict=True,
            return_tensors="pt",padding = True)

    # Clone input_ids to labels and mask out everything except suffix
    labels = batch_enc.input_ids.clone()

    # Mask pad tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Mask <image> tokens
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    labels[labels == image_token_id] = -100

    # Mask prefix tokens: keep only suffix as target
    for i, suffix in enumerate(suffixes):
        suffix_ids = processor.tokenizer(suffix, add_special_tokens=False).input_ids
        labels[i, :-len(suffix_ids)] = -100

    return batch_enc, labels
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

    # texts = [processor.apply_chat_template(m, tokenize=False) for m in messages]
    # # Tokenize and encode images
    # batch_enc = processor(text=texts, images=images, return_tensors="pt", padding=True)
    batch_enc = processor.apply_chat_template(messages, tokenize=True,return_dict=True,
            return_tensors="pt",padding = True)
    #print(batch_enc)

    # input_ids = batch_enc["input_ids"]
    # attention_mask = batch_enc["attention_mask"]
    # pixel_values = batch_enc["pixel_values"]

    prefixes = ["<image>" + entry["prefix"] for entry in data]
    suffixes = [entry["suffix"] for entry in data]
    return batch_enc, prefixes, suffixes

