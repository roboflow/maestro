from typing import Any

from PIL import Image


def train_collate_fn(batch: list[tuple[Image.Image, dict[str, Any]]], processor):
    images, data = zip(*batch)
    prefixes = [entry["prefix"] for entry in data]
    suffixes = [entry["suffix"] for entry in data]
    inputs = processor(text=prefixes, images=images, return_tensors="pt", padding=True)

    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]

    labels = processor.tokenizer(
        text=suffixes, return_tensors="pt", padding=True, return_token_type_ids=False
    ).input_ids

    return input_ids, pixel_values, labels


def evaluation_collate_fn(batch: list[tuple[Image.Image, dict[str, Any]]], processor):
    images, data = zip(*batch)
    prefixes = [entry["prefix"] for entry in data]
    suffixes = [entry["suffix"] for entry in data]
    inputs = processor(text=prefixes, images=images, return_tensors="pt", padding=True)

    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    return input_ids, pixel_values, prefixes, suffixes
