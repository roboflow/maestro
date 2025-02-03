from typing import Any

from PIL import Image


def train_collate_fn(batch: list[tuple[Image.Image, dict[str, Any]]], processor):
    images, data = zip(*batch)
    prefixes = [entry["prefix"] for entry in data]
    inputs = processor(text=prefixes, images=images, return_tensors="pt", padding=True)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]
    return input_ids, attention_mask, pixel_values


def evaluation_collate_fn(batch: list[tuple[Image.Image, dict[str, Any]]], processor):
    images, data = zip(*batch)
    prefixes = [entry["prefix"] for entry in data]
    suffixes = [entry["suffix"] for entry in data]
    inputs = processor(text=prefixes, images=images, return_tensors="pt", padding=True)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]
    return input_ids, attention_mask, pixel_values, prefixes, suffixes
