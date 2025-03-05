# Parts of the loader is based on https://huggingface.co/microsoft/Phi-4-multimodal-instruct/resolve/main/sample_finetune_vision.py

from typing import Any, Optional

import torch
from PIL import Image
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor

_IGNORE_INDEX = -100
_MAX_TRAINING_LENGTH = 8192


def cat_with_pad(tensors: list[Tensor], dim: int = 0, padding_value: int = 0) -> Tensor:
    """Concatenate tensors with padding along the specified dimension.

    Args:
        tensors: list of tensors to concatenate.
        dim: Dimension along which to concatenate.
        padding_value: Value to use for padding.

    Returns:
        Tensor: Concatenated tensor with padding.
    """
    max_shape = [max(s[i] for s in [t.shape for t in tensors]) for i in range(len(tensors[0].shape))]
    result = []

    for tensor in tensors:
        pad_size = []
        for i, (dim_size, max_dim_size) in enumerate(zip(tensor.shape, max_shape)):
            pad_before = 0
            pad_after = max_dim_size - dim_size
            pad_size.extend([pad_before, pad_after])

        padded = torch.nn.functional.pad(tensor, pad_size[::-1], value=padding_value)
        result.append(padded)

    return torch.cat(result, dim=dim)


def train_collate_fn(
    batch: list[tuple[Image.Image, dict[str, Any]]], processor: AutoProcessor, system_message: Optional[str] = None
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Collate function for training data.

    Args:
        batch: list of tuples containing (image, data) pairs.
        processor: Processor for tokenization and image processing.
        system_message: Optional system message to include in prompts.

    Returns:
        tuple containing:
            - input_ids: Tensor of token ids
            - attention_mask: Tensor indicating which tokens to attend to
            - input_image_embeds: Tensor of image embeddings
            - image_attention_mask: Tensor indicating which image patches to attend to
            - image_sizes: Tensor containing image dimensions
            - labels: Tensor of target token ids (with ignored positions marked as _IGNORE_INDEX)
            - input_mode: Tensor with value 1 indicating input mode
    """
    images, data = zip(*batch)
    prefixes = ["<|image_1|>" + entry["prefix"] for entry in data]
    suffixes = [entry["suffix"] for entry in data]

    processed_inputs = []

    for image, prefix, suffix in zip(images, prefixes, suffixes):
        user_message = {"role": "user", "content": prefix}
        messages = [user_message]

        if system_message:
            system_msg = {"role": "system", "content": system_message}
            messages = [system_msg, *messages]

        if suffix:
            assistant_message = {"role": "assistant", "content": suffix}
            messages.append(assistant_message)

        prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(prompt, images=[image], return_tensors="pt")

        if suffix:
            labels = torch.full_like(inputs["input_ids"], _IGNORE_INDEX)
            assistant_text = processor.tokenizer.apply_chat_template([assistant_message], tokenize=False)
            assistant_ids = processor.tokenizer(assistant_text, return_tensors="pt").input_ids
            labels[:, -assistant_ids.shape[1] :] = inputs["input_ids"][:, -assistant_ids.shape[1] :]
        else:
            labels = torch.full_like(inputs["input_ids"], _IGNORE_INDEX)

        inputs["labels"] = labels

        if inputs["input_ids"].size(1) > _MAX_TRAINING_LENGTH:
            inputs["input_ids"] = inputs["input_ids"][:, :_MAX_TRAINING_LENGTH]
            inputs["labels"] = inputs["labels"][:, :_MAX_TRAINING_LENGTH]

        processed_inputs.append(inputs)

    input_ids_list = []
    labels_list = []
    input_image_embeds_list = []
    image_attention_mask_list = []
    image_sizes_list = []

    for inputs in processed_inputs:
        input_ids_list.append(inputs["input_ids"][0])
        labels_list.append(inputs["labels"][0])
        input_image_embeds_list.append(inputs["input_image_embeds"])
        image_attention_mask_list.append(inputs["image_attention_mask"])
        image_sizes_list.append(inputs["image_sizes"])

    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
    labels = pad_sequence(labels_list, batch_first=True, padding_value=_IGNORE_INDEX)
    attention_mask = (input_ids != processor.tokenizer.pad_token_id).long()
    input_image_embeds = cat_with_pad(input_image_embeds_list, dim=0)
    image_attention_mask = cat_with_pad(image_attention_mask_list, dim=0)
    image_sizes = torch.cat(image_sizes_list)

    input_mode = torch.tensor([1])  # vision mode as 1-d tensor

    return (
        input_ids,
        attention_mask,
        input_image_embeds,
        image_attention_mask,
        image_sizes,
        labels,
        input_mode,
    )


def evaluation_collate_fn(
    batch: list[tuple[Image.Image, dict[str, Any]]], processor: AutoProcessor, system_message: Optional[str] = None
) -> tuple[list[Image.Image], list[str], list[Optional[str]], Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Collate function for evaluation data.

    Args:
        batch: list of tuples containing (image, data) pairs.
        processor: Processor for tokenization and image processing.
        system_message: Optional system message to include in prompts.

    Returns:
        tuple containing:
            - images: list of original images
            - prefixes: list of prefix strings
            - suffixes: list of suffix strings (may contain None)
            - input_ids: Tensor of token ids
            - attention_mask: Tensor indicating which tokens to attend to
            - input_image_embeds: Tensor of image embeddings
            - image_attention_mask: Tensor indicating which image patches to attend to
            - image_sizes: Tensor containing image dimensions
            - input_mode: Tensor with value 1 indicating input mode
    """
    images, data = zip(*batch)
    prefixes = ["<|image_1|>" + entry["prefix"] for entry in data]
    suffixes = [entry.get("suffix") for entry in data]

    processed_inputs = []

    for image, prefix in zip(images, prefixes):
        user_message = {"role": "user", "content": prefix}
        messages = [user_message]

        if system_message:
            system_msg = {"role": "system", "content": system_message}
            messages = [system_msg, *messages]

        prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(prompt, images=[image], return_tensors="pt")
        processed_inputs.append(inputs)

    input_ids_list = []
    input_image_embeds_list = []
    image_attention_mask_list = []
    image_sizes_list = []

    for inputs in processed_inputs:
        input_ids_list.append(inputs["input_ids"][0])
        input_image_embeds_list.append(inputs["input_image_embeds"])
        image_attention_mask_list.append(inputs["image_attention_mask"])
        image_sizes_list.append(inputs["image_sizes"])

    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
    attention_mask = (input_ids != processor.tokenizer.pad_token_id).long()
    input_image_embeds = cat_with_pad(input_image_embeds_list, dim=0)
    image_attention_mask = cat_with_pad(image_attention_mask_list, dim=0)
    image_sizes: Tensor = torch.cat(image_sizes_list)

    input_mode = torch.tensor([1])  # vision mode as 1-d tensor

    return (
        images,
        prefixes,
        suffixes,
        input_ids,
        attention_mask,
        input_image_embeds,
        image_attention_mask,
        image_sizes,
        input_mode,
    )
