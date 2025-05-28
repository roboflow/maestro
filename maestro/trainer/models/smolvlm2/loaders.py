from typing import Any

from PIL import Image
from transformers import  AutoProcessor
import supervision as sv
from torch.nn.utils.rnn import pad_sequence

def train_collate_fn(
    batch: list[tuple[Image.Image, dict[str, Any]]],
      processor: AutoProcessor ):
    images, data = zip(*batch)
    instances = []
    for i in range(len(images)):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": images[i]},
                    {"type": "text", "text": data[i]["prefix"]},
                ]
            },
        ]

        instance = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        instances.append(instance)



    #prefixes = ["<image>" + entry["prefix"] for entry in data]
    suffixes = [entry["suffix"] for entry in data]
    #inputs = processor(text=prefixes, images=images, return_tensors="pt", padding=True)

    input_ids = [i["input_ids"] for i in instances]#inputs["input_ids"]
    pixel_values = [i["pixel_values"] for i in instances]#inputs["pixel_values"]
    attention_mask = [i["attention_mask"] for i in instances]#inputs["attention_mask"]

    labels = processor.tokenizer(
        text=suffixes, return_tensors="pt", padding=True, return_token_type_ids=False
    ).input_ids

    return input_ids,attention_mask, pixel_values, labels

def evaluation_collate_fn(batch: list[tuple[Image.Image, dict[str, Any]]], processor: AutoProcessor):
    images, data = zip(*batch)
    prefixes = ["<image>" + entry["prefix"] for entry in data]
    suffixes = [entry["suffix"] for entry in data]
    inputs = processor(text=prefixes, images=images, return_tensors="pt", padding=True)

    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    attention_mask = inputs["attention_mask"]

    return input_ids, attention_mask, pixel_values, images, prefixes, suffixes









# from typing import Optional

# import torch
# from PIL import Image
# from torch.utils.data import DataLoader, Dataset
# from transformers import AutoProcessor


# class SmolVLM2Dataset(Dataset):
#     """Dataset for SmolVLM2 model."""

#     def __init__(
#         self, image_paths: list[str], texts: Optional[list[str]] = None, processor: Optional[AutoProcessor] = None
#     ):
#         """
#         Initialize dataset.

#         Args:
#             image_paths: List of paths to images
#             texts: Optional list of corresponding texts
#             processor: Model processor for preprocessing
#         """
#         self.image_paths = image_paths
#         self.texts = texts
#         self.processor = processor

#     def __len__(self) -> int:
#         return len(self.image_paths)

#     def __getitem__(self, idx: int) -> dict:
#         """Get a single item from the dataset."""
#         image = Image.open(self.image_paths[idx])

#         if self.texts is not None:
#             text = self.texts[idx]
#         else:
#             text = ""

#         if self.processor is not None:
#             return self.processor(images=image, text=text, return_tensors="pt")
#         else:
#             return {"image": image, "text": text}


# def train_collate_fn(batch: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     Collate function for training data.

#     Args:
#         batch: List of processed samples

#     Returns:
#         Tuple of (input_ids, pixel_values, labels)
#     """
#     input_ids = torch.stack([item["input_ids"].squeeze(0) for item in batch])
#     pixel_values = torch.stack([item["pixel_values"].squeeze(0) for item in batch])
#     labels = torch.stack([item["labels"].squeeze(0) for item in batch])

#     return input_ids, pixel_values, labels


# def evaluation_collate_fn(
#     batch: list[dict],
# ) -> tuple[torch.Tensor, torch.Tensor, list[Image.Image], list[str], list[str]]:
#     """
#     Collate function for evaluation data.

#     Args:
#         batch: List of processed samples

#     Returns:
#         Tuple of (input_ids, pixel_values, images, prompts, targets)
#     """
#     input_ids = torch.stack([item["input_ids"].squeeze(0) for item in batch])
#     pixel_values = torch.stack([item["pixel_values"].squeeze(0) for item in batch])
#     images = [item["image"] for item in batch]
#     prompts = [item["text"] for item in batch]
#     targets = [item["text"] for item in batch]  # In evaluation, target is same as prompt

#     return input_ids, pixel_values, images, prompts, targets


# def create_dataloader(
#     dataset: Dataset, batch_size: int = 8, num_workers: int = 4, shuffle: bool = True, collate_fn=None
# ) -> DataLoader:
#     """
#     Create a DataLoader for the dataset.

#     Args:
#         dataset: Dataset to create loader for
#         batch_size: Batch size
#         num_workers: Number of worker processes
#         shuffle: Whether to shuffle the data
#         collate_fn: Optional collate function
#     Returns:
#         DataLoader instance
#     """
#     return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, collate_fn=collate_fn)
