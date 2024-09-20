import json
import os
from typing import Any

from PIL import Image
from transformers.pipelines.base import Dataset


class JSONLDataset:
    def __init__(self, jsonl_file_path: str, image_directory_path: str) -> None:
        self.jsonl_file_path = jsonl_file_path
        self.image_directory_path = image_directory_path
        self.entries = self._load_entries()

    def _load_entries(self) -> list[dict[str, Any]]:
        entries = []
        with open(self.jsonl_file_path) as file:
            for line in file:
                data = json.loads(line)
                entries.append(data)
        return entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[Image.Image, dict[str, Any]]:
        if idx < 0 or idx >= len(self.entries):
            raise IndexError("Index out of range")

        entry = self.entries[idx]
        image_path = os.path.join(self.image_directory_path, entry["image"])
        try:
            image = Image.open(image_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file {image_path} not found.")
        else:
            return (image, entry)


class DetectionDataset(Dataset):
    def __init__(self, jsonl_file_path: str, image_directory_path: str) -> None:
        self.dataset = JSONLDataset(jsonl_file_path, image_directory_path)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        image, data = self.dataset[idx]
        prefix = data["prefix"]
        suffix = data["suffix"]
        return prefix, suffix, image
