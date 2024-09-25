from transformers.pipelines.base import Dataset

from maestro.trainer.common.data_loaders.datasets import JSONLDataset


class Qwen2VLDataset(Dataset):
    def __init__(self, jsonl_file_path: str, image_directory_path: str) -> None:
        self.dataset = JSONLDataset(jsonl_file_path, image_directory_path)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        image, data = self.dataset[idx]
        prefix = data["prefix"]
        suffix = data["suffix"]
        # fmt: off
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prefix}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": suffix}
                    ]
                }
            ]
        }
        # fmt: on
