from __future__ import annotations

import random
from typing import List

from torch.utils.data import Dataset

from maestro.trainer.common.utils.file_system import read_jsonl


class JSONLDataset(Dataset):
    # TODO: implementation could be better - avoiding loading
    #  whole files to memory

    @classmethod
    def from_jsonl_file(cls, path: str) -> JSONLDataset:
        file_content = read_jsonl(path=path)
        random.shuffle(file_content)
        return cls(jsons=file_content)

    def __init__(self, jsons: List[dict]):
        self.jsons = jsons

    def __getitem__(self, index):
        return self.jsons[index]

    def __len__(self):
        return len(self.jsons)

    def shuffle(self):
        random.shuffle(self.jsons)
