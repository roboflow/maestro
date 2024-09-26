from transformers.pipelines.base import Dataset

from maestro.trainer.common.data_loaders.datasets import JSONLDataset

START_TOKEN_1 = 151644
START_TOKEN_2 = 77091
END_TOKEN = 151645


def extract_assistant_content_ranges(token_list: list[int]) -> list[tuple[int, int]]:
    """
    Identify the start and end indexes of assistant content ranges within a list of
    tokens.

    The function searches for sequences that mark the start and end of assistant content
    in the tokenized list, returning the corresponding index ranges.

    Args:
        token_list (list[int]): A list of tokens to search.

    Returns:
        list[tuple[int, int]]: A list of (start_index, end_index) tuples indicating the
        assistant content ranges in the input list.

    Note:
        - Assistant content starts with the sequence [START_TOKEN_1, START_TOKEN_2],
        which corresponds to the tokenized value of `"<|im_start|>assistant"`.
        - Assistant content ends with END_TOKEN, which corresponds to the tokenized
        value of `"<|im_end|>"`.
        - Each start sequence has a corresponding end token.
    """
    start_indexes = []
    end_indexes = []

    for i in range(len(token_list) - 1):
        if token_list[i] == START_TOKEN_1 and token_list[i + 1] == START_TOKEN_2:
            start_indexes.append(i)
            for j in range(i + 2, len(token_list)):
                if token_list[j] == END_TOKEN:
                    end_indexes.append(j)
                    break

    return list(zip(start_indexes, end_indexes))


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
