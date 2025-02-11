import json
import os
from typing import Any, ClassVar

from PIL import Image
from torch.utils.data import Dataset

from maestro.trainer.common.datasets.base import BaseVLDataset
from maestro.trainer.logger import get_maestro_logger

logger = get_maestro_logger()


class JSONLDataset(Dataset, BaseVLDataset):
    """
    A dataset for loading images and annotations from a JSON Lines (JSONL) file.

    This class reads annotation entries from a specified JSONL file and ensures that each entry
    contains the required keys and that the corresponding image file exists in the given directory.
    Entries that fail validation (due to JSON parsing errors, missing keys, or non-existent image files)
    are skipped with an appropriate warning logged.

    Parameters:
        annotations_path (str): Filesystem path to the JSONL file containing dataset annotations.
        images_directory_path (str): Filesystem path to the directory containing image files.

    Example:
        ```
        from roboflow import download_dataset, login
        from maestro.trainer.common.datasets.jsonl import JSONLDataset

        login()

        dataset = download_dataset("universe.roboflow.com/roboflow-jvuqo/pallet-load-manifest-json/2", "jsonl")
        ds = JSONLDataset(
            annotations_path=f"{dataset.location}/test/annotations.jsonl",
            images_directory_path=f"{dataset.location}/test"
        )
        len(ds)
        # 10
        ```
    """

    ROBOFLOW_JSONL_FILENAME: ClassVar[str] = "annotations.jsonl"
    REQUIRED_KEYS: ClassVar[set[str]] = {"image", "prefix", "suffix"}

    def __init__(self, annotations_path: str, images_directory_path: str) -> None:
        self.image_directory_path = images_directory_path
        self.entries = self._load_entries(annotations_path, images_directory_path)

    @classmethod
    def _load_entries(cls, annotations_path: str, images_dir: str) -> list[dict]:
        """
        Load and validate dataset entries from a JSON Lines (JSONL) file.

        Reads each line in the specified file, attempts to parse it as JSON, and verifies that
        every resulting entry contains the required keys. Additionally, it ensures that the
        associated image file exists in the given directory. Entries that cannot be parsed or do not
        meet the validation criteria are skipped with a warning.

        Parameters:
            annotations_path (str): Filesystem path to the JSONL file.
            images_dir (str): Filesystem path to the image directory.

        Returns:
            list[dict]: A list of valid annotation dictionaries.
        """
        if not os.path.isfile(annotations_path):
            logger.warning(f"Annotations file does not exist: '{annotations_path}'")
            return []

        entries = []
        total_lines = 0
        skipped_count = 0

        with open(annotations_path, encoding="utf-8") as file:
            for line_idx, line in enumerate(file, start=1):
                total_lines += 1
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as e:
                    skipped_count += 1
                    logger.warning(f"Skipping line {line_idx} (JSON parse error): {e}")
                    continue
                missing_keys = cls.REQUIRED_KEYS - entry.keys()
                if missing_keys:
                    skipped_count += 1
                    logger.warning(f"Skipping line {line_idx}: missing key(s) {missing_keys}")
                    continue
                image_path = os.path.join(images_dir, entry["image"])
                if not os.path.exists(image_path):
                    skipped_count += 1
                    logger.warning(f"Skipping line {line_idx}: image file not found '{image_path}'")
                    continue
                entries.append(entry)

        loaded_count = total_lines - skipped_count
        if total_lines > 0:
            logger.info(
                f"Loaded {loaded_count} valid entries out of {total_lines} "
                f"from '{annotations_path}'. Skipped {skipped_count}."
            )
        else:
            logger.warning(f"No lines found in '{annotations_path}'.")

        return entries

    def __len__(self) -> int:
        """
        Return the number of valid entries in the dataset.

        Returns:
            int: Total count of dataset entries.
        """
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[Image.Image, dict[str, Any]]:
        """
        Retrieve the image and its corresponding annotation entry at the specified index.

        Parameters:
            idx (int): The zero-based index of the desired entry.

        Returns:
            tuple: A tuple containing:
                - PIL.Image.Image: The image object.
                - dict: The corresponding annotation entry.

        Raises:
            IndexError: If the index is out of the valid range.
        """
        if idx >= len(self.entries):
            raise IndexError(f"Index {idx} is out of range.")
        entry = self.entries[idx]
        image_path = os.path.join(self.image_directory_path, entry["image"])
        image = Image.open(image_path).convert("RGB")
        return image, entry


def is_jsonl_dataset(dataset_location: str) -> bool:
    """
    Checks if a directory structure matches the JSONL dataset format. A JSONL dataset
    is typically organized with separate subdirectories (e.g. `train`, `valid`, `test`),
    each containing an `annotations.jsonl` file and corresponding images:

    Expected structure (example):
    ```
    dataset/
    ├── train/
    │   ├── annotations.jsonl
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── valid/
    │   ├── annotations.jsonl
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── test/
        ├── annotations.jsonl
        ├── image1.jpg
        ├── image2.jpg
        └── ...
    ```

    This function checks for at least one subdirectory named `train`, `valid`, or `test`
    that contains an `annotations.jsonl` file. If found, it returns `True`; otherwise,
    `False`.

    Args:
        dataset_location (str): The path to the dataset directory.

    Returns:
        bool: `True` if the dataset follows the JSONL format; otherwise, `False`.
    """
    splits = ["train", "valid", "test"]
    for split in splits:
        annotations_path = os.path.join(dataset_location, split, "annotations.jsonl")
        if os.path.isfile(annotations_path):
            return True
    return False
