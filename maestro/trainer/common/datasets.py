import json
import os
from typing import Any, Callable, ClassVar, Optional

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from maestro.trainer.logger import get_maestro_logger

logger = get_maestro_logger()


class JSONLDataset(Dataset):
    """
    A dataset for loading images and annotations from a JSON Lines (JSONL) file.

    This class reads annotation entries from a specified JSONL file and ensures that each entry
    contains the required keys and that the corresponding image file exists in the given directory.
    Entries that fail validation (due to JSON parsing errors, missing keys, or non-existent image files)
    are skipped with an appropriate warning logged.

    Parameters:
        annotations_path (str): Filesystem path to the JSONL file containing dataset annotations.
        image_directory_path (str): Filesystem path to the directory containing image files.

    Example:
        ```
        from roboflow import download_dataset, login
        from maestro.trainer.common.datasets import RoboflowJSONLDataset

        login()

        dataset = download_dataset("universe.roboflow.com/roboflow-jvuqo/pallet-load-manifest-json/2", "jsonl")
        ds = RoboflowJSONLDataset(
            annotations_path=f"{dataset.location}/test/annotations.jsonl",
            image_directory_path=f"{dataset.location}/test"
        )
        len(ds)
        # 10
        ```
    """

    ROBOFLOW_JSONL_FILENAME: ClassVar[str] = "annotations.jsonl"
    REQUIRED_KEYS: ClassVar[set[str]] = {"image", "prefix", "suffix"}

    def __init__(self, annotations_path: str, image_directory_path: str) -> None:
        self.image_directory_path = image_directory_path
        self.entries = self._load_entries(annotations_path, image_directory_path)

    @classmethod
    def _load_entries(cls, annotations_path: str, image_dir: str) -> list[dict]:
        """
        Load and validate dataset entries from a JSON Lines (JSONL) file.

        Reads each line in the specified file, attempts to parse it as JSON, and verifies that
        every resulting entry contains the required keys. Additionally, it ensures that the
        associated image file exists in the given directory. Entries that cannot be parsed or do not
        meet the validation criteria are skipped with a warning.

        Parameters:
            annotations_path (str): Filesystem path to the JSONL file.
            image_dir (str): Filesystem path to the image directory.

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
                image_path = os.path.join(image_dir, entry["image"])
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

    def __getitem__(self, idx: int):
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


def load_split_dataset(dataset_location: str, split_name: str) -> Optional[Dataset]:
    """
    Load a dataset split from the specified location.

    Args:
        dataset_location (str): Path to the dataset directory.
        split_name (str): Name of the dataset split (e.g., "train", "valid", "test").

    Returns:
        Optional[Dataset]: A dataset object for the split, or `None` if the split does not exist.
    """
    annotations_path = os.path.join(dataset_location, split_name, JSONLDataset.ROBOFLOW_JSONL_FILENAME)
    image_directory_path = os.path.join(dataset_location, split_name)

    if not os.path.exists(annotations_path) or not os.path.exists(image_directory_path):
        print(f"Dataset split {split_name} not found at {dataset_location}")
        return None

    return JSONLDataset(annotations_path, image_directory_path)


def create_data_loaders(
    dataset_location: str,
    train_batch_size: int,
    train_collect_fn: Callable[[list[Any]], Any],
    train_num_workers: int = 0,
    test_batch_size: Optional[int] = None,
    test_collect_fn: Optional[Callable[[list[Any]], Any]] = None,
    test_num_workers: Optional[int] = None,
) -> tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Create DataLoader instances for training, validation, and testing datasets.

    Args:
        dataset_location (str): Path to the dataset directory.
        train_batch_size (int): Batch size for the training dataset. Must be a positive integer.
        train_collect_fn (Callable[[List[Any]], Any]): Function to collate training samples into a batch.
        train_num_workers (int): Number of worker threads for the training DataLoader. Defaults to 0.
        test_batch_size (Optional[int]): Batch size for validation and test datasets. Defaults to the value of
            `train_batch_size` if not provided.
        test_collect_fn (Optional[Callable[[List[Any]], Any]]): Function to collate validation and test samples into a
            batch. Defaults to the value of `train_collect_fn` if not provided.
        test_num_workers (Optional[int]): Number of worker threads for validation and test DataLoaders. Defaults to the
            value of `train_num_workers` if not provided.

    Returns:
        Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]: A tuple containing the DataLoader for the
            training dataset, and optionally for the validation and testing datasets. If a dataset split is not found,
            the corresponding DataLoader is `None`.

    Raises:
        ValueError: If batch sizes are not positive integers or no dataset splits are found.
    """
    if train_batch_size <= 0:
        raise ValueError("train_batch_size must be a positive integer.")

    test_batch_size = test_batch_size or train_batch_size
    if test_batch_size <= 0:
        raise ValueError("test_batch_size must be a positive integer.")

    test_num_workers = test_num_workers or train_num_workers
    test_collect_fn = test_collect_fn or train_collect_fn

    train_dataset = load_split_dataset(dataset_location, "train")
    valid_dataset = load_split_dataset(dataset_location, "valid")
    test_dataset = load_split_dataset(dataset_location, "test")

    if not any([train_dataset, valid_dataset, test_dataset]):
        raise ValueError(f"No dataset splits found at {dataset_location}. Ensure the dataset is correctly structured.")
    else:
        print(f"Found dataset splits at {dataset_location}:")
        if train_dataset:
            print(f"  - train: {len(train_dataset)} samples")
        if valid_dataset:
            print(f"  - valid: {len(valid_dataset)} samples")
        if test_dataset:
            print(f"  - test: {len(test_dataset)} samples")

    train_loader = (
        DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=train_num_workers,
            collate_fn=train_collect_fn,
        )
        if train_dataset
        else None
    )

    valid_loader = (
        DataLoader(
            valid_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=test_num_workers,
            collate_fn=test_collect_fn,
        )
        if valid_dataset
        else None
    )

    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=test_num_workers,
            collate_fn=test_collect_fn,
        )
        if test_dataset
        else None
    )

    return train_loader, valid_loader, test_loader
