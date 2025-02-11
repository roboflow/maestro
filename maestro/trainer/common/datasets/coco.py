import os
from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
import supervision as sv
from PIL import Image
from supervision.dataset.formats.coco import load_coco_annotations
from torch.utils.data import Dataset

from maestro.trainer.common.datasets.base import BaseDetectionDataset, BaseVLDataset
from maestro.trainer.logger import get_maestro_logger

logger = get_maestro_logger()


class COCODataset(Dataset, BaseDetectionDataset):
    """
    A dataset for loading images and COCO-format annotations from a JSON file.

    This class reads annotation entries from a specified COCO annotations file and ensures that each image
    referenced in the file exists in the supplied image directory. It leverages the standard COCO format to
    extract the list of classes, image file paths, and associated detection annotations. Entries that fail
    validation (for example, due to missing image files) are skipped with appropriate warnings logged.

    Parameters:
        annotations_path (str): Filesystem path to the COCO annotations JSON file.
        images_directory_path (str): Filesystem path to the directory where image files are stored.

    Example:
        ```
        from roboflow import download_dataset, login
        from maestro.trainer.common.datasets.coco import COCODataset

        login()

        dataset = download_dataset("universe.roboflow.com/huyifei/tft-id/1", "coco")
        ds = COOCDataset(
            annotations_path=f"{dataset.location}/test/_annotations.jsonl",
            images_directory_path=f"{dataset.location}/test"
        )
        len(ds)
        # 430
        ```
    """

    ROBOFLOW_COCO_FILENAME: ClassVar[str] = "_annotations.coco.json"
    REQUIRED_KEYS: ClassVar[list[str]] = ["images", "annotations", "categories"]

    def __init__(self, annotations_path: str, images_directory_path: str) -> None:
        self.images_directory_path = images_directory_path
        self.classes, self.entries = self._load_entries(annotations_path, images_directory_path)

    @classmethod
    def _load_entries(cls, annotations_path: str, images_dir: str) -> tuple[list[str], list[tuple[str, sv.Detections]]]:
        if not os.path.isfile(annotations_path):
            logger.warning(f"Annotations file does not exist: '{annotations_path}'")
            return [], []

        try:
            classes, images, annotation_dict = load_coco_annotations(
                images_directory_path=images_dir, annotations_path=annotations_path, force_masks=False
            )
        except Exception as e:
            logger.warning(f"Could not parse annotations file '{annotations_path}': {e}")
            return [], []

        total_images = len(images)
        skipped_count = 0
        empty_detections_count = 0
        valid_entries: list[tuple[str, sv.Detections]] = []

        for image_path in images:
            if not os.path.exists(image_path):
                skipped_count += 1
                logger.warning(f"Skipping file: image file not found '{image_path}'")
                continue

            detections = annotation_dict[image_path]
            if detections.xyxy.shape[0] == 0:
                empty_detections_count += 1

            valid_entries.append((image_path, detections))

        loaded_count = total_images - skipped_count
        if total_images > 0:
            logger.info(
                f"Loaded {loaded_count} valid entries out of {total_images} from '{annotations_path}'. "
                f"Skipped {skipped_count}. Found {empty_detections_count} entries with empty detections."
            )
        else:
            logger.warning(f"No images found in '{annotations_path}'.")

        return classes, valid_entries

    def __len__(self) -> int:
        """
        Return the number of valid entries in the dataset.

        Returns:
            int: Total count of dataset entries.
        """
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[Image.Image, sv.Detections]:
        """
        Retrieve the image and its corresponding annotation entry at the specified index.

        Parameters:
            idx (int): The zero-based index of the desired entry.

        Returns:
            tuple: A tuple containing:
                - PIL.Image.Image: The image object.
                - sv.Detections: The corresponding annotation entry.

        Raises:
            IndexError: If the index is out of the valid range.
        """
        if idx >= len(self.entries):
            raise IndexError(f"Index {idx} is out of range.")
        image_path, detections = self.entries[idx]
        image = Image.open(image_path).convert("RGB")
        return image, detections


class COCOVLMAdapter(Dataset, BaseVLDataset):
    """
    A dataset adapter for converting a COCO-format dataset into a vision-language dataset format.

    This adapter wraps a COCODataset instance and applies specified formatting functions to the detection
    annotations to generate prefix and suffix components for vision-language tasks. Each entry in the
    dataset is a tuple containing the image and a dictionary with formatted "prefix" and "suffix" values.

    Parameters:
        coco_dataset (COCODataset): An instance of the COCO dataset containing images and detection annotations.
        prefix_formatter (Callable[[np.ndarray, np.ndarray, list[str], tuple[int, int]]]):
            A function that formats raw detection data (bounding boxes, class IDs, class names, image size) to produce
            the prefix.
        suffix_formatter (Callable[[np.ndarray, np.ndarray, list[str], tuple[int, int]]]):
            A function that formats raw detection data (bounding boxes, class IDs, class names, image size) to produce
            the suffix.
    """

    def __init__(
        self,
        coco_dataset: COCODataset,
        prefix_formatter: Callable[[np.ndarray, np.ndarray, list[str], tuple[int, int]], str],
        suffix_formatter: Callable[[np.ndarray, np.ndarray, list[str], tuple[int, int]], str],
    ):
        self.coco_dataset = coco_dataset
        self.prefix_formatter = prefix_formatter
        self.suffix_formatter = suffix_formatter

    def __len__(self) -> int:
        """
        Return the number of valid entries in the vision-language dataset.

        Returns:
            int: The total count of entries in the dataset.
        """
        return len(self.coco_dataset)

    def __getitem__(self, idx: int) -> tuple[Image.Image, dict[str, Any]]:
        """
        Retrieve the image and its corresponding formatted entry at the specified index.

        Parameters:
            idx (int): The zero-based index of the desired entry.

        Returns:
            tuple: A tuple containing:
                - PIL.Image.Image: The image object.
                - dict[str, Any]: A dictionary with keys "prefix" and "suffix" representing formatted detection data.

        Raises:
            IndexError: If the index is out of the valid range.
        """
        image, detections = self.coco_dataset[idx]
        entry = {
            "prefix": self.prefix_formatter(
                detections.xyxy, detections.class_id, self.coco_dataset.classes, image.size
            ),
            "suffix": self.suffix_formatter(
                detections.xyxy, detections.class_id, self.coco_dataset.classes, image.size
            ),
        }
        return image, entry


def is_coco_dataset(dataset_location: str) -> bool:
    """
    Checks if a directory structure matches the COCO dataset format. A COCO dataset
    is typically organized with separate subdirectories (e.g. `train`, `valid`, `test`),
    each containing a `_annotations.coco.json` file and corresponding images:

    Expected structure (example):
    ```
    dataset/
    ├── train/
    │   ├── _annotations.coco.json
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── valid/
    │   ├── _annotations.coco.json
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── test/
        ├── _annotations.coco.json
        ├── image1.jpg
        ├── image2.jpg
        └── ...
    ```

    This function checks for at least one subdirectory named `train`, `valid`, or `test`
    that contains an `_annotations.coco.json` file. If found, it returns `True`; otherwise,
    `False`.

    Args:
        dataset_location (str): The path to the dataset directory.

    Returns:
        bool: `True` if the dataset follows the COCO format; otherwise, `False`.
    """
    splits = ["train", "valid", "test"]
    for split in splits:
        annotations_path = os.path.join(dataset_location, split, "_annotations.coco.json")
        if os.path.isfile(annotations_path):
            return True
    return False
