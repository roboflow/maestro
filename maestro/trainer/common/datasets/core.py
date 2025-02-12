import os
from typing import Any, Callable, Optional

import numpy as np
from roboflow import Roboflow
from torch.utils.data import DataLoader, Dataset

from maestro.cli.env import ROBOFLOW_API_KEY_ENV
from maestro.trainer.common.datasets.coco import COCODataset, COCOVLMAdapter, is_coco_dataset
from maestro.trainer.common.datasets.jsonl import JSONLDataset, is_jsonl_dataset
from maestro.trainer.common.datasets.roboflow import ROBOFLOW_PROJECT_TYPE_TO_DATASET_FORMAT, parse_roboflow_identifier
from maestro.trainer.logger import get_maestro_logger

logger = get_maestro_logger()


def resolve_dataset_path(dataset_id: str) -> Optional[str]:
    """
    Resolves the dataset path from a given identifier or local path.

    This function first checks if the provided string corresponds to an existing local path.
    If not, it attempts to interpret the string as a Roboflow identifier, download the dataset
    using the Roboflow API, and then returns the local path where the dataset has been downloaded.
    The function requires that the 'ROBOFLOW_API_KEY' environment variable is set.

    Args:
        dataset_id (str): A local path to the dataset or a Roboflow identifier string.

    Returns:
        Optional[str]: The local path to the dataset if found or successfully downloaded;
        otherwise, None if the identifier could not be parsed.

    Raises:
        ValueError: If the Roboflow API key is missing, the dataset type is unsupported, or no dataset
        versions are available.
    """
    from maestro.trainer.logger import get_maestro_logger

    logger = get_maestro_logger()

    if os.path.exists(dataset_id):
        logger.info(f"Dataset found locally at: {dataset_id}")
        return dataset_id

    parsed = parse_roboflow_identifier(dataset_id)
    if parsed is None:
        return None

    workspace_id, project_id, dataset_version = parsed

    api_key = os.environ.get(ROBOFLOW_API_KEY_ENV)
    if not api_key:
        logger.error("Missing Roboflow API key: please set the 'ROBOFLOW_API_KEY' environment variable.")
        raise ValueError("Missing Roboflow API key: please set the 'ROBOFLOW_API_KEY' environment variable.")

    rf = Roboflow(api_key=api_key)
    workspace = rf.workspace(workspace_id)
    project = workspace.project(project_id)

    if project.type not in ROBOFLOW_PROJECT_TYPE_TO_DATASET_FORMAT:
        logger.error(f"Maestro does not support {project.type} Roboflow datasets.")
        raise ValueError(f"Maestro does not support {project.type} Roboflow datasets.")

    dataset_format = ROBOFLOW_PROJECT_TYPE_TO_DATASET_FORMAT[project.type]
    if dataset_version:
        version = project.version(dataset_version)
    else:
        versions = project.versions()
        if not versions:
            logger.error("No dataset versions available: project has not been versioned yet.")
            raise ValueError("No dataset versions available: project has not been versioned yet.")
        version = versions[0]

    logger.info("Starting download of dataset...")
    dataset = version.download(dataset_format)
    logger.info(f"Completed download of dataset at: {dataset.location}")
    return dataset.location


def create_data_loaders(
    dataset_location: str,
    train_batch_size: int,
    train_collect_fn: Callable[[list[Any]], Any],
    train_num_workers: int = 0,
    test_batch_size: Optional[int] = None,
    test_collect_fn: Optional[Callable[[list[Any]], Any]] = None,
    test_num_workers: Optional[int] = None,
    detections_to_prefix_formatter: Optional[
        Callable[[np.ndarray, np.ndarray, list[str], tuple[int, int]], str]
    ] = None,
    detections_to_suffix_formatter: Optional[
        Callable[[np.ndarray, np.ndarray, list[str], tuple[int, int]], str]
    ] = None,
) -> tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Create DataLoader instances for training, validation, and testing datasets.
    Now supports both JSONL and COCO, deciding which to load based on the dataset structure.
    If COCO is detected but these two formatter functions are not provided, an error is raised.

    Args:
        dataset_location (str): Path to the dataset directory.
        train_batch_size (int): Batch size for the training dataset. Must be a positive integer.
        train_collect_fn (Callable[[list[Any]], Any]): Collation function for the training set.
        train_num_workers (int): Number of worker threads for training.
        test_batch_size (Optional[int]): Batch size for val/test. Defaults to train_batch_size if not provided.
        test_collect_fn (Optional[Callable[[list[Any]], Any]]): Collation function for val/test. Defaults to
            train_collect_fn if not provided.
        test_num_workers (Optional[int]): Number of worker threads for val/test. Defaults to train_num_workers if not
            provided.
        detections_to_prefix_formatter (Optional[Callable]): Function mapping COCO detections → prefix text.
        detections_to_suffix_formatter (Optional[Callable]): Function mapping COCO detections → suffix text.

    Returns:
        tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
            (train_loader, valid_loader, test_loader). Any of them can be None if that split does not exist.

    Raises:
        ValueError:
            - If batch sizes are invalid.
            - If no splits are found.
            - If the dataset is COCO but the required detections_to_prefix_formatter and
              detections_to_suffix_formatter are missing.
            - If the dataset does not appear to be JSONL or COCO.
    """

    if train_batch_size <= 0:
        raise ValueError("train_batch_size must be a positive integer.")
    test_batch_size = test_batch_size or train_batch_size
    if test_batch_size <= 0:
        raise ValueError("test_batch_size must be a positive integer.")

    test_num_workers = test_num_workers or train_num_workers
    test_collect_fn = test_collect_fn or train_collect_fn

    logger.info(f"Creating data loaders from '{dataset_location}'...")

    is_jsonl = is_jsonl_dataset(dataset_location)
    is_coco = is_coco_dataset(dataset_location)

    if not is_jsonl and not is_coco:
        error_message = (
            f"Dataset format not recognized at '{dataset_location}'.\n"
            "Expected JSONL or COCO structure (annotations.jsonl or _annotations.coco.json)."
        )
        logger.error(error_message)
        raise ValueError(error_message)

    if is_jsonl:
        logger.info("Detected JSONL dataset format.")
    else:
        logger.info("Detected COCO dataset format.")

    def load_split(split_name: str) -> Optional[Dataset]:
        split_path = os.path.join(dataset_location, split_name)

        if is_jsonl:
            annotations_path = os.path.join(split_path, JSONLDataset.ROBOFLOW_JSONL_FILENAME)
            if os.path.isfile(annotations_path):
                logger.info(f"Found JSONL split '{split_name}' at: {annotations_path}")
                return JSONLDataset(annotations_path, split_path)
            else:
                logger.info(f"No JSONL found for split '{split_name}' at path: {annotations_path}")
                return None

        if is_coco:
            if not (detections_to_prefix_formatter and detections_to_suffix_formatter):
                error_message = (
                    "COCO dataset detected, but detections_to_prefix_formatter and detections_to_suffix_formatter "
                    "were not provided. This is required to produce prefix/suffix for the model."
                )
                logger.error(error_message)
                raise ValueError(error_message)

            annotations_path = os.path.join(split_path, COCODataset.ROBOFLOW_COCO_FILENAME)
            if os.path.isfile(annotations_path):
                logger.info(f"Found COCO split '{split_name}' at: {annotations_path}")
                base_ds = COCODataset(annotations_path, split_path)
                adapter = COCOVLMAdapter(
                    coco_dataset=base_ds,
                    prefix_formatter=detections_to_prefix_formatter,
                    suffix_formatter=detections_to_suffix_formatter,
                )
                return adapter
            else:
                logger.info(f"No COCO found for split '{split_name}' at path: {annotations_path}")
                return None

        return None

    train_dataset = load_split("train")
    valid_dataset = load_split("valid")
    test_dataset = load_split("test")

    if not all([train_dataset, valid_dataset, test_dataset]):
        error_message = (
            f"All dataset splits (train, valid, and test) must be present at {dataset_location}. "
            f"Ensure the dataset is correctly structured."
        )
        logger.error(error_message)
        raise ValueError(error_message)

    logger.info("Initializing data loaders...")

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

    logger.info("Data loaders created successfully.")
    return train_loader, valid_loader, test_loader
