import os
from typing import Any, Callable, Optional

from roboflow import Roboflow
from torch.utils.data import DataLoader, Dataset

from maestro.cli.env import ROBOFLOW_API_KEY_ENV
from maestro.trainer.common.datasets.jsonl import JSONLDataset
from maestro.trainer.common.datasets.roboflow import ROBOFLOW_PROJECT_TYPE_TO_DATASET_FORMAT, parse_roboflow_identifier


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
