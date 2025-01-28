from dataclasses import dataclass, field, replace
from functools import partial
from typing import Literal, Optional

import torch

from maestro.trainer.common.data_loaders.datasets import create_data_loaders
from maestro.trainer.common.utils.file_system import create_new_run_directory
from maestro.trainer.common.utils.metrics import BaseMetric
from maestro.trainer.common.utils.reproducibility import make_it_reproducible
from maestro.trainer.models.paligemma_2.checpoints import (
    DEFAULT_PALIGEMMA2_MODEL_ID,
    DEFAULT_PALIGEMMA2_MODEL_REVISION,
    DEVICE, load_model, OptimizationStrategy,
)
from maestro.trainer.models.paligemma_2.loaders import train_collate_fn, evaluation_collate_fn


@dataclass(frozen=True)
class Configuration:
    """
    Configuration for a Paligemma 2 model.

    This class encapsulates all the parameters needed for training a Paligemma 2 model,
    including dataset paths, model specifications, training hyperparameters, and output
    settings.

    Attributes:
        dataset (str): Path to the dataset used for training.
        model_id (str): Identifier for the Paligemma 2 model.
        revision (str): Revision of the model to use.
        device (torch.device): Device to use for training.
        cache_dir (Optional[str]): Directory to cache the model.
        epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        val_batch_size (Optional[int]): Batch size for validation.
        num_workers (int): Number of workers for data loading.
        val_num_workers (Optional[int]): Number of workers for validation data loading.
        output_dir (str): Directory to save output files.
        metrics (List[BaseMetric]): List of metrics to track during training.
    """

    dataset: str
    model_id: str = DEFAULT_PALIGEMMA2_MODEL_ID
    revision: str = DEFAULT_PALIGEMMA2_MODEL_REVISION
    device: torch.device = DEVICE
    optimization_strategy: Literal["lora", "qlora", "freeze", "none"] = "lora"
    cache_dir: Optional[str] = None
    epochs: int = 10
    lr: float = 1e-5
    batch_size: int = 4
    val_batch_size: Optional[int] = None
    num_workers: int = 0
    val_num_workers: Optional[int] = None

    output_dir: str = "./training/paligemma2"
    metrics: list[BaseMetric] = field(default_factory=list)


def train(config: Configuration) -> None:
    """Train a PaliGemma 2 model using the provided configuration.

    This function sets up the training environment, prepares the model and data loaders,
    and runs the training loop. It also handles metric tracking and checkpoint saving.

    Args:
        config (Configuration): The configuration object containing all necessary
            parameters for training.

    Returns:
        None

    Raises:
        ValueError: If an unsupported optimizer is specified in the configuration.
    """
    make_it_reproducible(avoid_non_deterministic_algorithms=False)
    run_dir = create_new_run_directory(
        base_output_dir=config.output_dir,
    )
    config = replace(
        config,
        output_dir=run_dir,
    )

    processor, model = load_model(
        model_id_or_path=config.model_id,
        revision=config.revision,
        device=config.device,
        optimization_strategy=OptimizationStrategy(config.optimization_strategy),
        cache_dir=config.cache_dir,
    )
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_location=config.dataset,
        train_batch_size=config.batch_size,
        train_collect_fn=partial(train_collate_fn, processor=processor),
        train_num_workers=config.num_workers,
        test_batch_size=config.val_batch_size,
        test_collect_fn=partial(evaluation_collate_fn, processor=processor),
        test_num_workers=config.val_num_workers,
    )