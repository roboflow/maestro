from dataclasses import dataclass, field, replace
from functools import partial
from typing import Literal, Optional

import dacite
import torch

from maestro.trainer.common.datasets import create_data_loaders
from maestro.trainer.common.metrics import BaseMetric, parse_metrics
from maestro.trainer.common.utils.device import device_is_available, parse_device_spec
from maestro.trainer.common.utils.path import create_new_run_directory
from maestro.trainer.common.utils.seed import ensure_reproducibility
from maestro.trainer.models.paligemma_2.checkpoints import (
    DEFAULT_PALIGEMMA2_MODEL_ID,
    DEFAULT_PALIGEMMA2_MODEL_REVISION,
    OptimizationStrategy,
    load_model,
)
from maestro.trainer.models.paligemma_2.loaders import evaluation_collate_fn, train_collate_fn


@dataclass()
class PaliGemma2Configuration:
    """
    Configuration for training the PaliGemma2 model.

    Attributes:
        dataset (str):
            Path to the dataset used for training.
        model_id (str):
            Identifier for the PaliGemma2 model.
        revision (str):
            Model revision to use.
        device (str | torch.device):
            Device to run training on. Can be a ``torch.device`` or a string such as
            "auto", "cpu", "cuda", or "mps". If "auto", the code will pick the best
            available device.
        optimization_strategy (Literal["lora", "qlora", "freeze", "none"]):
            Strategy for optimizing the model parameters.
        cache_dir (Optional[str]):
            Directory to cache the model weights locally.
        epochs (int):
            Number of training epochs.
        lr (float):
            Learning rate for training.
        batch_size (int):
            Training batch size.
        accumulate_grad_batches (int):
            Number of batches to accumulate before performing a gradient update.
        val_batch_size (Optional[int]):
            Validation batch size. If None, defaults to the training batch size.
        num_workers (int):
            Number of workers for data loading.
        val_num_workers (Optional[int]):
            Number of workers for validation data loading. If None, defaults to num_workers.
        output_dir (str):
            Directory to store training outputs.
        metrics (list[BaseMetric] | list[str]):
            Metrics to track during training. Can be a list of metric objects or metric names.
        random_seed (Optional[int]):
            Random seed for ensuring reproducibility. If None, no seeding is applied.
    """

    dataset: str
    model_id: str = DEFAULT_PALIGEMMA2_MODEL_ID
    revision: str = DEFAULT_PALIGEMMA2_MODEL_REVISION
    device: str | torch.device = "auto"
    optimization_strategy: Literal["lora", "qlora", "freeze", "none"] = "lora"
    cache_dir: Optional[str] = None
    epochs: int = 10
    lr: float = 1e-5
    batch_size: int = 4
    accumulate_grad_batches: int = 8
    val_batch_size: Optional[int] = None
    num_workers: int = 0
    val_num_workers: Optional[int] = None
    output_dir: str = "./training/paligemma_2"
    metrics: list[BaseMetric] | list[str] = field(default_factory=list)
    random_seed: Optional[int] = None

    def __post_init__(self):
        if self.val_batch_size is None:
            self.val_batch_size = self.batch_size

        if self.val_num_workers is None:
            self.val_num_workers = self.num_workers

        if isinstance(self.metrics, list) and all(isinstance(m, str) for m in self.metrics):
            self.metrics = parse_metrics(self.metrics)

        self.device = parse_device_spec(self.device)
        if not device_is_available(self.device):
            raise ValueError(f"Requested device '{self.device}' is not available.")


def train(config: PaliGemma2Configuration | dict) -> None:
    if isinstance(config, dict):
        config = dacite.from_dict(data_class=PaliGemma2Configuration, data=config)
    assert isinstance(config, PaliGemma2Configuration)  # ensure mypy understands it's not a dict

    ensure_reproducibility(seed=config.random_seed, avoid_non_deterministic_algorithms=False)
    run_dir = create_new_run_directory(base_output_dir=config.output_dir)
    config = replace(config, output_dir=run_dir)

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
