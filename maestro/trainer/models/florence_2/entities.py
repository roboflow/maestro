import os
from dataclasses import dataclass
from typing import Optional, Literal, Union

import torch

from maestro.trainer.common.configuration.env import CUDA_DEVICE_ENV, DEFAULT_CUDA_DEVICE


LoraInitLiteral = Literal["gaussian", "olora", "pissa", "pissa_niter_[number of iters]", "loftq"]


DEFAULT_FLORENCE2_MODEL_ID = "microsoft/Florence-2-base-ft"
DEFAULT_FLORENCE2_MODEL_REVISION = "refs/pr/20"
DEVICE = os.getenv(CUDA_DEVICE_ENV, DEFAULT_CUDA_DEVICE)


@dataclass(frozen=True)
class TrainingConfiguration:
    dataset_location: str
    model_id_or_path: str = DEFAULT_FLORENCE2_MODEL_ID
    revision: str = DEFAULT_FLORENCE2_MODEL_REVISION
    device: torch.device = torch.device(DEVICE)
    transformers_cache_dir: Optional[str] = None
    training_epochs: int = 10
    optimiser: Literal["SGD", "adamw", "adam"] = "adamw"
    learning_rate: float = 1e-5
    lr_scheduler: Literal["linear", "cosine", "polynomial"] = "linear"
    train_batch_size: int = 4
    test_batch_size: Optional[int] = None
    loaders_workers: int = 0
    test_loaders_workers: Optional[int] = None
    lora_r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    bias: Literal["none", "all", "lora_only"] = "none"
    use_rslora: bool = True
    init_lora_weights: Union[bool, LoraInitLiteral] = "gaussian"
    training_dir: str = "./training/florence-2"
    max_checkpoints_to_keep: int = 3
    num_samples_to_visualise: int = 64
