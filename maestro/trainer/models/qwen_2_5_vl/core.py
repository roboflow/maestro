import os
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Literal, Optional

import dacite
import lightning
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

from maestro.trainer.common.callbacks import SaveCheckpoint
from maestro.trainer.common.datasets import create_data_loaders
from maestro.trainer.common.metrics import BaseMetric, MetricsTracker, parse_metrics, save_metric_plots
from maestro.trainer.common.training import MaestroTrainer
from maestro.trainer.common.utils.device import device_is_available, parse_device_spec
from maestro.trainer.common.utils.path import create_new_run_directory
from maestro.trainer.common.utils.seed import ensure_reproducibility
from maestro.trainer.models.qwen_2_5_vl.checkpoints import (
    DEFAULT_QWEN2_5_VL_MODEL_ID,
    DEFAULT_QWEN2_5_VL_MODEL_REVISION,
    OptimizationStrategy,
    load_model,
    save_model,
)
from maestro.trainer.models.qwen_2_5_vl.inference import predict_with_inputs
from maestro.trainer.models.qwen_2_5_vl.loaders import evaluation_collate_fn, train_collate_fn


@dataclass()
class Qwen25VLConfiguration:
    """
    Configuration for training the Qwen2.5-VL model.

    Attributes:
        dataset (str): Path to the dataset in Roboflow JSONL format.
        model_id (str): Identifier for the Qwen2.5-VL model from HuggingFace Hub.
        revision (str): Model revision to use.
        device (torch.device): Device to run training on.
        optimization_strategy (Literal["lora", "qlora", "none"]): Optimization strategy.
        epochs (int): Number of training epochs.
        lr (float): Learning rate for training.
        batch_size (int): Training batch size.
        accumulate_grad_batches (int): Number of batches to accumulate for gradient updates.
        val_batch_size (Optional[int]): Validation batch size.
        num_workers (int): Number of workers for data loading.
        val_num_workers (Optional[int]): Number of workers for validation data loading.
        output_dir (str): Directory to store training outputs.
        metrics (list[BaseMetric] | list[str]): Metrics to track during training.
        system_message (Optional[str]): System message used during data loading.
        min_pixels (int): Minimum number of pixels for input images.
        max_pixels (int): Maximum number of pixels for input images.
        max_new_tokens (int): Maximum number of new tokens generated during inference.
        random_seed (Optional[int]): Random seed for ensuring reproducibility. If `None`, no seed is set.
    """

    dataset: str
    model_id: str = DEFAULT_QWEN2_5_VL_MODEL_ID
    revision: str = DEFAULT_QWEN2_5_VL_MODEL_REVISION
    device: str | torch.device = "auto"
    optimization_strategy: Literal["lora", "qlora", "none"] = "lora"
    epochs: int = 10
    lr: float = 2e-4
    batch_size: int = 4
    accumulate_grad_batches: int = 8
    val_batch_size: Optional[int] = None
    num_workers: int = 0
    val_num_workers: Optional[int] = None
    output_dir: str = "./training/qwen_2_5_vl"
    metrics: list[BaseMetric] | list[str] = field(default_factory=list)
    system_message: Optional[str] = None
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 1280 * 28 * 28
    max_new_tokens: int = 1024
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


class Qwen25VLTrainer(MaestroTrainer):
    """
    Trainer for fine-tuning the Qwen2.5-VL model.

    Attributes:
        processor (Qwen2_5_VLProcessor): Tokenizer and processor for model inputs.
        model (Qwen2_5_VLForConditionalGeneration): Pre-trained Qwen2.5-VL model.
        train_loader (DataLoader): DataLoader for training data.
        valid_loader (DataLoader): DataLoader for validation data.
        config (Qwen25VLConfiguration): Configuration object containing training parameters.
    """

    def __init__(
        self,
        processor: Qwen2_5_VLProcessor,
        model: Qwen2_5_VLForConditionalGeneration,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        config: Qwen25VLConfiguration,
    ):
        super().__init__(processor, model, train_loader, valid_loader)
        self.config = config

        # TODO: Redesign metric tracking system
        self.train_metrics_tracker = MetricsTracker.init(metrics=["loss"])
        metrics = ["loss"]
        for metric in config.metrics:
            if isinstance(metric, BaseMetric):
                metrics += metric.describe()  # ensure mypy understands it's BaseMetric
        self.valid_metrics_tracker = MetricsTracker.init(metrics=metrics)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values, image_grid_thw, labels = batch
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
        )
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=self.config.batch_size)
        self.train_metrics_tracker.register("loss", epoch=self.current_epoch, step=batch_idx, value=loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values, image_grid_thw, prefixes, suffixes = batch
        generated_suffixes = predict_with_inputs(
            model=self.model,
            processor=self.processor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            device=self.config.device,
        )

        for metric in self.config.metrics:
            result = metric.compute(predictions=generated_suffixes, targets=suffixes)
            for key, value in result.items():
                self.valid_metrics_tracker.register(
                    metric=key,
                    epoch=self.current_epoch,
                    step=batch_idx,
                    value=value,
                )
                self.log(key, value, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.config.lr)
        return optimizer

    def on_fit_end(self) -> None:
        save_metrics_path = os.path.join(self.config.output_dir, "metrics")
        save_metric_plots(
            training_tracker=self.train_metrics_tracker,
            validation_tracker=self.valid_metrics_tracker,
            output_dir=save_metrics_path,
        )


def train(config: Qwen25VLConfiguration | dict) -> None:
    """
    Trains the Qwen2.5-VL model based on the given configuration.

    Args:
        config (Qwen25VLConfiguration | dict): Training configuration or dictionary with configuration parameters.

    Returns:
        None
    """
    if isinstance(config, dict):
        config = dacite.from_dict(data_class=Qwen25VLConfiguration, data=config)
    assert isinstance(config, Qwen25VLConfiguration)  # ensure mypy understands it's not a dict

    ensure_reproducibility(seed=config.random_seed, avoid_non_deterministic_algorithms=False)
    run_dir = create_new_run_directory(base_output_dir=config.output_dir)
    config = replace(config, output_dir=run_dir)

    processor, model = load_model(
        model_id_or_path=config.model_id,
        revision=config.revision,
        device=config.device,
        optimization_strategy=OptimizationStrategy(config.optimization_strategy),
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
    )
    train_loader, valid_loader, test_loader = create_data_loaders(
        dataset_location=config.dataset,
        train_batch_size=config.batch_size,
        train_collect_fn=partial(train_collate_fn, processor=processor, system_message=config.system_message),
        train_num_workers=config.num_workers,
        test_batch_size=config.val_batch_size,
        test_collect_fn=partial(evaluation_collate_fn, processor=processor, system_message=config.system_message),
        test_num_workers=config.val_num_workers,
    )
    pl_module = Qwen25VLTrainer(
        processor=processor, model=model, train_loader=train_loader, valid_loader=valid_loader, config=config
    )
    save_checkpoints_path = os.path.join(config.output_dir, "checkpoints")
    save_checkpoint_callback = SaveCheckpoint(result_path=save_checkpoints_path, save_model_callback=save_model)
    trainer = lightning.Trainer(
        max_epochs=config.epochs,
        accumulate_grad_batches=config.accumulate_grad_batches,
        check_val_every_n_epoch=1,
        limit_val_batches=1,
        log_every_n_steps=10,
        callbacks=[save_checkpoint_callback],
    )
    trainer.fit(pl_module)
