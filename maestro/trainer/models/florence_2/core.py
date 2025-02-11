import os
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Literal, Optional

import dacite
import lightning
import supervision as sv
import torch
from torch.optim import AdamW

from maestro.trainer.common.callbacks import SaveCheckpoint
from maestro.trainer.common.datasets.core import create_data_loaders, resolve_dataset_path
from maestro.trainer.common.metrics import (
    BaseMetric,
    MeanAveragePrecisionMetric,
    MetricsTracker,
    parse_metrics,
    save_metric_plots,
)
from maestro.trainer.common.training import MaestroTrainer
from maestro.trainer.common.utils.device import device_is_available, parse_device_spec
from maestro.trainer.common.utils.path import create_new_run_directory
from maestro.trainer.common.utils.seed import ensure_reproducibility
from maestro.trainer.models.florence_2.checkpoints import (
    DEFAULT_FLORENCE2_MODEL_ID,
    DEFAULT_FLORENCE2_MODEL_REVISION,
    OptimizationStrategy,
    load_model,
    save_model,
)
from maestro.trainer.models.florence_2.detection import (
    detections_to_prefix_formatter,
    detections_to_suffix_formatter,
    result_to_detections_formatter,
)
from maestro.trainer.models.florence_2.inference import predict_with_inputs
from maestro.trainer.models.florence_2.loaders import evaluation_collate_fn, train_collate_fn


@dataclass()
class Florence2Configuration:
    """
    Configuration for training the Florence-2 model.

    Attributes:
        dataset (str):
            Local path or Roboflow identifier. If not found locally, it will be resolved (and downloaded) automatically.
        model_id (str):
            Identifier for the Florence-2 model.
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
        max_new_tokens (int):
            Maximum number of new tokens generated during inference.
        random_seed (Optional[int]):
            Random seed for ensuring reproducibility. If None, no seeding is applied.
    """

    dataset: str
    model_id: str = DEFAULT_FLORENCE2_MODEL_ID
    revision: str = DEFAULT_FLORENCE2_MODEL_REVISION
    device: str | torch.device = "auto"
    optimization_strategy: Literal["lora", "freeze", "none"] = "lora"
    cache_dir: Optional[str] = None
    epochs: int = 10
    lr: float = 1e-5
    batch_size: int = 4
    accumulate_grad_batches: int = 8
    val_batch_size: Optional[int] = None
    num_workers: int = 0
    val_num_workers: Optional[int] = None
    output_dir: str = "./training/florence_2"
    metrics: list[BaseMetric] | list[str] = field(default_factory=list)
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


class Florence2Trainer(MaestroTrainer):
    """
    Trainer for fine-tuning the Florence-2 model.

    Attributes:
        processor (AutoProcessor): Processor for model inputs.
        model (AutoModelForCausalLM): The Florence-2 model.
        train_loader (DataLoader): DataLoader for training data.
        valid_loader (DataLoader): DataLoader for validation data.
        config (Florence2Configuration): Configuration object with training parameters.
    """

    def __init__(self, processor, model, train_loader, valid_loader, config):
        super().__init__(processor, model, train_loader, valid_loader)
        self.config = config

        # TODO: Redesign metric tracking system
        self.train_metrics_tracker = MetricsTracker.init(metrics=["loss"])
        metrics = ["loss"]
        for metric in config.metrics:
            if isinstance(metric, BaseMetric):
                metrics += metric.describe()
        self.valid_metrics_tracker = MetricsTracker.init(metrics=metrics)

    def training_step(self, batch, batch_idx):
        input_ids, pixel_values, labels = batch
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=labels,
        )
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=self.config.batch_size)
        self.train_metrics_tracker.register("loss", epoch=self.current_epoch, step=batch_idx, value=loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, pixel_values, images, prefixes, suffixes = batch
        generated_suffixes = predict_with_inputs(
            model=self.model,
            processor=self.processor,
            input_ids=input_ids,
            pixel_values=pixel_values,
            device=self.config.device,
            max_new_tokens=self.config.max_new_tokens,
        )
        for metric in self.config.metrics:
            if isinstance(metric, MeanAveragePrecisionMetric):
                predictions_list = []
                targets_list = []
                for image, generated_suffix, reference_suffix in zip(images, generated_suffixes, suffixes):
                    predicted_boxes, predicted_class_ids = result_to_detections_formatter(
                        text=generated_suffix, resolution_wh=image.size
                    )
                    reference_boxes, reference_class_ids = result_to_detections_formatter(
                        text=reference_suffix, resolution_wh=image.size
                    )
                    predictions_list.append(sv.Detections(xyxy=predicted_boxes, class_id=predicted_class_ids))
                    targets_list.append(sv.Detections(xyxy=reference_boxes, class_id=reference_class_ids))

                print("predictions_list", predictions_list)
                print("targets_list", targets_list)
                result = metric.compute(predictions=predictions_list, targets=targets_list)
                for key, value in result.items():
                    self.valid_metrics_tracker.register(
                        metric=key,
                        epoch=self.current_epoch,
                        step=batch_idx,
                        value=value,
                    )
                    self.log(key, value, prog_bar=True, logger=True)
            else:
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


def train(config: Florence2Configuration | dict) -> None:
    if isinstance(config, dict):
        config = dacite.from_dict(data_class=Florence2Configuration, data=config)
    assert isinstance(config, Florence2Configuration)  # ensure mypy understands it's not a dict

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
    dataset_location = resolve_dataset_path(config.dataset)
    if dataset_location is None:
        return
    train_loader, valid_loader, test_loader = create_data_loaders(
        dataset_location=dataset_location,
        train_batch_size=config.batch_size,
        train_collect_fn=partial(train_collate_fn, processor=processor),
        train_num_workers=config.num_workers,
        test_batch_size=config.val_batch_size,
        test_collect_fn=partial(evaluation_collate_fn, processor=processor),
        test_num_workers=config.val_num_workers,
        detections_to_prefix_formatter=detections_to_prefix_formatter,
        detections_to_suffix_formatter=detections_to_suffix_formatter,
    )
    pl_module = Florence2Trainer(
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
