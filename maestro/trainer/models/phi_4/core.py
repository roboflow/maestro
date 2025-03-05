import os
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Literal, Optional

import dacite
import lightning
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoProcessor

from maestro.trainer.common.callbacks import SaveCheckpoint
from maestro.trainer.common.datasets.core import create_data_loaders, resolve_dataset_path
from maestro.trainer.common.metrics import BaseMetric, MetricsTracker, parse_metrics, save_metric_plots
from maestro.trainer.common.training import MaestroTrainer
from maestro.trainer.common.utils.device import device_is_available, parse_device_spec
from maestro.trainer.common.utils.path import create_new_run_directory
from maestro.trainer.common.utils.seed import ensure_reproducibility
from maestro.trainer.logger import get_maestro_logger
from maestro.trainer.models.phi_4.checkpoints import (
    DEFAULT_PHI_4_MODEL_ID,
    DEFAULT_PHI_4_MODEL_REVISION,
    OptimizationStrategy,
    filter_audio_components,
    load_model,
    process_model_inputs,
    save_model,
)
from maestro.trainer.models.phi_4.loaders import evaluation_collate_fn, train_collate_fn

logger = get_maestro_logger()


@dataclass()
class Phi4Configuration:
    """
    Configuration for training the Phi-4 multimodal model.

    Attributes:
        dataset (str):
            Local path or Roboflow identifier. If not found locally, it will be resolved (and downloaded) automatically.
        model_id (str):
            Identifier for the Phi-4 model.
        revision (str):
            Model revision to use.
        device (str | torch.device):
            Device to run training on.
        optimization_strategy (Literal["lora", "qlora", "none"]):
            Strategy for optimizing the model parameters.
        cache_dir (Optional[str]):
            Directory to cache the model weights locally.
        use_flash_attention (bool):
            Whether to use Flash Attention 2 for faster training.
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
        system_message (Optional[str]):
            System message to include in prompts.
        max_new_tokens (int):
            Maximum number of new tokens generated during inference.
        random_seed (Optional[int]):
            Random seed for ensuring reproducibility. If None, no seeding is applied.
    """

    dataset: str
    model_id: str = DEFAULT_PHI_4_MODEL_ID
    revision: str = DEFAULT_PHI_4_MODEL_REVISION
    device: str | torch.device = "auto"
    optimization_strategy: Literal["lora", "qlora", "none"] = "lora"
    cache_dir: Optional[str] = None
    use_flash_attention: bool = True
    epochs: int = 10
    lr: float = 1e-5
    batch_size: int = 4
    accumulate_grad_batches: int = 8
    val_batch_size: Optional[int] = None
    num_workers: int = 0
    val_num_workers: Optional[int] = None
    output_dir: str = "./training/phi_4"
    metrics: list[BaseMetric] | list[str] = field(default_factory=list)
    system_message: Optional[str] = None
    max_new_tokens: int = 512
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


class Phi4Trainer(MaestroTrainer):
    """
    Trainer for fine-tuning the Phi-4 multimodal model.

    Attributes:
        processor (AutoProcessor): Tokenizer and processor for model inputs.
        model (AutoModelForCausalLM): Pre-trained Phi-4 model.
        train_loader (DataLoader): DataLoader for training data.
        valid_loader (DataLoader): DataLoader for validation data.
        config (Phi4Configuration): Configuration object containing training parameters.
    """

    def __init__(
        self,
        processor: AutoProcessor,
        model: AutoModelForCausalLM,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        config: Phi4Configuration,
    ):
        super().__init__(processor, model, train_loader, valid_loader)
        self.config = config

        self.train_metrics_tracker = MetricsTracker.init(metrics=["loss"])
        metrics = ["loss"]
        for metric in config.metrics:
            if isinstance(metric, BaseMetric):
                metrics += metric.describe()  # ensure mypy understands it's BaseMetric
        self.valid_metrics_tracker = MetricsTracker.init(metrics=metrics)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, input_image_embeds, image_attention_mask, image_sizes, labels, input_mode = batch

        outputs = process_model_inputs(
            model=self.model,
            inputs={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "input_image_embeds": input_image_embeds,
                "image_attention_mask": image_attention_mask,
                "image_sizes": image_sizes,
                "labels": labels,
                "input_mode": input_mode,  # Specific for Phi 4 as it as both audio and vision component
            },
        )

        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=self.config.batch_size)
        self.train_metrics_tracker.register("loss", epoch=self.current_epoch, step=batch_idx, value=loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        (
            images,
            prefixes,
            suffixes,
            input_ids,
            attention_mask,
            input_image_embeds,
            image_attention_mask,
            image_sizes,
            input_mode,
        ) = batch

        input_length = input_ids.shape[1]
        filtered_inputs = filter_audio_components(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "input_image_embeds": input_image_embeds,
                "image_attention_mask": image_attention_mask,
                "image_sizes": image_sizes,
                "input_mode": input_mode,
            }
        )

        filtered_inputs = {k: v.to(self.device) for k, v in filtered_inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **filtered_inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
            )

        generated_ids = outputs[:, input_length:]
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        if batch_idx == 0:
            logger.info(f"Sample validation prefix: {prefixes[0]}")
            logger.info(f"Sample ground truth: {suffixes[0]}")
            logger.info(f"Sample model output: {generated_texts[0]}")

        for metric in self.config.metrics:
            result = metric.compute(predictions=generated_texts, targets=suffixes)
            for key, value in result.items():
                self.valid_metrics_tracker.register(
                    metric=key,
                    epoch=self.current_epoch,
                    step=batch_idx,
                    value=value,
                )
                self.log(key, value, prog_bar=True, logger=True, batch_size=self.config.val_batch_size)

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


def train(config: Phi4Configuration | dict) -> None:
    """
    Trains the Phi-4 model based on the given configuration.

    Args:
        config (Phi4Configuration | dict): Training configuration or dictionary with configuration parameters.

    Returns:
        None
    """
    if isinstance(config, dict):
        config = dacite.from_dict(data_class=Phi4Configuration, data=config)
    assert isinstance(config, Phi4Configuration)  # ensure mypy understands it's not a dict

    ensure_reproducibility(seed=config.random_seed, avoid_non_deterministic_algorithms=False)
    run_dir = create_new_run_directory(base_output_dir=config.output_dir)
    config = replace(config, output_dir=run_dir)

    processor, model = load_model(
        model_id_or_path=config.model_id,
        revision=config.revision,
        device=config.device,
        optimization_strategy=OptimizationStrategy(config.optimization_strategy),
        cache_dir=config.cache_dir,
        use_flash_attention=config.use_flash_attention,
    )
    dataset_location = resolve_dataset_path(config.dataset)
    if dataset_location is None:
        return

    train_loader, valid_loader, test_loader = create_data_loaders(
        dataset_location=dataset_location,
        train_batch_size=config.batch_size,
        train_collect_fn=partial(train_collate_fn, processor=processor, system_message=config.system_message),
        train_num_workers=config.num_workers,
        test_batch_size=config.val_batch_size,
        test_collect_fn=partial(evaluation_collate_fn, processor=processor, system_message=config.system_message),
        test_num_workers=config.val_num_workers,
    )

    _, train_entry = train_loader.dataset[0]
    logger.info(f"Sample train prefix: {train_entry['prefix']}")
    logger.info(f"Sample train suffix: {train_entry.get('suffix', 'No suffix available')}")

    pl_module = Phi4Trainer(
        processor=processor, model=model, train_loader=train_loader, valid_loader=valid_loader, config=config
    )

    save_checkpoints_path = os.path.join(config.output_dir, "checkpoints")
    save_checkpoint_callback = SaveCheckpoint(result_path=save_checkpoints_path, save_model_callback=save_model)

    trainer = lightning.Trainer(
        max_epochs=config.epochs,
        accumulate_grad_batches=config.accumulate_grad_batches,
        check_val_every_n_epoch=1,
        limit_val_batches=1.0,
        log_every_n_steps=10,
        callbacks=[save_checkpoint_callback],
    )

    trainer.fit(pl_module)
