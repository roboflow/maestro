import os
from typing import Optional, Union

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, Trainer

import lightning
import dacite
from functools import partial


import numpy as np
import supervision as sv
from maestro.trainer.common.callbacks import SaveCheckpoint
from maestro.trainer.common.datasets.core import create_data_loaders, resolve_dataset_path
from maestro.trainer.common.metrics import BaseMetric, MetricsTracker, parse_metrics, save_metric_plots
from maestro.trainer.common.training import MaestroTrainer
from maestro.trainer.common.utils.device import device_is_available, parse_device_spec
from maestro.trainer.common.utils.path import create_new_run_directory
from maestro.trainer.common.utils.seed import ensure_reproducibility
from maestro.trainer.logger import get_maestro_logger
from maestro.trainer.models.smolvlm2.checkpoints import (
    DEFAULT_SMOLVLM2_MODEL_ID,
    DEFAULT_SMOLVLM2_MODEL_REVISION,
    OptimizationStrategy,
    load_model,
    save_model,
)
from maestro.trainer.models.smolvlm2.inference import predict_with_inputs
from maestro.trainer.models.smolvlm2.loaders import evaluation_collate_fn, train_collate_fn
from typing import Literal, Optional
from dataclasses import dataclass, field, replace
from torch.utils.data import DataLoader
from torch.optim import AdamW
from maestro.trainer.common.metrics import (
    BaseMetric,
    MeanAveragePrecisionMetric,
    MetricsTracker,
    parse_metrics,
    save_metric_plots,
)
from maestro.trainer.models.florence_2.detection import (
    detections_to_prefix_formatter,
    detections_to_suffix_formatter,
    result_to_detections_formatter,
)
logger = get_maestro_logger()


@dataclass()
class SmolVLM2Configuration:
    """
    Configuration for training the SmolVLM2 model.

    Attributes:
        dataset (str):
            Local path or Roboflow identifier. If not found locally, it will be resolved (and downloaded) automatically.
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
        max_new_tokens (int):
            Maximum number of new tokens generated during inference.
        random_seed (Optional[int]):
            Random seed for ensuring reproducibility. If None, no seeding is applied.
        peft_advanced_params (Optional[dict]):
            Custom LoRA configuration . If None, default configuration is applied.
    """

    dataset: str
    model_id: str = DEFAULT_SMOLVLM2_MODEL_ID
    revision: str = DEFAULT_SMOLVLM2_MODEL_REVISION
    device: str | torch.device = "auto"
    optimization_strategy: Literal["lora", "qlora", "freeze", "none"] = "lora"
    cache_dir: Optional[str] = None
    epochs: int = 10
    lr: float = 2e-5
    batch_size: int = 4
    accumulate_grad_batches: int = 4
    val_batch_size: Optional[int] = None
    num_workers: int = 0
    val_num_workers: Optional[int] = None
    output_dir: str = "./training/smol_vlm_2"
    metrics: list[BaseMetric] | list[str] = field(default_factory=list)
    max_new_tokens: int = 512
    random_seed: Optional[int] = None
    peft_advanced_params: Optional[dict] = None

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


class SmolVLM2Trainer(MaestroTrainer):
    """
    Trainer for fine-tuning the SmolVLM-2 model.

    Attributes:
        processor (AutoProcessor): Processor for model inputs.
        model (AutoModelForImageTextToText): The SmolVLM-2 model.
        train_loader (DataLoader): DataLoader for training data.
        valid_loader (DataLoader): DataLoader for validation data.
        config (SmolVLM2Configuration): Configuration object with training parameters.
    """

    def __init__(
        self,
        processor: AutoProcessor,
        model: AutoModelForVision2Seq,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        config: SmolVLM2Configuration,
    ):
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

        if batch_idx == 0:
            logger.info(f"sample valid prefix: {prefixes[0]}")
            logger.info(f"sample valid suffix: {suffixes[0]}")
            logger.info(f"sample generated suffix: {generated_suffixes[0]}")

        for metric in self.config.metrics:
            result = metric.compute(predictions=generated_suffixes, targets=suffixes)
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






def train(config: SmolVLM2Configuration | dict) -> None:
    if isinstance(config, dict):
        config = dacite.from_dict(data_class=SmolVLM2Configuration, data=config)
    assert isinstance(config, SmolVLM2Configuration)  # ensure mypy understands it's not a dict

    ensure_reproducibility(seed=config.random_seed, avoid_non_deterministic_algorithms=False)
    run_dir = create_new_run_directory(base_output_dir=config.output_dir)
    config = replace(config, output_dir=run_dir)

    processor, model = load_model(
        model_id_or_path=config.model_id,
        revision=config.revision,
        device=config.device,
        optimization_strategy=OptimizationStrategy(config.optimization_strategy),
        peft_advanced_params=config.peft_advanced_params,
        cache_dir=config.cache_dir,
    )
    dataset_location = resolve_dataset_path(config.dataset)
    if dataset_location is None:
        return
    train_loader, valid_loader, test_loader = create_data_loaders(
        dataset_location=dataset_location,
        train_batch_size=config.batch_size,
        train_collect_fn=partial(train_collate_fn, processor=processor, max_length=config.max_new_tokens),
        train_num_workers=config.num_workers,
        test_batch_size=config.val_batch_size,
        test_collect_fn=partial(evaluation_collate_fn, processor=processor),
        test_num_workers=config.val_num_workers,
    )

    _, train_entry = train_loader.dataset[0]
    logger.info(f"sample train prefix: {train_entry['prefix']}")
    logger.info(f"sample train suffix: {train_entry['suffix']}")

    pl_module = SmolVLM2Trainer(
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












# class SmolVLM2Core:
#     """Core SmolVLM2 model implementation."""

#     def __init__(
#         self,
#         model_name: str = "smol-ai/smolvlm2-500m",
#         device: str = "cuda" if torch.cuda.is_available() else "cpu",
#         **kwargs,
#     ):
#         """
#         Initialize SmolVLM2 model.

#         Args:
#             model_name: Name or path of the model to load
#             device: Device to run the model on
#             **kwargs: Additional arguments to pass to the model
#         """
#         self.model_name = model_name
#         self.device = device

#         self.processor = AutoProcessor.from_pretrained(model_name)
#         self.model = AutoModelForVision2Seq.from_pretrained(model_name)
#         self.model.to(device)

#     def process_inputs(self, images: Union[str, list[str]], prompt: Optional[str] = None) -> dict:
#         """Process input images and text."""
#         if isinstance(images, str):
#             images = [images]

#         return self.processor(images=images, text=prompt if prompt else "", return_tensors="pt").to(self.device)

#     def generate(self, inputs: dict, max_new_tokens: int = 512, **kwargs) -> torch.Tensor:
#         """Generate text from processed inputs."""
#         return self.model.generate(**inputs, max_new_tokens=max_new_tokens, **kwargs)

#     def decode_outputs(self, outputs: torch.Tensor, skip_special_tokens: bool = True) -> list[str]:
#         """Decode model outputs to text."""
#         return self.processor.batch_decode(outputs, skip_special_tokens=skip_special_tokens)


# def train(config: dict) -> dict:
#     """
#     Train SmolVLM2 model with provided configuration.

#     Args:
#         config: Dictionary containing training configuration
#             - dataset: Path to dataset directory or file
#             - epochs: Number of training epochs
#             - batch_size: Training batch size
#             - optimization_strategy: Strategy for optimization (qlora, lora, freeze_vision)
#             - metrics: List of metrics to evaluate during training
#             - output_dir: Directory to save trained model
#     Returns:
#         Dictionary containing training results and metrics
#     """

#     from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
#     from transformers import BitsAndBytesConfig, TrainingArguments

#     from maestro.trainer.common.datasets.core import create_data_loaders, resolve_dataset_path
#     from maestro.trainer.models.smolvlm2.loaders import evaluation_collate_fn, train_collate_fn

#     # Load dataset
#     dataset_path = config["dataset"]
#     dataset_location = resolve_dataset_path(dataset_path)
#     if dataset_location is None:
#         return {"error": "Dataset not found"}

#     # Create model with the specified optimization strategy
#     model_name = config.get("model_name", "smol-ai/smolvlm2-500m")
#     strategy = config.get("optimization_strategy", "qlora")

#     if strategy == "qlora":
#         # Configure QLoRA
#         bnb_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_compute_dtype=torch.float16,
#             bnb_4bit_use_double_quant=True,
#         )

#         model = AutoModelForVision2Seq.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
#         model = prepare_model_for_kbit_training(model)

#         lora_config = LoraConfig(
#             r=16,
#             lora_alpha=32,
#             target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#             lora_dropout=0.05,
#             bias="none",
#             task_type="CAUSAL_LM",
#         )

#         model = get_peft_model(model, lora_config)

#     elif strategy == "lora":
#         # Configure LoRA without quantization
#         model = AutoModelForVision2Seq.from_pretrained(model_name)

#         lora_config = LoraConfig(
#             r=16,
#             lora_alpha=32,
#             target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#             lora_dropout=0.05,
#             bias="none",
#             task_type="CAUSAL_LM",
#         )

#         model = get_peft_model(model, lora_config)

#     elif strategy == "freeze_vision":
#         # Freeze vision encoder, train only language model part
#         model = AutoModelForVision2Seq.from_pretrained(model_name)

#         # Freeze vision encoder parameters
#         for param in model.vision_model.parameters():
#             param.requires_grad = False
#     else:
#         raise ValueError(f"Unsupported optimization strategy: {strategy}")

#     # Load processor and datasets
#     processor = AutoProcessor.from_pretrained(model_name)

#     # Create processor wrapper to preprocess data before collating
#     def process_batch(batch):
#         processed_batch = []
#         for item in batch:
#             processed_item = processor(images=item.get("image"), text=item.get("text", ""), return_tensors="pt")
#             processed_batch.append(processed_item)
#         return processed_batch

#     train_loader, valid_loader, test_loader = create_data_loaders(
#         dataset_location=dataset_location,
#         train_batch_size=config.get("batch_size", 4),
#         train_collect_fn=lambda batch: train_collate_fn(process_batch(batch)),
#         train_num_workers=config.get("num_workers", 0),
#         test_batch_size=config.get("val_batch_size", config.get("batch_size", 4)),
#         test_collect_fn=lambda batch: evaluation_collate_fn(process_batch(batch)),
#         test_num_workers=config.get("val_num_workers", config.get("num_workers", 0)),
#     )

#     # Set up training arguments
#     output_dir = config.get("output_dir", "./smolvlm2-finetuned")
#     os.makedirs(output_dir, exist_ok=True)

#     training_args = TrainingArguments(
#         output_dir=output_dir,
#         num_train_epochs=config.get("epochs", 10),
#         per_device_train_batch_size=config.get("batch_size", 4),
#         per_device_eval_batch_size=config.get("val_batch_size", config.get("batch_size", 4)),
#         gradient_accumulation_steps=4,
#         learning_rate=2e-5,
#         weight_decay=0.01,
#         warmup_steps=100,
#         save_strategy="epoch",
#         save_total_limit=2,
#         logging_steps=10,
#         evaluation_strategy="epoch",
#         load_best_model_at_end=True,
#         remove_unused_columns=False,
#     )

#     # Safely handle potential None loaders by directly checking
#     # train_loader/valid_loader before accessing dataset attribute
#     train_dataset = None
#     if train_loader is not None:
#         train_dataset = train_loader.dataset

#     eval_dataset = None
#     if valid_loader is not None:
#         eval_dataset = valid_loader.dataset

#     # Create data_collator that matches the train_collate_fn signature (doesn't pass processor)
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         data_collator=lambda batch: train_collate_fn(process_batch(batch)),
#     )

#     # Train model
#     trainer.train()

#     # Save model and processor
#     model.save_pretrained(output_dir)
#     processor.save_pretrained(output_dir)

#     # Return results
#     return {
#         "model_path": output_dir,
#         "metrics": trainer.state.log_history[-1] if trainer.state.log_history else {"loss": "N/A"},
#         "status": "Training completed",
#     }
