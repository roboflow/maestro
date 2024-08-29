import logging
import os
import shutil
from dataclasses import dataclass, replace
from functools import partial
from glob import glob
from typing import Optional, Tuple, List, Literal, Union

from PIL import Image
import torch
from peft import PeftModel, LoraConfig, get_peft_model
from torch.optim import Optimizer, AdamW, Adam, SGD
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, get_scheduler

from maestro.trainer.common.configuration.env import CUDA_DEVICE_ENV, DEFAULT_CUDA_DEVICE
from maestro.trainer.common.data_loaders.datasets import DetectionDataset
from maestro.trainer.common.utils.leaderboard import CheckpointsLeaderboard
from maestro.trainer.common.utils.reproducibility import make_it_reproducible
from maestro.trainer.models.paligemma.training import LoraInitLiteral

DEFAULT_FLORENCE2_MODEL_ID = "microsoft/Florence-2-base-ft"
DEFAULT_FLORENCE2_MODEL_REVISION = "refs/pr/20"
DEVICE = torch.device("cpu") if not torch.cuda.is_available() else os.getenv(CUDA_DEVICE_ENV, DEFAULT_CUDA_DEVICE)


@dataclass(frozen=True)
class TrainingConfiguration:
    dataset_location: str
    model_id_or_path: str = DEFAULT_FLORENCE2_MODEL_ID
    revision: str = DEFAULT_FLORENCE2_MODEL_REVISION
    device: torch.device = DEVICE
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


def train(configuration: TrainingConfiguration) -> None:
    make_it_reproducible()
    training_run_dir = _establish_training_run_dir(
        training_dir=configuration.training_dir,
    )
    configuration = replace(
        configuration,
        training_dir=training_run_dir,
    )
    checkpoints_leaderboard = CheckpointsLeaderboard(
        max_checkpoints=configuration.max_checkpoints_to_keep,
    )
    processor, model = load_model(
        model_id_or_path=configuration.model_id_or_path,
        revision=configuration.revision,
        device=configuration.device,
        cache_dir=configuration.transformers_cache_dir,
    )
    train_loader, val_loader, test_loader = prepare_data_loaders(
        dataset_location=configuration.dataset_location,
        train_batch_size=configuration.train_batch_size,
        processor=processor,
        device=configuration.device,
        num_workers=configuration.loaders_workers,
        test_loaders_workers=configuration.test_loaders_workers,
    )
    if test_loader is None:
        test_loader = val_loader
    peft_model = prepare_peft_model(
        model=model,
        r=configuration.lora_r,
        lora_alpha=configuration.lora_alpha,
        lora_dropout=configuration.lora_dropout,
        bias=configuration.bias,
        use_rslora=configuration.use_rslora,
        init_lora_weights=configuration.init_lora_weights,
        revision=configuration.revision,
    )
    run_training_loop(
        processor=processor,
        model=peft_model,
        data_loaders=(train_loader, val_loader),
        configuration=configuration,
        checkpoints_leaderboard=checkpoints_leaderboard,
    )
    best_model_path = checkpoints_leaderboard.get_best_model()
    print(f"Loading best model from {best_model_path}")
    processor, model = load_model(
        model_id_or_path=best_model_path,
    )
    if test_loader is not None:
        run_validation_epoch(
            processor=processor,
            model=model,
            loader=test_loader,
            epoch_number=None,
            configuration=configuration,
            title="Test",
        )
    best_model_dir = os.path.join(configuration.training_dir, "best_model")
    print(f"Saving best model: {best_model_dir}")
    model.save_pretrained(best_model_dir)
    processor.save_pretrained(best_model_dir)


def load_model(
    model_id_or_path: str = DEFAULT_FLORENCE2_MODEL_ID,
    revision: str = DEFAULT_FLORENCE2_MODEL_REVISION,
    device: torch.device = DEVICE,
    cache_dir: Optional[str] = None,
) -> Tuple[AutoProcessor, AutoModelForCausalLM]:
    processor = AutoProcessor.from_pretrained(
        model_id_or_path,
        trust_remote_code=True,
        revision=revision,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        trust_remote_code=True,
        revision=revision,
        cache_dir=cache_dir,
    ).to(device)
    return processor, model


def prepare_data_loaders(
    dataset_location: str,
    train_batch_size: int,
    processor: AutoProcessor,
    device: torch.device,
    num_workers: int = 0,
    test_batch_size: Optional[int] = None,
    test_loaders_workers: Optional[int] = None,
) -> Tuple[
    DataLoader,
    Optional[DataLoader],
    Optional[DataLoader],
]:
    test_batch_size = test_batch_size or train_batch_size
    test_loaders_workers = test_loaders_workers or num_workers
    train_data_loader = prepare_detection_data_loader(
        dataset_location=dataset_location,
        split_name="train",
        batch_size=train_batch_size,
        processor=processor,
        device=device,
        num_workers=num_workers,
        shuffle=True,
    )
    if train_data_loader is None:
        raise RuntimeError("Could not initialise train data loader")
    valid_data_loader = prepare_detection_data_loader(
        dataset_location=dataset_location,
        split_name="valid",
        batch_size=test_batch_size,
        processor=processor,
        device=device,
        num_workers=test_loaders_workers,
        shuffle=False,
    )
    test_data_loader = prepare_detection_data_loader(
        dataset_location=dataset_location,
        split_name="test",
        batch_size=test_batch_size,
        processor=processor,
        device=device,
        num_workers=test_loaders_workers,
        shuffle=False,
    )
    return train_data_loader, valid_data_loader, test_data_loader


def prepare_detection_data_loader(
    dataset_location: str,
    split_name: str,
    batch_size: int,
    processor: AutoProcessor,
    device: torch.device,
    num_workers: int = 0,
    shuffle: bool = True,
) -> Optional[DataLoader]:
    image_directory_path = os.path.join(dataset_location, split_name)
    jsonl_file_path = os.path.join(dataset_location, split_name, "annotations.jsonl")
    if os.path.exists(image_directory_path):
        logging.warning(f"Could not data directory: {image_directory_path}")
        return None
    if not os.path.exists(jsonl_file_path):
        logging.warning(f"Could not find JSONL file: {jsonl_file_path}")
        return None
    dataset = DetectionDataset(
        jsonl_file_path=jsonl_file_path,
        image_directory_path=image_directory_path,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=partial(_collate_fn, processor=processor, device=device),
        num_workers=num_workers,
        shuffle=shuffle,
    )


def prepare_peft_model(
    model: AutoModelForCausalLM,
    r: int = 8,
    lora_alpha: int = 8,
    lora_dropout: float = 0.05,
    bias: Literal["none", "all", "lora_only"] = "none",
    inference_mode: bool = False,
    use_rslora: bool = True,
    init_lora_weights: Union[bool, LoraInitLiteral] = "gaussian",
    revision: str = DEFAULT_FLORENCE2_MODEL_REVISION,
) -> PeftModel:
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "linear", "Conv2d", "lm_head", "fc2"],
        task_type="CAUSAL_LM",
        lora_dropout=lora_dropout,
        bias=bias,
        inference_mode=inference_mode,
        use_rslora=use_rslora,
        init_lora_weights=init_lora_weights,
        revision=revision,
    )
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    return peft_model.to(model.device)


def run_training_loop(
    processor: AutoProcessor,
    model: PeftModel,
    data_loaders: Tuple[DataLoader, Optional[DataLoader]],
    configuration: TrainingConfiguration,
    checkpoints_leaderboard: CheckpointsLeaderboard,
) -> None:
    train_loader, val_loader = data_loaders
    optimizer = _get_optimizer(model=model, configuration=configuration)
    total_num_training_steps = configuration.training_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name=configuration.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_num_training_steps,
    )
    for epoch in range(configuration.training_epochs):
        run_training_epoch(
            processor=processor,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epoch_number=epoch,
            configuration=configuration,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            checkpoints_leaderboard=checkpoints_leaderboard,
        )


def run_training_epoch(
    processor: AutoProcessor,
    model: PeftModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epoch_number: int,
    configuration: TrainingConfiguration,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    checkpoints_leaderboard: CheckpointsLeaderboard,
) -> None:
    model.train()
    training_losses: List[float] = []
    last_100_losses = training_losses[-100:]
    loss_moving_average = sum(last_100_losses) / len(last_100_losses) if len(last_100_losses) > 0 else 0.0
    for inputs, answers in tqdm(
        train_loader,
        desc=f"Epoch {epoch_number + 1}/{configuration.training_epochs}. " f"Loss: {round(loss_moving_average, 4)}",
    ):
        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]
        labels = processor.tokenizer(
            text=answers, return_tensors="pt", padding=True, return_token_type_ids=False
        ).input_ids.to(configuration.device)
        outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        training_losses.append(loss.item())
    if len(training_losses) > 0:
        avg_train_loss = sum(training_losses) / len(training_losses)
        print(f"Average Training Loss: {avg_train_loss}")
    if val_loader is None or len(val_loader) == 0:
        return None
    validation_loss = run_validation_epoch(
        processor=processor,
        model=model,
        loader=val_loader,
        epoch_number=epoch_number,
        configuration=configuration,
    )
    checkpoint_dir = os.path.join(configuration.training_dir, "checkpoints", str(epoch_number))
    should_save, to_remove = checkpoints_leaderboard.register_checkpoint(
        epoch=epoch_number,
        path=checkpoint_dir,
        loss=validation_loss,
    )
    if should_save:
        print(f"Saving checkpoint under {checkpoint_dir}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        processor.save_pretrained(checkpoint_dir)
    if to_remove is not None:
        print(f"Removing checkpoint {to_remove}")
        shutil.rmtree(to_remove, ignore_errors=True)


def run_validation_epoch(
    processor: AutoProcessor,
    model: Union[PeftModel, AutoModelForCausalLM],
    loader: DataLoader,
    epoch_number: Optional[int],
    configuration: TrainingConfiguration,
    title: str = "Validation",
) -> float:
    val_loss = 0.0
    epoch_marker = ""
    if epoch_number is not None:
        epoch_marker = f"Epoch {epoch_number + 1}/{configuration.training_epochs}"
    with torch.no_grad():
        for inputs, answers in tqdm(loader, desc=f"{title} | {epoch_marker}"):
            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            labels = processor.tokenizer(
                text=answers, return_tensors="pt", padding=True, return_token_type_ids=False
            ).input_ids.to(configuration.device)
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()
        avg_val_loss = val_loss / len(loader)
        print(f"Average {title} Loss: {avg_val_loss}")
        return avg_val_loss


def save_model(
    target_dir: str,
    processor: AutoProcessor,
    model: AutoModelForCausalLM,
) -> None:
    os.makedirs(target_dir, exist_ok=True)
    processor.save_pretrained(target_dir)
    model.save_pretrained(target_dir)


def _establish_training_run_dir(training_dir: str) -> str:
    training_dir = os.path.abspath(training_dir)
    existing_directory_entries = glob(os.path.join(training_dir, "*"))
    subdirectories = [path for path in existing_directory_entries if os.path.isdir(path)]
    run_id = len(subdirectories) + 1
    training_run_dir = os.path.join(training_dir, str(run_id))
    os.makedirs(training_run_dir, exist_ok=True)
    return training_run_dir


def _get_optimizer(model: PeftModel, configuration: TrainingConfiguration) -> Optimizer:
    if configuration.optimiser == "adamw":
        return AdamW(model.parameters(), lr=configuration.learning_rate)
    if configuration.optimiser == "adam":
        return Adam(model.parameters(), lr=configuration.learning_rate)
    return SGD(model.parameters(), lr=configuration.learning_rate)


def _collate_fn(
    batch: Tuple[List[str], List[str], List[Image.Image]],
    processor: AutoProcessor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    questions, answers, images = zip(*batch)
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(device)
    return inputs, answers
