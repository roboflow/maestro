import os
from collections.abc import Iterator
from typing import Literal, Optional, Union

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from torch import optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from maestro.trainer.common.configuration.env import (
    CUDA_DEVICE_ENV,
    DEFAULT_CUDA_DEVICE,
    HF_TOKEN_ENV,
)
from maestro.trainer.common.data_loaders.jsonl import JSONLDataset
from maestro.trainer.common.utils.metrics import MetricsTracker
from maestro.trainer.common.utils.reproducibility import make_it_reproducible

DEFAULT_PALIGEMMA_MODEL_ID = "google/paligemma-3b-pt-224"
DEVICE = torch.device("cpu") if not torch.cuda.is_available() else os.getenv(CUDA_DEVICE_ENV, DEFAULT_CUDA_DEVICE)

LoraInitLiteral = Literal["gaussian", "olora", "pissa", "pissa_niter_[number of iters]", "loftq"]


def train(
    model: PaliGemmaForConditionalGeneration,
    processor: AutoProcessor,
    train_dataset: JSONLDataset,
    dataset_root: str,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    device: torch.device = DEVICE,
) -> MetricsTracker:
    make_it_reproducible()
    if device.type == "cup":
        raise RuntimeError("PaliGemma training process requires GPU")
    metrics_tracker = MetricsTracker.init(metrics=["loss"])
    peft_model = prepare_peft_model(model=model).train()
    train_steps = len(train_dataset) // batch_size
    with torch.amp.autocast(device.type, torch.float16):
        lora_layers = filter(lambda p: p.requires_grad, peft_model.parameters())
        optimizer = optim.SGD(lora_layers, lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            epochs * train_steps + 1,
            eta_min=learning_rate / 10,
        )
        run_training_epochs(
            peft_model=peft_model,
            processor=processor,
            epochs=epochs,
            train_steps=train_steps,
            train_dataset=train_dataset,
            batch_size=batch_size,
            dataset_root=dataset_root,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics_tracker=metrics_tracker,
        )
    return metrics_tracker


def run_training_epochs(
    peft_model: PeftModel,
    processor: AutoProcessor,
    epochs: int,
    train_steps: int,
    train_dataset: JSONLDataset,
    batch_size: int,
    dataset_root: str,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    metrics_tracker: MetricsTracker,
) -> None:
    for epoch in tqdm(range(epochs), desc="EPOCHS"):
        train_dataset.shuffle()
        dataset_iterator = iter(train_dataset)
        progress_bar = tqdm(range(train_steps), desc="STEPS")
        run_training_steps(
            peft_model=peft_model,
            processor=processor,
            epoch=epoch,
            train_steps=train_steps,
            batch_size=batch_size,
            dataset_iterator=dataset_iterator,
            dataset_root=dataset_root,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics_tracker=metrics_tracker,
            progress_bar=progress_bar,
        )


def run_training_steps(
    peft_model: PeftModel,
    processor: AutoProcessor,
    epoch: int,
    train_steps: int,
    batch_size: int,
    dataset_iterator: Iterator[dict],
    dataset_root: str,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    metrics_tracker: MetricsTracker,
    progress_bar,
) -> None:
    for step in range(1, train_steps + 1):
        batch = collect_batch(
            batch_size=batch_size,
            dataset_iterator=dataset_iterator,
            dataset_root=dataset_root,
            processor=processor,
        )
        loss_tensor = peft_model(**batch)["loss"]
        loss_tensor.backward()
        loss = loss_tensor.cpu().detach().numpy()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        progress_bar.update(1)
        progress_bar.set_description(f"Loss: {loss}")
        metrics_tracker.register(metric="loss", epoch=epoch, step=step, value=loss)


def collect_batch(
    batch_size: int,
    dataset_iterator: Iterator[dict],
    dataset_root: str,
    processor: AutoProcessor,
) -> torch.Tensor:
    with torch.no_grad():
        examples = []
        for _ in range(batch_size):
            examples.append(next(dataset_iterator))
        return _collate_fn(
            examples=examples,
            dataset_root=dataset_root,
            processor=processor,
        )


def load_model(
    model_id: str = DEFAULT_PALIGEMMA_MODEL_ID,
    revision: str = "float16",
    device: torch.device = DEVICE,
    hf_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> tuple[AutoProcessor, PaliGemmaForConditionalGeneration]:
    if hf_token is None:
        hf_token = os.getenv(HF_TOKEN_ENV)
    processor = AutoProcessor.from_pretrained(model_id, token=hf_token, cache_dir=cache_dir)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        revision=revision,
        device_map=device,
        cache_dir=cache_dir,
        token=hf_token,
        torch_dtype=torch.float16,
    ).eval()
    return processor, model


def prepare_peft_model(
    model: PaliGemmaForConditionalGeneration,
    r: int = 12,
    lora_alpha: int = 12,
    lora_dropout: float = 0.05,
    bias: Literal["none", "all", "lora_only"] = "none",
    inference_mode: bool = False,
    use_rslora: bool = True,
    init_lora_weights: Union[bool, LoraInitLiteral] = "gaussian",
    revision: str = "float16",
    device: torch.device = DEVICE,
) -> PeftModel:
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "linear"],
        task_type="CAUSAL_LM",
        lora_dropout=lora_dropout,
        bias=bias,
        inference_mode=inference_mode,
        use_rslora=use_rslora,
        init_lora_weights=init_lora_weights,
        revision=revision,
    )
    return get_peft_model(model, config).to(device)


def _collate_fn(
    examples: list[dict],
    dataset_root: str,
    processor: AutoProcessor,
    device: torch.device = DEVICE,
    image_file_key: str = "image",
    prefix_key: str = "prefix",
    suffix_key: str = "suffix",
) -> torch.Tensor:
    images = [
        _load_image_from_dataset(
            image_name=example[image_file_key],
            dataset_root=dataset_root,
        )
        for example in examples
    ]
    tokens = processor(
        text=[example[prefix_key] for example in examples],
        suffix=[example[suffix_key] for example in examples],
        images=images,
        return_tensors="pt",
        padding="longest",
    )
    return tokens.to(device)


def _load_image_from_dataset(image_name: str, dataset_root: str) -> Image.Image:
    image_path = os.path.join(dataset_root, "dataset", image_name)
    return Image.open(image_path).convert("RGB")
