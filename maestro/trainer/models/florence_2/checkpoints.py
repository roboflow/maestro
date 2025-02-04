import os
from enum import Enum
from typing import Optional

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoProcessor

from maestro.trainer.common.utils.device import parse_device_spec

DEFAULT_FLORENCE2_MODEL_ID = "microsoft/Florence-2-base-ft"
DEFAULT_FLORENCE2_MODEL_REVISION = "refs/pr/20"


class OptimizationStrategy(Enum):
    """Enumeration for optimization strategies."""

    LORA = "lora"
    FREEZE = "freeze"
    NONE = "none"


def load_model(
    model_id_or_path: str = DEFAULT_FLORENCE2_MODEL_ID,
    revision: str = DEFAULT_FLORENCE2_MODEL_REVISION,
    device: str | torch.device = "auto",
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.LORA,
    cache_dir: Optional[str] = None,
) -> tuple[AutoProcessor, AutoModelForCausalLM]:
    """Loads a Florence 2 model and its associated processor.

    Args:
        model_id_or_path (str): The identifier or path of the Florence 2 model to load.
        revision (str): The specific model revision to use.
        device (torch.device): The device to load the model onto.
        optimization_strategy (OptimizationStrategy): The optimization strategy to apply to the model.
        cache_dir (Optional[str]): Directory to cache the downloaded model files.

    Returns:
        tuple(AutoProcessor, AutoModelForCausalLM):
            A tuple containing the loaded processor and model.

    Raises:
        ValueError: If the model or processor cannot be loaded.
    """
    device = parse_device_spec(device)
    processor = AutoProcessor.from_pretrained(model_id_or_path, trust_remote_code=True, revision=revision)

    if optimization_strategy == OptimizationStrategy.LORA:
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "linear", "Conv2d", "lm_head", "fc2"],
            task_type="CAUSAL_LM",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            revision=revision,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        model = get_peft_model(model, config).to(device)
        model.print_trainable_parameters()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            revision=revision,
            trust_remote_code=True,
            cache_dir=cache_dir,
        ).to(device)

        if optimization_strategy == OptimizationStrategy.FREEZE:
            for param in model.vision_tower.parameters():
                param.is_trainable = False

    return processor, model


def save_model(
    target_dir: str,
    processor: AutoProcessor,
    model: AutoModelForCausalLM,
) -> None:
    os.makedirs(target_dir, exist_ok=True)
    processor.save_pretrained(target_dir)
    model.save_pretrained(target_dir)
