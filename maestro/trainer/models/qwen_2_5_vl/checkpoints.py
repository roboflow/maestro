import os
from enum import Enum
from typing import Optional

import torch
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

from maestro.trainer.common.utils.device import parse_device_spec

DEFAULT_QWEN2_5_VL_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_QWEN2_5_VL_MODEL_REVISION = "refs/heads/main"


class OptimizationStrategy(Enum):
    """Enumeration for optimization strategies."""

    LORA = "lora"
    QLORA = "qlora"
    NONE = "none"


def load_model(
    model_id_or_path: str = DEFAULT_QWEN2_5_VL_MODEL_ID,
    revision: str = DEFAULT_QWEN2_5_VL_MODEL_REVISION,
    device: str | torch.device = "auto",
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.LORA,
    cache_dir: Optional[str] = None,
    min_pixels: int = 256 * 28 * 28,
    max_pixels: int = 1280 * 28 * 28,
) -> tuple[Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration]:
    """
    Loads a Qwen2.5-VL model and its associated processor with optional LoRA or QLoRA.

    Args:
        model_id_or_path (str): The model name or path.
        revision (str): The model revision to load.
        device (str | torch.device): The device to load the model onto.
        optimization_strategy (OptimizationStrategy): LORA, QLORA, or NONE.
        cache_dir (Optional[str]): Directory to cache downloaded model files.
        min_pixels (int): Minimum number of pixels allowed in the resized image.
        max_pixels (int): Maximum number of pixels allowed in the resized image.

    Returns:
        (Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration):
            A tuple containing the loaded processor and model.
    """
    device = parse_device_spec(device)
    processor = Qwen2_5_VLProcessor.from_pretrained(
        model_id_or_path,
        revision=revision,
        trust_remote_code=True,
        cache_dir=cache_dir,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    if optimization_strategy in {OptimizationStrategy.LORA, OptimizationStrategy.QLORA}:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )

        bnb_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_type=torch.bfloat16,
            )
            if optimization_strategy == OptimizationStrategy.QLORA
            else None
        )

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id_or_path,
            revision=revision,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id_or_path,
            revision=revision,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
        )
        model.to(device)

    return processor, model


def save_model(
    target_dir: str,
    processor: Qwen2_5_VLProcessor,
    model: Qwen2_5_VLForConditionalGeneration,
) -> None:
    """
    Save a Qwen2.5-VL model and its processor to disk.

    Args:
        target_dir: Directory path where the model and processor will be saved.
            Will be created if it doesn't exist.
        processor: The Qwen2.5-VL processor to save.
        model: The Qwen2.5-VL model to save.
    """
    os.makedirs(target_dir, exist_ok=True)
    processor.save_pretrained(target_dir)
    model.save_pretrained(target_dir)
