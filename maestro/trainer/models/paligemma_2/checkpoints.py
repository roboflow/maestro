import os
from enum import Enum
from typing import Optional

import torch
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig, PaliGemmaForConditionalGeneration, PaliGemmaProcessor

from maestro.trainer.common.utils.device import parse_device_spec

DEFAULT_PALIGEMMA2_MODEL_ID = "google/paligemma2-3b-pt-224"
DEFAULT_PALIGEMMA2_MODEL_REVISION = "refs/heads/main"


class OptimizationStrategy(Enum):
    """Enumeration for optimization strategies."""

    LORA = "lora"
    QLORA = "qlora"
    FREEZE = "freeze"
    NONE = "none"


def load_model(
    model_id_or_path: str = DEFAULT_PALIGEMMA2_MODEL_ID,
    revision: str = DEFAULT_PALIGEMMA2_MODEL_REVISION,
    device: str | torch.device = "auto",
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.LORA,
    cache_dir: Optional[str] = None,
) -> tuple[PaliGemmaProcessor, PaliGemmaForConditionalGeneration]:
    """Loads a PaliGemma 2 model and its associated processor.

    Args:
        model_id_or_path (str): The identifier or path of the model to load.
        revision (str): The specific model revision to use.
        device (torch.device): The device to load the model onto.
        optimization_strategy (OptimizationStrategy): The optimization strategy to apply to the model.
        cache_dir (Optional[str]): Directory to cache the downloaded model files.

    Returns:
        (PaliGemmaProcessor, PaliGemmaForConditionalGeneration):
            A tuple containing the loaded processor and model.

    Raises:
        ValueError: If the model or processor cannot be loaded.
    """
    device = parse_device_spec(device)
    processor = PaliGemmaProcessor.from_pretrained(model_id_or_path, trust_remote_code=True, revision=revision)

    if optimization_strategy in {OptimizationStrategy.LORA, OptimizationStrategy.QLORA}:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        bnb_config = (
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_type=torch.bfloat16)
            if optimization_strategy == OptimizationStrategy.QLORA
            else None
        )

        model = PaliGemmaForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_id_or_path,
            revision=revision,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_id_or_path,
            revision=revision,
            trust_remote_code=True,
            device_map="auto",
            cache_dir=cache_dir
        ).to(device)

        if optimization_strategy == OptimizationStrategy.FREEZE:
            for param in model.vision_tower.parameters():
                param.requires_grad = False

            for param in model.multi_modal_projector.parameters():
                param.requires_grad = False

    return processor, model


def save_model(
    target_dir: str,
    processor: PaliGemmaProcessor,
    model: PaliGemmaForConditionalGeneration,
) -> None:
    """
    Save a PaliGemma 2 model and its processor to disk.

    Args:
        target_dir: Directory path where the model and processor will be saved.
            Will be created if it doesn't exist.
        processor: The PaliGemma 2 processor to save.
        model: The PaliGemma 2model to save.
    """
    os.makedirs(target_dir, exist_ok=True)
    processor.save_pretrained(target_dir)
    model.save_pretrained(target_dir)
