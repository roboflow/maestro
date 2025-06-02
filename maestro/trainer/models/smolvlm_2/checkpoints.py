import os
from typing import Optional
from enum import Enum

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from maestro.trainer.common.utils.device import parse_device_spec
from maestro.trainer.logger import get_maestro_logger
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig

DEFAULT_SMOLVLM_2_MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"#"HuggingFaceTB/SmolVLM2-2.2B-Instruct"
DEFAULT_SMOLVLM_2_MODEL_REVISION = "refs/heads/main"
DEFAULT_SMOLVLM_2_LORA_PARAMS = {
    "r": 8,
    "lora_alpha": 8,
    "lora_dropout": 0.1,
    "bias": "none",
    "target_modules": ['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
    "init_lora_weights": "gaussian",
    "use_dora": True
}
DEFAULT_SMOLVLM_2_QLORA_PARAMS = {
    "r": 8,
    "lora_alpha": 8,
    "lora_dropout": 0.1,
    "bias": "none",
    "target_modules": ['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
    "init_lora_weights": "gaussian",
    "use_dora": False
}
logger = get_maestro_logger()


def save_checkpoint(
    model: AutoModelForImageTextToText, processor: AutoProcessor, path: str, metadata: Optional[dict] = None
) -> None:
    """
    Save model checkpoint.

    Args:
        model: Model to save
        processor: Processor to save
        path: Path to save checkpoint
        metadata: Optional metadata to save
    """
    os.makedirs(path, exist_ok=True)

    # Save model
    model.save_pretrained(path)

    # Save processor
    processor.save_pretrained(path)

    # Save metadata if provided
    if metadata is not None:
        torch.save(metadata, os.path.join(path, "metadata.pt"))

def save_model(
    target_dir: str,
    processor: AutoProcessor,
    model: AutoModelForImageTextToText,
) -> None:
    """
    Save a SmolVLM 2 model and its processor to disk.

    Args:
        target_dir: Directory path where the model and processor will be saved.
            Will be created if it doesn't exist.
        processor: The SmolVLM 2 processor to save.
        model: The SmolVLM 2model to save.
    """
    os.makedirs(target_dir, exist_ok=True)
    processor.save_pretrained(target_dir)
    model.save_pretrained(target_dir)

class OptimizationStrategy(Enum):
    """Enumeration for optimization strategies."""

    LORA = "lora"
    QLORA = "qlora"
    FREEZE = "freeze"
    NONE = "none"
def load_model(
    model_id_or_path: str = DEFAULT_SMOLVLM_2_MODEL_ID,
    revision: str = DEFAULT_SMOLVLM_2_MODEL_REVISION,
    device: str | torch.device = "auto",
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.NONE,
    peft_advanced_params: Optional[dict] = None,
    cache_dir: Optional[str] = None,
    longest_edge: int = 512
) -> tuple[AutoProcessor, AutoModelForImageTextToText]:
    device = parse_device_spec(device)
    processor = AutoProcessor.from_pretrained(
        model_id_or_path,
        do_resize=True, size={"longest_edge": longest_edge},
        trust_remote_code=True,
        revision=revision
    )

    if optimization_strategy in {OptimizationStrategy.LORA, OptimizationStrategy.QLORA}:
        default_params = DEFAULT_SMOLVLM_2_QLORA_PARAMS if optimization_strategy == OptimizationStrategy.QLORA else DEFAULT_SMOLVLM_2_LORA_PARAMS
        if peft_advanced_params is not None:
            default_params.update(peft_advanced_params)
            try:
                lora_config = LoraConfig(**default_params)
                logger.info("Successfully created LoraConfig")
            except TypeError:
                logger.exception("Invalid parameters for LoraConfig")
                raise
        else:
            logger.info("No additiopnal LoRA parameters provided. Using default configuration.")
            lora_config = LoraConfig(**default_params)

        bnb_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            if optimization_strategy == OptimizationStrategy.QLORA
            else None
        )

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name_or_path=model_id_or_path,
            revision=revision,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            _attn_implementation="flash_attention_2",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name_or_path=model_id_or_path,
            revision=revision,
            trust_remote_code=True,
            device_map="auto",
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2"
        ).to(device)

        if optimization_strategy == OptimizationStrategy.FREEZE:
            for param in model.model.vision_model.parameters():
                param.requires_grad = False

    return processor, model
