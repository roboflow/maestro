import os
from typing import Optional
from enum import Enum

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from maestro.trainer.common.utils.device import parse_device_spec
from maestro.trainer.logger import get_maestro_logger
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

DEFAULT_SMOLVLM2_MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"#"smol-ai/smolvlm2-500m"
DEFAULT_SMOLVLM2_MODEL_REVISION = "refs/heads/main"
DEFAULT_SMOLVLM2_PEFT_PARAMS = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "bias": "none",
    "target_modules": ["q_proj", "o_proj", "k_proj", "v_proj", "linear", "Conv2d", "lm_head", "fc2"],
    "task_type": "CAUSAL_LM",
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

# def load_checkpoint(path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> dict:
#     """
#     Load model checkpoint.

#     Args:
#         path: Path to checkpoint
#         device: Device to load model on

#     Returns:
#         Dictionary containing model, processor, and metadata
#     """
#     # Load model
#     model = AutoModelForImageTextToText.from_pretrained(path)
#     model.to(device)

#     # Load processor
#     processor = AutoProcessor.from_pretrained(path)

#     # Load metadata if exists
#     metadata_path = os.path.join(path, "metadata.pt")
#     metadata = torch.load(metadata_path) if os.path.exists(metadata_path) else None

#     return {"model": model, "processor": processor, "metadata": metadata}

class OptimizationStrategy(Enum):
    """Enumeration for optimization strategies."""

    LORA = "lora"
    QLORA = "qlora"
    FREEZE = "freeze"
    NONE = "none"
def load_model(
    model_id_or_path: str = DEFAULT_SMOLVLM2_MODEL_ID,
    revision: str = DEFAULT_SMOLVLM2_MODEL_REVISION,
    device: str | torch.device = "auto",
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.NONE,
    peft_advanced_params: Optional[dict] = None,
    cache_dir: Optional[str] = None,
) -> tuple[AutoProcessor, AutoModelForImageTextToText]:
    """Loads a PaliGemma 2 model and its associated processor.

    Args:
        model_id_or_path (str): The identifier or path of the model to load.
        revision (str): The specific model revision to use.
        device (torch.device): The device to load the model onto.
        optimization_strategy (OptimizationStrategy): The optimization strategy to apply to the model.
        peft_advanced_params: custom lora configuration
        cache_dir (Optional[str]): Directory to cache the downloaded model files.

    Returns:
        (PaliGemmaProcessor, PaliGemmaForConditionalGeneration):
            A tuple containing the loaded processor and model.

    Raises:
        ValueError: If the model or processor cannot be loaded.
    """
    device = parse_device_spec(device)
    #processor = AutoProcessor.from_pretrained(model_id_or_path, trust_remote_code=True, revision=revision)
    processor = PaliGemmaProcessor.from_pretrained(model_id_or_path)

    if optimization_strategy in {OptimizationStrategy.LORA, OptimizationStrategy.QLORA}:
        default_params = DEFAULT_SMOLVLM2_PEFT_PARAMS
        if peft_advanced_params is not None:
            default_params.update(peft_advanced_params)
            try:
                lora_config = LoraConfig(**default_params)
                logger.info("Successfully created LoraConfig")
            except TypeError:
                logger.exception("Invalid parameters for LoraConfig")
                raise
        
        else:
            logger.info("No LoRA parameters provided. Using default configuration.")
            lora_config = LoraConfig(**default_params)
        
        bnb_config = (BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            #bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        ) if optimization_strategy == OptimizationStrategy.QLORA
            else None)
        
        model = AutoModelForImageTextToText.from_pretrained(
            model_id_or_path,
            revision=revision,
            trust_remote_code=True,
            quantization_config=bnb_config,
            cache_dir=cache_dir,
            #torch_dtype=torch.bfloat16, 
        ).to(device)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:

        model = AutoModelForImageTextToText.from_pretrained(
            model_id_or_path,
            revision=revision,
            trust_remote_code=True,
            cache_dir=cache_dir,).to(device)

        if optimization_strategy == OptimizationStrategy.FREEZE:
            # Freeze vision encoder parameters
            for param in model.model.vision_model.parameters():
                param.requires_grad = False

            # for param in model.multi_modal_projector.parameters():
            #     param.requires_grad = False



    return processor, model

