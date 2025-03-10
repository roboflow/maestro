import json
import os
from enum import Enum
from typing import Any, Optional

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

from maestro.trainer.common.utils.device import parse_device_spec
from maestro.trainer.logger import get_maestro_logger

DEFAULT_PHI_4_MODEL_ID = "microsoft/Phi-4-multimodal-instruct"
DEFAULT_PHI_4_MODEL_REVISION = "refs/heads/main"
logger = get_maestro_logger()


class OptimizationStrategy(Enum):
    """Enumeration for optimization strategies."""

    LORA = "lora"
    QLORA = "qlora"
    NONE = "none"


def load_model(
    model_id_or_path: str = DEFAULT_PHI_4_MODEL_ID,
    revision: str = DEFAULT_PHI_4_MODEL_REVISION,
    device: str | torch.device = "auto",
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.NONE,
    cache_dir: Optional[str] = None,
    use_flash_attention: bool = True,
) -> tuple[AutoProcessor, AutoModelForCausalLM]:
    """
    Loads a Phi-4 multimodal model and its associated processor with optional LoRA or QLoRA.
    Configures the model for vision-only processing by removing audio-related layers.

    Args:
        model_id_or_path (str): The model name or path.
        revision (str): The model revision to load.
        device (str | torch.device): The device to load the model onto.
        optimization_strategy (OptimizationStrategy): LORA, QLORA, or NONE.
        cache_dir (Optional[str]): Directory to cache downloaded model files.
        use_flash_attention (bool): Whether to use Flash Attention 2.

    Returns:
        (AutoProcessor, AutoModelForCausalLM):
            A tuple containing the loaded processor and model configured for vision-only tasks.
    """
    device = parse_device_spec(device)
    processor = AutoProcessor.from_pretrained(
        model_id_or_path,
        revision=revision,
        trust_remote_code=True,
        cache_dir=cache_dir,
        use_fast=True,
    )

    processor.tokenizer.padding_side = "right"
    attn_implementation = "flash_attention_2" if use_flash_attention else "eager"

    if optimization_strategy in {OptimizationStrategy.LORA, OptimizationStrategy.QLORA}:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "v_proj"],  # Todo: Check what target modules will be better
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

        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            revision=revision,
            trust_remote_code=True,
            device_map="auto",  # Todo: Check this for multi GPU it might be loading the model in multi GPU and can cause issues
            quantization_config=bnb_config,
            torch_dtype="auto",
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            revision=revision,
            trust_remote_code=True,
            device_map="auto",  # Todo: Check this for multi GPU it might be loading the model in multi GPU and can cause issues
            torch_dtype="auto",
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
        )
        model.to(device)
    # os.makedirs("save_test", exist_ok=True)
    # save_model("save_test", processor, model)
    return processor, model


def save_model(
    target_dir: str,
    processor: AutoProcessor,
    model: AutoModelForCausalLM,
) -> None:
    """
    Save a Phi-4 model and its processor to disk with options for audio layer handling.

    Args:
        target_dir: Directory path where the model and processor will be saved.
            Will be created if it doesn't exist.
        processor: The Phi-4 processor to save.
        model: The Phi-4 model to save.
    """
    os.makedirs(target_dir, exist_ok=True)

    processor.save_pretrained(target_dir)
    model.save_pretrained(target_dir)

    chat_template_path = os.path.join(target_dir, "chat_template.json")
    if os.path.exists(chat_template_path):
        os.remove(chat_template_path)
        logger.info(f"Removed {chat_template_path}")

    preprocessor_config_path = os.path.join(target_dir, "preprocessor_config.json")
    if os.path.exists(preprocessor_config_path):
        try:
            with open(preprocessor_config_path) as f:
                preprocessor_config = json.load(f)

            for param in ["feature_size", "sampling_rate", "padding_value"]:
                if param in preprocessor_config:
                    del preprocessor_config[param]
                    logger.info(f"Removed '{param}' from preprocessor_config.json")

            audio_params = {"audio_compression_rate": 8, "audio_downsample_rate": 1, "audio_feat_stride": 1}

            for param, value in audio_params.items():
                preprocessor_config[param] = value
                logger.info(f"Added '{param}': {value} to preprocessor_config.json")

            with open(preprocessor_config_path, "w") as f:
                json.dump(preprocessor_config, f, indent=2)
        except Exception as e:
            logger.warning(f"Error modifying preprocessor_config.json: {e}")


def _remove_audio_layers(model):
    """
    Remove audio-related parameters from the model to optimize for vision-only tasks.

    This function removes the audio embedding layers and audio-specific LoRA components
    to reduce memory usage and focus the model on vision processing.

    Args:
        model: The Phi-4 model from which to remove audio-related layers.

    Returns:
        The modified model with audio layers removed.
    """
    try:
        logger.info("Removing audio layers to optimize for vision-only processing...")

        if hasattr(model, "model") and hasattr(model.model, "embed_tokens_extend"):
            if hasattr(model.model.embed_tokens_extend, "audio_embed"):
                del model.model.embed_tokens_extend.audio_embed

        if hasattr(model, "model") and hasattr(model.model, "layers"):
            for layer_idx, layer in enumerate(model.model.layers):
                removed_components = 0
                lora_components = [
                    (layer.mlp.down_proj, "lora_A", "speech"),
                    (layer.mlp.down_proj, "lora_B", "speech"),
                    (layer.mlp.gate_up_proj, "lora_A", "speech"),
                    (layer.mlp.gate_up_proj, "lora_B", "speech"),
                    (layer.self_attn.o_proj, "lora_A", "speech"),
                    (layer.self_attn.o_proj, "lora_B", "speech"),
                    (layer.self_attn.qkv_proj, "lora_A", "speech"),
                    (layer.self_attn.qkv_proj, "lora_B", "speech"),
                ]

                for component, lora_type, key in lora_components:
                    try:
                        if hasattr(component, lora_type) and hasattr(getattr(component, lora_type), key):
                            delattr(getattr(component, lora_type), key)
                            removed_components += 1
                    except AttributeError:
                        continue

                if removed_components > 0:
                    logger.debug(f"Removed {removed_components} audio LoRA components from layer {layer_idx}")

        logger.info("Audio layer removal complete")
    except Exception as e:
        logger.warning(
            f"Could not remove some audio layers. This is expected if using a different model variant. Error: {e}"
        )

    return model


def filter_audio_components(inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Filter out audio-related components from the input dictionary.

    Args:
        inputs: Dictionary containing model inputs, potentially including audio components.

    Returns:
        A new dictionary with audio-related keys removed.
    """
    audio_related_keys = ["input_audio_embeds", "audio_embed_sizes", "audio_attention_mask"]

    filtered_inputs = {k: v for k, v in inputs.items() if k not in audio_related_keys}

    return filtered_inputs
