import os
from enum import Enum
from typing import Any, Optional

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoProcessor, BatchFeature, BitsAndBytesConfig

from maestro.trainer.common.utils.device import parse_device_spec

DEFAULT_PHI_4_MODEL_ID = "microsoft/Phi-4-multimodal-instruct"
DEFAULT_PHI_4_MODEL_REVISION = "refs/heads/main"


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
    )
    processor.tokenizer.padding_side = "right"
    attn_implementation = "flash_attention_2" if use_flash_attention else None

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

        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            revision=revision,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype="auto",
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
        )

        # Remove audio-related parameters
        model = _remove_audio_layers(model)

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            revision=revision,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="auto",
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
        )

        # Always remove audio-related parameters for vision-only processing
        model = _remove_audio_layers(model)

        if device != "auto" and device is not None:
            model.to(device)

    return processor, model


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
        print("Removing audio layers to optimize for vision-only processing...")

        # Remove audio encoder
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens_extend"):
            if hasattr(model.model.embed_tokens_extend, "audio_embed"):
                print("Removing audio embedding layer")
                del model.model.embed_tokens_extend.audio_embed

        # Remove audio lora layers from each transformer layer
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
                    print(f"Removed {removed_components} audio LoRA components from layer {layer_idx}")

        print("Audio layer removal complete")
    except Exception as e:
        print(
            f"Warning: Could not remove some audio layers. This is expected if using a different model variant. Error: {e}"
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

    # Create a copy of the inputs to avoid modifying the original
    filtered_inputs = {k: v for k, v in inputs.items() if k not in audio_related_keys}

    return filtered_inputs


def process_model_inputs(model: AutoModelForCausalLM, inputs: dict[str, Any], **kwargs) -> Any:
    """
    Process inputs before passing to the model, removing audio components.

    Args:
        model: The model to use for processing.
        inputs: Dictionary of input tensors and parameters.
        **kwargs: Additional arguments to pass to the model.

    Returns:
        The model's output after processing the filtered inputs.
    """
    # Filter out audio-related components
    filtered_inputs = filter_audio_components(inputs)

    # Add any additional kwargs
    filtered_inputs.update(kwargs)

    # Pass the filtered inputs to the model
    return model(**filtered_inputs)


def generate_with_model(model: AutoModelForCausalLM, inputs: dict[str, Any], **generation_kwargs) -> torch.LongTensor:
    """
    Generate text with the model after filtering out audio components.

    Args:
        model: The model to use for generation.
        inputs: Dictionary of input tensors and parameters.
        **generation_kwargs: Additional arguments for generation.

    Returns:
        The generated token IDs.
    """
    # Filter out audio-related components
    filtered_inputs = filter_audio_components(inputs)

    # Add any generation-specific parameters
    filtered_inputs.update(generation_kwargs)

    # Call the generate method with filtered inputs
    return model.generate(**filtered_inputs)


def save_model(
    target_dir: str,
    processor: AutoProcessor,
    model: AutoModelForCausalLM,
) -> None:
    """
    Save a Phi-4 model and its processor to disk.

    Args:
        target_dir: Directory path where the model and processor will be saved.
            Will be created if it doesn't exist.
        processor: The Phi-4 processor to save.
        model: The Phi-4 model to save.
    """
    os.makedirs(target_dir, exist_ok=True)
    processor.save_pretrained(target_dir)
    model.save_pretrained(target_dir)


def main():
    """Test function to verify model loading with vision-only processing."""
    import argparse
    from io import BytesIO

    import requests
    from PIL import Image

    # Parse arguments
    parser = argparse.ArgumentParser(description="Test Phi-4 model loading (vision-only)")
    parser.add_argument("--model_id", default=DEFAULT_PHI_4_MODEL_ID, help="Model ID or path")
    parser.add_argument("--revision", default=DEFAULT_PHI_4_MODEL_REVISION, help="Model revision")
    parser.add_argument(
        "--optimization", choices=["none", "lora", "qlora"], default="none", help="Optimization strategy"
    )
    parser.add_argument("--cache_dir", default=None, help="Cache directory for model")
    parser.add_argument("--no_flash_attention", action="store_true", help="Disable Flash Attention")
    args = parser.parse_args()

    # Get optimization strategy
    opt_strategy = OptimizationStrategy.NONE
    if args.optimization == "lora":
        opt_strategy = OptimizationStrategy.LORA
    elif args.optimization == "qlora":
        opt_strategy = OptimizationStrategy.QLORA

    print(f"Loading model {args.model_id} with {args.optimization} optimization (vision-only)...")

    # Load model
    processor, model = load_model(
        model_id_or_path=args.model_id,
        revision=args.revision,
        optimization_strategy=opt_strategy,
        cache_dir=args.cache_dir,
        use_flash_attention=not args.no_flash_attention,
    )

    print("Vision-only model loaded successfully!")

    try:
        sample_image_url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        response = requests.get(sample_image_url)
        image = Image.open(BytesIO(response.content))
        # Define prompt structure
        user_prompt = "<|user|>"
        assistant_prompt = "<|assistant|>"
        prompt_suffix = "<|end|>"
        prompt = f"{user_prompt}<|image_1|>What is shown in this image?{prompt_suffix}{assistant_prompt}"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

        # Filter out audio components before processing
        inputs = filter_audio_components(inputs)
        inputs = BatchFeature(inputs)
        input_len = inputs.input_ids.size(1)
        print("Generating response...")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, eos_token_id=processor.tokenizer.eos_token_id)

        generated_text = processor.batch_decode(outputs[:, input_len:], skip_special_tokens=True)[0]
        print("\nPrompt:", prompt)
        print("Response:", generated_text.split(prompt)[-1].strip())

    except Exception as e:
        print(f"Error during model testing: {e}")

    print("Test completed.")


if __name__ == "__main__":
    main()
