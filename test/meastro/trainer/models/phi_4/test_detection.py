from io import BytesIO

import pytest
import requests
import torch
from PIL import Image
from transformers import BatchFeature

from maestro.trainer.models.phi_4.checkpoints import (
    DEFAULT_PHI_4_MODEL_ID,
    DEFAULT_PHI_4_MODEL_REVISION,
    OptimizationStrategy,
    filter_audio_components,
    load_model,
)


@pytest.fixture(scope="module")
def sample_image():
    """Fixture to load a sample image for testing."""
    sample_image_url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    response = requests.get(sample_image_url)
    return Image.open(BytesIO(response.content))


@pytest.fixture(scope="module")
def prompt_structure():
    """Fixture to provide consistent prompt structure."""
    user_prompt = "<|user|>"
    assistant_prompt = "<|assistant|>"
    prompt_suffix = "<|end|>"
    return {"user_prompt": user_prompt, "assistant_prompt": assistant_prompt, "prompt_suffix": prompt_suffix}


@pytest.mark.parametrize(
    "optimization_strategy",
    [
        OptimizationStrategy.NONE,
    ],
)
@pytest.mark.parametrize("use_flash_attention", [True])
def test_phi4_vision_detection(
    sample_image,
    prompt_structure,
    optimization_strategy,
    use_flash_attention,
    cache_dir=None,
    model_id=DEFAULT_PHI_4_MODEL_ID,
    revision=DEFAULT_PHI_4_MODEL_REVISION,
):
    """Test Phi-4 model's ability to detect objects in images."""
    # Load model
    processor, model = load_model(
        model_id_or_path=model_id,
        revision=revision,
        optimization_strategy=optimization_strategy,
        cache_dir=cache_dir,
        use_flash_attention=use_flash_attention,
    )

    formatted_prompt = (
        f"{prompt_structure['user_prompt']}<|image_1|>What is shown in this image?"
        f"{prompt_structure['prompt_suffix']}{prompt_structure['assistant_prompt']}"
    )

    inputs = processor(text=formatted_prompt, images=sample_image, return_tensors="pt").to(model.device)

    inputs = filter_audio_components(inputs)
    inputs = BatchFeature(inputs)
    input_len = inputs.input_ids.size(1)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, eos_token_id=processor.tokenizer.eos_token_id)

    generated_text = processor.batch_decode(outputs[:, input_len:], skip_special_tokens=True)[0]
    response_text = generated_text.split(formatted_prompt)[-1].strip().lower()

    assert "stop sign" in response_text

    assert len(response_text) > 10, "Response is too short"
    assert isinstance(response_text, str), "Response should be a string"
