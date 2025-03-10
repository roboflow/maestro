---
comments: true
---

## Overview

Phi-4 is Microsoft's state-of-the-art multimodal model that combines advanced vision capabilities with powerful language understanding. This multimodal model can process both images, audio and texts text inputs to perform various tasks including image understanding, reasoning, and generating contextually relevant responses.

The model excels at tasks requiring visual comprehension alongside natural language processing, making it suitable for applications such as visual question answering, image captioning, and multimodal reasoning.

## Install

To use Phi-4 with Maestro, install the required dependencies:

```bash
pip install "maestro[phi_4]"
```

This will install the necessary packages including transformers, torch, flash-attention (for non-macOS platforms), and other dependencies required for working with Phi-4.

## Train

Fine-tune the Phi-4 model using LoRA or other optimization strategies to adapt it to your specific multimodal tasks.

### CLI

Start training from the command line with the following command:

```bash
maestro phi_4 train \
  --dataset "dataset/location" \
  --epochs 10 \
  --batch-size 4 \
  --optimization_strategy "lora" \
  --metrics "edit_distance"
```

Customize the command with your specific dataset path and adjust hyperparameters like learning rate and batch size according to your requirements.

### Python

For more control over the training process, use the Python API:

```python
from maestro.trainer.models.phi_4.core import train

config = {
    "dataset": "dataset/location",
    "model_id": "microsoft/Phi-4-multimodal-instruct",
    "epochs": 10,
    "batch_size": 4,
    "lr": 1e-5,
    "optimization_strategy": "lora",
    "metrics": ["edit_distance"],
    "use_flash_attention": True,
}

train(config)
```

## Load

Load a pre-trained or fine-tuned Phi-4 model along with its processor:

```python
from maestro.trainer.models.phi_4.checkpoints import (
    OptimizationStrategy, load_model
)

processor, model = load_model(
    model_id_or_path="microsoft/Phi-4-multimodal-instruct",  # or your fine-tuned model path
    optimization_strategy=OptimizationStrategy.NONE,
    use_flash_attention=True
)
```

## Predict

Generate predictions using your Phi-4 model with the dedicated prediction function:

```python
from PIL import Image
from maestro.trainer.models.phi_4.inference import predict

# Load an image
image = Image.open("path/to/your/image.jpg")

# Generate a prediction
result = predict(
    model=model,
    processor=processor,
    image=image,
    prefix="Describe this image in detail:"
)

print(result)
```

For more control over the prediction process, you can use the lower-level API:

```python
import torch
from PIL import Image
from transformers import BatchFeature
from maestro.trainer.models.phi_4.checkpoints import filter_audio_components

# Load an image
image = Image.open("path/to/your/image.jpg")

# Prepare inputs
inputs = processor(
    text="Describe this image in detail:",
    images=image,
    return_tensors="pt"
)

# Filter out audio components if any
inputs = filter_audio_components(inputs)
inputs = BatchFeature(inputs)
input_len = inputs.input_ids.size(1)

# Generate response
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        eos_token_id=processor.tokenizer.eos_token_id
    )

# Decode the generated text
generated_text = processor.batch_decode(outputs[:, input_len:], skip_special_tokens=True)[0]
formatted_prompt = "Describe this image in detail:".strip()
response_text = generated_text.split(formatted_prompt)[-1].strip()

print(response_text)
```

This provides workflows for using Phi-4 for image understanding and text generation tasks. The simplified `predict` function handles the common case, while the lower-level API gives you more control over the generation process.
