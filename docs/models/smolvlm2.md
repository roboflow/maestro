---
comments: true
---

## Overview

SmolVLM2 is a lightweight vision-language model developed by Smol AI. It offers impressive capabilities for multimodal understanding while maintaining a compact size compared to larger VLMs. The model excels at tasks such as image captioning, visual question answering, and object detection, making it accessible for applications with limited computational resources.

Built to balance performance and efficiency, SmolVLM2 provides a valuable option for developers seeking to implement vision-language capabilities without the overhead of larger models. The 500M parameter variant delivers practical results while being significantly more resource-friendly than multi-billion parameter alternatives.

## Install

```bash
pip install "maestro[smolvlm2]"
```

## Train

The training routines support various optimization strategies such as LoRA, QLoRA, and freezing the vision encoder. Customize your fine-tuning process via CLI or Python to align with your dataset and task requirements.

### CLI

Kick off training from the command line by running the command below. Be sure to replace the dataset path and adjust the hyperparameters (such as epochs and batch size) to suit your needs.

```bash
maestro smolvlm2 train \
  --dataset "dataset/location" \
  --epochs 10 \
  --batch-size 4 \
  --optimization_strategy "qlora" \
  --metrics "edit_distance"
```

### Python

For more control, you can fine-tune SmolVLM2 using the Python API. Create a configuration dictionary with your training parameters and pass it to the train function to integrate the process into your custom workflow.

```python
from maestro.trainer.models.smolvlm2.core import train

config = {
    "dataset": "dataset/location",
    "epochs": 10,
    "batch_size": 4,
    "optimization_strategy": "qlora",
    "metrics": ["edit_distance"],
}

results = train(config)
```

## Inference

Use SmolVLM2 for inference on images using either the CLI or Python API.

### CLI

```bash
maestro smolvlm2 predict \
  --image "path/to/image.jpg" \
  --prompt "Describe this image"
```

### Python

```python
from maestro.trainer.models.smolvlm2.entrypoint import SmolVLM2

model = SmolVLM2()
result = model.generate(
    images="path/to/image.jpg",
    prompt="Describe this image",
    max_new_tokens=512
)

print(result["text"])
```

## Object Detection

SmolVLM2 can perform object detection on images, identifying and localizing objects with bounding boxes.

```python
from maestro.trainer.models.smolvlm2.entrypoint import SmolVLM2
from maestro.trainer.models.smolvlm2.detection import result_to_detections_formatter

model = SmolVLM2()
result = model.generate(
    images="path/to/image.jpg",
    prompt="Detect the following objects: person, car, dog"
)

# Convert text output to detections format
boxes, class_ids = result_to_detections_formatter(
    text=result["text"],
    resolution_wh=(640, 480),
    classes=["person", "car", "dog"]
)
```
