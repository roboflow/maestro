## Overview

Qwen2.5-VL is a cutting-edge vision-language model that integrates powerful visual understanding and advanced language processing in a unified framework. It excels across a range of tasks—from extensive image recognition and precise object grounding to sophisticated text extraction, document parsing, and dynamic video comprehension—making it ideal for both desktop and mobile applications.

Building on significant improvements over its predecessor, Qwen2-VL, the Qwen2.5-VL series (including the high-performing 7B-Instruct and the edge-optimized 3B variants) sets new standards by outperforming models like GPT-4o-mini in various tasks.

## Install

```bash
pip install maestro[qwen_2_5_vl]
pip install git+https://github.com/huggingface/transformers
```

!!! warning
    Support for Qwen2.5-VL in transformers is experimental.
    Please install transformers from source to ensure compatibility.

## Train

The training routines support various optimization strategies such as LoRA, QLoRA, and freezing the vision encoder. Customize your fine-tuning process via CLI or Python to align with your dataset and task requirements.

### CLI

Kick off training from the command line by running the command below. Be sure to replace the dataset path and adjust the hyperparameters (such as epochs and batch size) to suit your needs.

```bash
maestro qwen_2_5_vl train \
  --dataset "dataset/location" \
  --epochs 10 \
  --batch-size 4 \
  --optimization_strategy "qlora" \
  --metrics "edit_distance"
```

### Python

For more control, you can fine-tune Qwen2.5-VL using the Python API. Create a configuration dictionary with your training parameters and pass it to the train function to integrate the process into your custom workflow.

```python
from maestro.trainer.models.qwen_2_5_vl.core import train

config = {
    "dataset": "dataset/location",
    "epochs": 10,
    "batch_size": 4,
    "optimization_strategy": "qlora",
    "metrics": ["edit_distance"],
}

train(config)
```

## Load

Load a pre-trained or fine-tuned Qwen2.5-VL model along with its processor using the load_model function. Specify your model's path and the desired optimization strategy.

```python
from maestro.trainer.models.qwen_2_5_vl.checkpoints import (
    OptimizationStrategy, load_model
)

processor, model = load_model(
    model_id_or_path="model/location",
    optimization_strategy=OptimizationStrategy.NONE
)
```

## Predict

Perform inference with Qwen2.5-VL using the predict function. Supply an image and a text prefix to obtain predictions, such as object detection outputs or captions.

```python
from maestro.trainer.common.datasets import RoboflowJSONLDataset
from maestro.trainer.models.qwen_2_5_vl.inference import predict

ds = RoboflowJSONLDataset(
    jsonl_file_path="dataset/location/test/annotations.jsonl",
    image_directory_path="dataset/location/test",
)

image, entry = ds[0]

predict(model=model, processor=processor, image=image, prefix=entry["prefix"])
```
