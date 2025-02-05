## Overview

Florence-2 is a lightweight vision-language model open-sourced by Microsoft under the MIT license. It offers strong zero-shot and fine-tuning capabilities for tasks such as image captioning, object detection, visual grounding, and segmentation. Despite its compact size, training on the extensive FLD-5B dataset (126 million images and 5.4 billion annotations) enables Florence-2 to perform on par with much larger models like Kosmos-2. You can try out the model via HF Spaces, Google Colab, or our interactive playground.

## Install

```bash
pip install maestro[florence_2]
```

## Train

The training routines support various optimization strategies such as LoRA, and freezing the vision encoder. Customize your fine-tuning process via CLI or Python to align with your dataset and task requirements.

### CLI

Kick off training from the command line by running the command below. Be sure to replace the dataset path and adjust the hyperparameters (such as epochs and batch size) to suit your needs.

```bash
maestro florence_2 train \
  --dataset "dataset/location" \
  --epochs 10 \
  --batch-size 4 \
  --optimization_strategy "lora" \
  --metrics "edit_distance"
```

### Python

For more control, you can fine-tune Florence-2 using the Python API. Create a configuration dictionary with your training parameters and pass it to the train function to integrate the process into your custom workflow.

```python
from maestro.trainer.models.florence_2.core import train

config = {
    "dataset": "dataset/location",
    "epochs": 10,
    "batch_size": 4,
    "optimization_strategy": "lora",
    "metrics": ["edit_distance"],
}

train(config)
```

## Load

Load a pre-trained or fine-tuned Florence-2 model along with its processor using the load_model function. Specify your model's path and the desired optimization strategy.

```python
from maestro.trainer.models.florence_2.checkpoints import (
    OptimizationStrategy, load_model)

processor, model = load_model(
    model_id_or_path="model/location",
    optimization_strategy=OptimizationStrategy.NONE
)
```

## Predict

Perform inference with Florence-2 using the predict function. Supply an image and a text prefix to obtain predictions, such as object detection outputs or captions.

```python
from maestro.trainer.common.datasets import RoboflowJSONLDataset
from maestro.trainer.models.florence_2.inference import predict

ds = RoboflowJSONLDataset(
    jsonl_file_path="dataset/location/test/annotations.jsonl",
    image_directory_path="dataset/location/test",
)

image, entry = ds[0]

predict(model=model, processor=processor, image=image, prefix=entry["prefix"])
```
