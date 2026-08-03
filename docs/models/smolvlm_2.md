---
comments: true
---

## Overview

SmolVLM2 is a lightweight vision-language model developed by Hugging Face. It offers impressive capabilities for multimodal understanding while maintaining a compact size compared to larger VLMs. The model excels at tasks such as image captioning, visual question answering, and object detection, making it accessible for applications with limited computational resources.

Built to balance performance and efficiency, SmolVLM2 provides a valuable option for developers seeking to implement vision-language capabilities without the overhead of larger models. The 500M parameter variant delivers practical results while being significantly more resource-friendly than multi-billion parameter alternatives.

## Install

```bash
pip install "maestro[smolvlm_2]"
```

## Train

The training routines support various optimization strategies such as LoRA, QLoRA, and freezing the vision encoder. Customize your fine-tuning process via CLI or Python to align with your dataset and task requirements.

### CLI

Kick off training from the command line by running the command below. Be sure to replace the dataset path and adjust the hyperparameters (such as epochs and batch size) to suit your needs.

```bash
maestro smolvlm_2 train \
  --model_id "HuggingFaceTB/SmolVLM-500M-Instruct" \
  --dataset "dataset/location" \
  --epochs 10 \
  --batch-size 4 \
  --accumulate_grad_batches 4 \
  --optimization_strategy "lora" \
  --metrics "edit_distance"
```



### Python
```python
from maestro.trainer.models.smolvlm_2.core import train

config = {
    "model_id": "HuggingFaceTB/SmolVLM-500M-Instruct",
    "dataset": "dataset/location",
    "lr": 2e-5,
    "epochs": 10,
    "batch_size": 4,
    "accumulate_grad_batches": 4,
    "num_workers": 0,
    "optimization_strategy": "lora",
    "metrics": ["edit_distance"],
    "device": "cuda"
}


train(config)
```


## Load

Load a pre-trained or fine-tuned SmolVLM model along with its processor using the load_model function. Specify your model's path and the desired optimization strategy.

```python
from maestro.trainer.models.smolvlm_2.checkpoints import (
    OptimizationStrategy, load_model
)

processor, model = load_model(
    model_id_or_path="model/location",
    optimization_strategy=OptimizationStrategy.NONE
)
```
## Predict

Perform inference with SmolVLM using the predict function. Supply an image and a text prefix to obtain predictions, such as object detection outputs or captions.

```python
from maestro.trainer.common.datasets.jsonl import JSONLDataset
from maestro.trainer.models.smolvlm_2.inference import predict

ds = JSONLDataset(
    jsonl_file_path="dataset/location/test/annotations.jsonl",
    image_directory_path="dataset/location/test",
)

image, entry = ds[0]

predict(model=model, processor=processor, image=image, prefix=entry["prefix"])
```

