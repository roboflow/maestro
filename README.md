<div align="center">

  <h1>maestro</h1>

  <h3>VLM fine-tuning for everyone</h1>

  <br>

  <div>
      <img
        src="https://github.com/user-attachments/assets/c9416f1f-a2bf-4590-86da-d2fc89ba559b"
        width="80"
        height="40"
      />
      <img
        src="https://github.com/user-attachments/assets/75dc7214-e82a-498d-950e-c64d90218e49"
        width="80"
        height="40"
      />
      <img
        src="https://github.com/user-attachments/assets/5d265473-b938-4501-b894-6a44a6a28a8c"
        width="80"
        height="40"
      />
      <img
        src="https://github.com/user-attachments/assets/b7ccdf39-ac77-4dbd-8608-0fa2d9dadf0a"
        width="80"
        height="40"
      />
  </div>

  <br>

  [![version](https://badge.fury.io/py/maestro.svg)](https://badge.fury.io/py/maestro)
  [![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow/maestro/blob/develop/cookbooks/maestro_qwen2_5_vl_json_extraction.ipynb)

</div>

## Hello

**maestro** is a streamlined tool to accelerate the fine-tuning of multimodal models.
By encapsulating best practices from our core modules, maestro handles configuration,
data loading, reproducibility, and training loop setup. It currently offers ready-to-use
recipes for popular vision-language models such as **Florence-2**, **PaliGemma 2**, and
**Qwen2.5-VL**.

![maestro](https://github.com/user-attachments/assets/3bb9ccba-b0ee-4964-bcd6-f71124a08bc2)

## Quickstart

### Install

To begin, install the model-specific dependencies. Since some models may have clashing requirements,
we recommend creating a dedicated Python environment for each model.

```bash
pip install "maestro[paligemma_2]"
```

### CLI

Kick off fine-tuning with our command-line interface, which leverages the configuration
and training routines defined in each model’s core module. Simply specify key parameters such as
the dataset location, number of epochs, batch size, optimization strategy, and metrics.

```bash
maestro paligemma_2 train \
  --dataset "dataset/location" \
  --epochs 10 \
  --batch-size 4 \
  --optimization_strategy "qlora" \
  --metrics "edit_distance"
```

### Python

For greater control, use the Python API to fine-tune your models.
Import the train function from the corresponding module and define your configuration
in a dictionary. The core modules take care of reproducibility, data preparation,
and training setup.

```python
from maestro.trainer.models.paligemma_2.core import train

config = {
    "dataset": "dataset/location",
    "epochs": 10,
    "batch_size": 4,
    "optimization_strategy": "qlora",
    "metrics": ["edit_distance"]
}

train(config)
```

## Cookbooks
Looking for a place to start? Try our cookbooks to learn how to fine-tune different VLMs on various vision tasks with **maestro**.


| description                                             |                                                                                          open in colab                                                                                           |
|:--------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Finetune Florence-2 for object detection with LoRA      | [![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow/maestro/blob/develop/cookbooks/maestro_florence_2_object_detection.ipynb) |
| Finetune PaliGemma 2 for JSON data extraction with LoRA | [![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow/maestro/blob/develop/cookbooks/maestro_paligemma_2_json_extraction.ipynb) |
| Finetune Qwen2.5-VL for JSON data extraction with QLoRA | [![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow/maestro/blob/develop/cookbooks/maestro_qwen2_5_vl_json_extraction.ipynb)  |

## Contribution

We appreciate your input as we continue refining Maestro. Your feedback is invaluable in guiding our improvements. To
learn how you can help, please check out our [Contributing Guide](https://github.com/roboflow/maestro/blob/develop/CONTRIBUTING.md).
If you have any questions or ideas, feel free to start a conversation in our [GitHub Discussions](https://github.com/roboflow/maestro/discussions).
Thank you for being a part of our journey!
