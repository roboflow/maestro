<div align="center">

  <h1>maestro</h1>

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

## Fine-tune VLMs for free

| model, task and acceleration                                |                                                                                          open in colab                                                                                           |
|:------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Florence-2 (0.9B) object detection with LoRA (experimental) | [![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow/maestro/blob/develop/cookbooks/maestro_florence_2_object_detection.ipynb) |
| PaliGemma 2 (3B) JSON data extraction with LoRA             | [![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow/maestro/blob/develop/cookbooks/maestro_paligemma_2_json_extraction.ipynb) |
| Qwen2.5-VL (3B) JSON data extraction with QLoRA             | [![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow/maestro/blob/develop/cookbooks/maestro_qwen2_5_vl_json_extraction.ipynb)  |

## News

- `2025/02/12` (`1.1.0rc1`): This prerelease adds native support for COCO datasets. Now you can fine-tune Florence-2 directly on your existing COCO data for seamless model adaptation.
- `2025/02/05` (`1.0.0`): This release introduces support for Florence-2, PaliGemma 2, and Qwen2.5-VL and includes LoRA, QLoRA, and graph freezing to keep hardware requirements in check. It offers a single CLI/SDK to reduce code complexity, and a consistent JSONL format to streamline data handling.

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

## Contribution

We appreciate your input as we continue refining Maestro. Your feedback is invaluable in guiding our improvements. To
learn how you can help, please check out our [Contributing Guide](https://github.com/roboflow/maestro/blob/develop/CONTRIBUTING.md).
If you have any questions or ideas, feel free to start a conversation in our [GitHub Discussions](https://github.com/roboflow/maestro/discussions).
Thank you for being a part of our journey!
