<div align="center">

  <h1>maestro</h1>

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
pip install maestro[paligemma_2]
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

We would love your help in making this repository even better! We are especially
looking for contributors with experience in fine-tuning vision-language models (VLMs).
If you notice any bugs or have suggestions for improvement, feel free to open an
[issue](https://github.com/roboflow/multimodal-maestro/issues) or submit a
[pull request](https://github.com/roboflow/multimodal-maestro/pulls).
