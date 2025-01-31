
<div align="center">

  <h1>maestro</h1>

  <br>
  
  <div>
    <a href="https://example1.com" style="margin: 0 10px;">
      <img
        src="https://github.com/user-attachments/assets/c9416f1f-a2bf-4590-86da-d2fc89ba559b"
        width="80"
        height="40"
      />
    </a>
    <a href="https://example2.com" style="margin: 0 10px;">
      <img
        src="https://github.com/user-attachments/assets/75dc7214-e82a-498d-950e-c64d90218e49"
        width="80"
        height="40"
      />
    </a>
    <a href="https://example3.com" style="margin: 0 10px;">
      <img
        src="https://github.com/user-attachments/assets/5d265473-b938-4501-b894-6a44a6a28a8c"
        width="80"
        height="40"
      />
    </a>
    <a href="https://example3.com" style="margin: 0 10px;">
      <img
        src="https://github.com/user-attachments/assets/b7ccdf39-ac77-4dbd-8608-0fa2d9dadf0a"
        width="80"
        height="40"
      />
    </a>
  </div>

  <br>

  [![version](https://badge.fury.io/py/maestro.svg)](https://badge.fury.io/py/maestro)

</div>

## Hello

**maestro** is a tool designed to streamline and accelerate the fine-tuning process for
multimodal models. It provides ready-to-use recipes for fine-tuning popular
vision-language models (VLMs) such as **Florence-2**, **PaliGemma 2**, and
**Qwen2.5-VL** on downstream vision-language tasks.

![maestro](https://github.com/user-attachments/assets/3bb9ccba-b0ee-4964-bcd6-f71124a08bc2)

## Quickstart

### Install

To get started with maestro, you’ll need to install the dependencies specific to the model you wish to fine-tune.

```bash
pip install maestro[qwen_2_5_vl]
```

**Note:** Some models may have clashing dependencies. We recommend creating a separate python environment for each model to avoid version conflicts.

### CLI

```bash
maestro qwen_2_5_vl train \
  --dataset "dataset/location" \
  --epochs 10 \
  --batch-size 4 \
  --optimization_strategy "qlora" \
  --metrics "edit_distance"
```

### Python

```python
from maestro.trainer.models.qwen_2_5_vl.core import train

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
