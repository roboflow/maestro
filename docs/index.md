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

</div>

## Hello

**maestro** is a streamlined tool to accelerate the fine-tuning of multimodal models.
By encapsulating best practices from our core modules, maestro handles configuration,
data loading, reproducibility, and training loop setup. It currently offers ready-to-use
recipes for popular vision-language models such as **Florence-2**, **PaliGemma 2**, and
**Qwen2.5-VL**.

## Quickstart

### Install

To begin, install the model-specific dependencies. Since some models may have clashing requirements,
we recommend creating a dedicated Python environment for each model.

=== "Florence-2"

    ```bash
    pip install maestro[florence_2]
    ```

=== "PaliGemma 2"

    ```bash
    pip install maestro[paligemma_2]
    ```

=== "Qwen2.5-VL"

    ```bash
    pip install maestro[qwen_2_5_vl]
    pip install git+https://github.com/huggingface/transformers
    ```

    !!! warning
        Support for Qwen2.5-VL in transformers is experimental.
        For now, please install transformers from source to ensure compatibility.

### CLI

Kick off fine-tuning with our command-line interface, which leverages the configuration
and training routines defined in each model’s core module. Simply specify key parameters such as
the dataset location, number of epochs, batch size, optimization strategy, and metrics.

=== "Florence-2"

    ```bash
    maestro florence_2 train \
      --dataset "dataset/location" \
      --epochs 10 \
      --batch-size 4 \
      --optimization_strategy "lora" \
      --metrics "edit_distance"
    ```

=== "PaliGemma 2"

    ```bash
    maestro paligemma_2 train \
      --dataset "dataset/location" \
      --epochs 10 \
      --batch-size 4 \
      --optimization_strategy "qlora" \
      --metrics "edit_distance"
    ```

=== "Qwen2.5-VL"

    ```bash
    maestro qwen_2_5_vl train \
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

=== "Florence-2"

    ```python
    from maestro.trainer.models.florence_2.core import train

    config = {
        "dataset": "dataset/location",
        "epochs": 10,
        "batch_size": 4,
        "optimization_strategy": "qlora",
        "metrics": ["edit_distance"],
    }

    train(config)
    ```

=== "PaliGemma 2"

    ```python
    from maestro.trainer.models.paligemma_2.core import train

    config = {
        "dataset": "dataset/location",
        "epochs": 10,
        "batch_size": 4,
        "optimization_strategy": "qlora",
        "metrics": ["edit_distance"],
    }

    train(config)
    ```

=== "Qwen2.5-VL"

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
