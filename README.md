
<div align="center">

  <h1>maestro</h1>

  <p>easy VLMs fine-tuning</p>
  
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

</div>

## 👋 hello

**maestro** is a tool designed to streamline and accelerate the fine-tuning process for
multimodal models. It provides ready-to-use recipes for fine-tuning popular
vision-language models (VLMs) such as **Florence-2**, **PaliGemma**, and
**Qwen2-VL** on downstream vision-language tasks.

## 💻 install

Pip install the supervision package in a
[**Python>=3.8**](https://www.python.org/) environment.

```bash
pip install maestro
```

## 🔥 quickstart

### CLI

VLMs can be fine-tuned on downstream tasks directly from the command line with
`maestro` command:

```bash
maestro florence2 train --dataset='<DATASET_PATH>' --epochs=10 --batch-size=8
```

### SDK

Alternatively, you can fine-tune VLMs using the Python SDK, which accepts the same
arguments as the CLI example above:

```python
from maestro.trainer.common import MeanAveragePrecisionMetric
from maestro.trainer.models.florence_2 import train, Configuration

config = Configuration(
    dataset='<DATASET_PATH>',
    epochs=10,
    batch_size=8,
    metrics=[MeanAveragePrecisionMetric()]
)

train(config)
```

## 📚 notebooks

Explore our collection of notebooks that demonstrate how to fine-tune various
vision-language models using maestro. Each notebook provides step-by-step instructions
and code examples to help you get started quickly.

| model and task | colab | video                                                                                  |
|----------------|-------|----------------------------------------------------------------------------------------|
| [Fine-tune Florence-2 for object detection](https://github.com/roboflow/multimodal-maestro/blob/develop/cookbooks/maestro_florence2_object_detection.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow/multimodal-maestro/blob/develop/cookbooks/maestro_florence2_object_detection.ipynb) | [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://youtu.be/i3KjYgxNH6w) |
| [Fine-tune Florence-2 for visual question answering (VQA)](https://github.com/roboflow/multimodal-maestro/blob/develop/cookbooks/maestro_florence2_visual_question_answering.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow/multimodal-maestro/blob/develop/cookbooks/maestro_florence2_visual_question_answering.ipynb) | [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://youtu.be/i3KjYgxNH6w) |

## 🦸 contribution

We would love your help in making this repository even better! We are especially
looking for contributors with experience in fine-tuning vision-language models (VLMs).
If you notice any bugs or have suggestions for improvement, feel free to open an
[issue](https://github.com/roboflow/multimodal-maestro/issues) or submit a
[pull request](https://github.com/roboflow/multimodal-maestro/pulls).
