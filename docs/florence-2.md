## Overview

Florence-2, released by Microsoft in June 2024, is a compact yet powerful
vision-language model (VLM) designed for a wide range of tasks, including object
detection, captioning, and optical character recognition (OCR). It uses a DaViT vision
encoder and BERT to process images and text into embeddings, which are then processed
through a transformer architecture. Florence-2’s strength lies in its pre-training on
the large FLD-5B dataset, which includes over 5 billion annotations for 126 million
images. The model can be fine-tuned for specific applications like visual question
answering (VQA) and document understanding, demonstrating significant improvements in
performance post-tuning. Its ability to adapt quickly makes it an attractive choice for
resource-constrained environments.

<iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/i3KjYgxNH6w" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen> </iframe>
Watch: Florence-2 Fine-tuning Overview

## Fine-tuning Examples

=== "Object Detection"

    ```txt

    ```

=== "Visual Question Answering"

    ```txt

    ```

## Dataset Format

The Florence-2 model expects a specific dataset structure for training and evaluation.
The dataset should be organized into train, test, and validation splits, with each
split containing image files and an `annotations.jsonl` file.

```
dataset/
├── train/
│   ├── 123e4567-e89b-12d3-a456-426614174000.png
│   ├── 987f6543-a21c-43c3-a562-926514273001.png
│   ├── ...
│   ├── annotations.jsonl
├── test/
│   ├── 456b7890-e32d-44f5-b678-724564172002.png
│   ├── 678c1234-e45b-67f6-c789-813264172003.png
│   ├── ...
│   ├── annotations.jsonl
└── valid/
    ├── 789d2345-f67c-89d7-e891-912354172004.png
    ├── 135e6789-d89f-12e3-f012-456464172005.png
    ├── ...
    └── annotations.jsonl
```

Each annotation entry contains the image filename and a description of the objects in
the image. The labels are structured as
`class name<loc_x_min><loc_y_min><loc_x_max><loc_y_max>`, where the bounding box
coordinates (`x_min`, `y_min`, `x_max`, `y_max`) are normalized to values between 0
and 1000.

```
{"image":"123e4567-e89b-12d3-a456-426614174000.png","prefix":"<OD>","suffix":"9 of clubs<loc_138><loc_100><loc_470><loc_448>10 of clubs<loc_388><loc_145><loc_670><loc_453>"}
{"image":"987f6543-a21c-43c3-a562-926514273001.png","prefix":"<OD>","suffix":"5 of clubs<loc_554><loc_2><loc_763><loc_467>6 of clubs<loc_399><loc_79><loc_555><loc_466>"}
...
```

## Examples

Below are two examples demonstrating how to train the Florence-2 model. The first
snippet is for CLI usage, allowing you to quickly train the model with simple commands.
The second snippet is for SDK usage, offering more customization and flexibility when
integrating Florence-2 into your own Python projects.

### CLI

```bash
maestro florence2 train --dataset='<DATASET_PATH>' --epochs=10 --batch-size=8
```

### SDK

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

## API

### Configuration

:::maestro.trainer.models.florence_2.core.Configuration

### train

:::maestro.trainer.models.florence_2.core.train

### evaluate

:::maestro.trainer.models.florence_2.core.evaluate
