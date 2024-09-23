## Overview

Florence-2 is a lightweight vision-language model open-sourced by Microsoft under the
MIT license. The model demonstrates strong zero-shot and fine-tuning capabilities
across tasks such as captioning, object detection, grounding, and segmentation.

<iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/i3KjYgxNH6w" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen> </iframe>
*Florence-2: Fine-tune Microsoft’s Multimodal Model.*

## Architecture

The model takes images and task prompts as input, generating the desired results in
text format. It uses a DaViT vision encoder to convert images into visual token
embeddings. These are then concatenated with BERT-generated text embeddings and
processed by a transformer-based multi-modal encoder-decoder to generate the response.

![florence-2-architecture](https://storage.googleapis.com/com-roboflow-marketing/maestro/florence-2-architecture.webp)
*Overview of Florence-2 architecture. Source: Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks.*


## Fine-tuning Examples

### Dataset Format

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

Depending on the vision task being performed, the structure of the `annotations.jsonl`
file will vary slightly.

!!! warning
    The dataset samples shown below are formatted for improved readability, with each
    JSON structure spread across multiple lines. In practice, the `annotations.jsonl`
    file must contain each JSON object on a single line, without any line breaks
    between the key-value pairs. Make sure to adhere to this structure to avoid parsing
    errors during model training.

=== "Object Detection"

    ```txt
    {
        "image":"123e4567-e89b-12d3-a456-426614174000.png",
        "prefix":"<OD>",
        "suffix":"9 of clubs<loc_138><loc_100><loc_470><loc_448>10 of clubs<loc_388><loc_145><loc_670><loc_453>"
    }
    {
        "image":"987f6543-a21c-43c3-a562-926514273001.png",
        "prefix":"<OD>",
        "suffix":"5 of clubs<loc_554><loc_2><loc_763><loc_467>6 of clubs<loc_399><loc_79><loc_555><loc_466>"
    }
    ...
    ```

=== "Visual Question Answering (VQA)"

    ```txt
    {
        "image":"123e4567-e89b-12d3-a456-426614174000.png",
        "prefix":"<VQA> Is the value of Favorable 38 in 2015?",
        "suffix":"Yes"
    }
    {
        "image":"987f6543-a21c-43c3-a562-926514273001.png",
        "prefix":"<VQA> How many values are below 40 in Unfavorable graph?",
        "suffix":"6"
    }
    ...
    ```

=== "Object Character Recognition (OCR)"

    ```txt
    {
        "image":"123e4567-e89b-12d3-a456-426614174000.png",
        "prefix":"<OCR>",
        "suffix":"ke begherte Die mi"
    }
    {
        "image":"987f6543-a21c-43c3-a562-926514273001.png",
        "prefix":"<OCR>",
        "suffix":"mi uort in de middelt"
    }
    ...
    ```

### CLI

!!! tip
    Depending on the GPU you are using, you may need to adjust the `batch-size` to
    ensure that your model trains within memory limits. For larger GPUs with more
    memory, you can increase the batch size for better performance.

!!! tip
    Depending on the vision task you are executing, you may need to select different
    vision metrics. For example, tasks like object detection typically use
    `mean_average_precision`, while VQA and OCR tasks use metrics like
    `word_error_rate` and `character_error_rate`.

!!! tip
    You may need to use different learning rates depending on the task. We have found
    that lower learning rates work better for tasks like OCR or VQA, as these tasks
    require more precision.


=== "Object Detection"

    ```bash
    maestro florence2 train --dataset='<DATASET_PATH>' \
    --epochs=10 --batch-size=8 --lr=5e-6 --metrics=mean_average_precision
    ```

=== "Visual Question Answering (VQA)"

    ```bash
    maestro florence2 train --dataset='<DATASET_PATH>' \
    --epochs=10 --batch-size=8 --lr=1e-6 \
    --metrics=word_error_rate, character_error_rate
    ```

=== "Object Character Recognition (OCR)"

    ```bash
    maestro florence2 train --dataset='<DATASET_PATH>' \
    --epochs=10 --batch-size=8 --lr=1e-6 \
    --metrics=word_error_rate, character_error_rate
    ```

### SDK

=== "Object Detection"

    ```python
    from maestro.trainer.common import MeanAveragePrecisionMetric
    from maestro.trainer.models.florence_2 import train, Configuration

    config = Configuration(
        dataset='<DATASET_PATH>',
        epochs=10,
        batch_size=8,
        lr=5e-6,
        metrics=[MeanAveragePrecisionMetric()]
    )

    train(config)
    ```

=== "Visual Question Answering (VQA)"

    ```python
    from maestro.trainer.common import WordErrorRateMetric, CharacterErrorRateMetric
    from maestro.trainer.models.florence_2 import train, Configuration

    config = Configuration(
        dataset='<DATASET_PATH>',
        epochs=10,
        batch_size=8,
        lr=1e-6,
        metrics=[WordErrorRateMetric(), CharacterErrorRateMetric()]
    )

    train(config)
    ```

=== "Object Character Recognition (OCR)"

    ```python
    from maestro.trainer.common import WordErrorRateMetric, CharacterErrorRateMetric
    from maestro.trainer.models.florence_2 import train, Configuration

    config = Configuration(
        dataset='<DATASET_PATH>',
        epochs=10,
        batch_size=8,
        lr=1e-6,
        metrics=[WordErrorRateMetric(), CharacterErrorRateMetric()]
    )

    train(config)
    ```

## API

<div class="md-typeset">
    <h2><a href="#maestro.trainer.models.florence_2.core.Configuration">Configuration</a></h2>
</div>

:::maestro.trainer.models.florence_2.core.Configuration

<div class="md-typeset">
    <h2><a href="#maestro.trainer.models.florence_2.core.train">train</a></h2>
</div>

:::maestro.trainer.models.florence_2.core.train

<div class="md-typeset">
    <h2><a href="#maestro.trainer.models.florence_2.core.evaluate">evaluate</a></h2>
</div>

:::maestro.trainer.models.florence_2.core.evaluate
