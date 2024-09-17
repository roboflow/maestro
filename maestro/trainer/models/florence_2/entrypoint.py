import dataclasses
from typing import Annotated, Literal, Optional, Union

import rich
import torch
import typer

from maestro.trainer.common.utils.metrics import BaseMetric, MeanAveragePrecisionMetric
from maestro.trainer.models.florence_2.checkpoints import (
    DEFAULT_FLORENCE2_MODEL_ID,
    DEFAULT_FLORENCE2_MODEL_REVISION,
    DEVICE,
)
from maestro.trainer.models.florence_2.core import LoraInitLiteral, TrainingConfiguration
from maestro.trainer.models.florence_2.core import evaluate as florence2_evaluate
from maestro.trainer.models.florence_2.core import train as florence2_train

florence_2_app = typer.Typer(help="Fine-tune and evaluate Florence 2 model")


METRIC_CLASSES: dict[str, type[BaseMetric]] = {
    "mean_average_precision": MeanAveragePrecisionMetric,
}


def parse_metrics(metrics: list[str]) -> list[BaseMetric]:
    metric_objects = []
    for metric_name in metrics:
        metric_class = METRIC_CLASSES.get(metric_name.lower())
        if metric_class:
            metric_objects.append(metric_class())
        else:
            raise ValueError(f"Unsupported metric: {metric_name}")
    return metric_objects


@florence_2_app.command(
    help="Train Florence 2 model", context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def train(
    dataset: Annotated[
        str,
        typer.Option("--dataset", help="Path to the dataset used for training"),
    ],
    model_id: Annotated[
        str,
        typer.Option("--model_id", help="Identifier for the Florence-2 model"),
    ] = DEFAULT_FLORENCE2_MODEL_ID,
    revision: Annotated[
        str,
        typer.Option("--revision", help="Revision of the model to use"),
    ] = DEFAULT_FLORENCE2_MODEL_REVISION,
    device: Annotated[
        str,
        typer.Option("--device", help="Device to use for training"),
    ] = DEVICE,
    cache_dir: Annotated[
        Optional[str],
        typer.Option("--cache_dir", help="Directory to cache the model"),
    ] = None,
    epochs: Annotated[
        int,
        typer.Option("--epochs", help="Number of training epochs"),
    ] = 10,
    optimizer: Annotated[
        Literal["sgd", "adamw", "adam"],
        typer.Option("--optimizer", help="Optimizer to use for training"),
    ] = "adamw",
    lr: Annotated[
        float,
        typer.Option("--lr", help="Learning rate for the optimizer"),
    ] = 1e-5,
    lr_scheduler: Annotated[
        Literal["linear", "cosine", "polynomial"],
        typer.Option("--lr_scheduler", help="Learning rate scheduler"),
    ] = "linear",
    batch_size: Annotated[
        int,
        typer.Option("--batch_size", help="Batch size for training"),
    ] = 4,
    val_batch_size: Annotated[
        Optional[int],
        typer.Option("--val_batch_size", help="Batch size for validation"),
    ] = None,
    num_workers: Annotated[
        int,
        typer.Option("--num_workers", help="Number of workers for data loading"),
    ] = 0,
    val_num_workers: Annotated[
        Optional[int],
        typer.Option("--val_num_workers", help="Number of workers for validation data loading"),
    ] = None,
    lora_r: Annotated[
        int,
        typer.Option("--lora_r", help="Rank of the LoRA update matrices"),
    ] = 8,
    lora_alpha: Annotated[
        int,
        typer.Option("--lora_alpha", help="Scaling factor for the LoRA update"),
    ] = 8,
    lora_dropout: Annotated[
        float,
        typer.Option("--lora_dropout", help="Dropout probability for LoRA layers"),
    ] = 0.05,
    bias: Annotated[
        Literal["none", "all", "lora_only"],
        typer.Option("--bias", help="Which bias to train"),
    ] = "none",
    use_rslora: Annotated[
        bool,
        typer.Option("--use_rslora/--no_use_rslora", help="Whether to use RSLoRA"),
    ] = True,
    init_lora_weights: Annotated[
        Union[bool, LoraInitLiteral],
        typer.Option("--init_lora_weights", help="How to initialize LoRA weights"),
    ] = "gaussian",
    output_dir: Annotated[
        str,
        typer.Option("--output_dir", help="Directory to save output files"),
    ] = "./training/florence-2",
    metrics: Annotated[
        list[str],
        typer.Option("--metrics", help="List of metrics to track during training"),
    ] = [],
) -> None:
    metric_objects = parse_metrics(metrics)
    config = TrainingConfiguration(
        dataset=dataset,
        model_id=model_id,
        revision=revision,
        device=torch.device(device),
        cache_dir=cache_dir,
        epochs=epochs,
        optimizer=optimizer,
        lr=lr,
        lr_scheduler=lr_scheduler,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        val_num_workers=val_num_workers,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        use_rslora=use_rslora,
        init_lora_weights=init_lora_weights,
        output_dir=output_dir,
        metrics=metric_objects,
    )
    typer.echo(typer.style(text="Training configuration", fg=typer.colors.BRIGHT_GREEN, bold=True))
    rich.print(dataclasses.asdict(config))
    florence2_train(config=config)


@florence_2_app.command(help="Evaluate Florence 2 model")
def evaluate(
    dataset: Annotated[
        str,
        typer.Option("--dataset", help="Path to the dataset used for evaluation"),
    ],
    model_id: Annotated[
        str,
        typer.Option("--model_id", help="Identifier for the Florence-2 model"),
    ] = DEFAULT_FLORENCE2_MODEL_ID,
    revision: Annotated[
        str,
        typer.Option("--revision", help="Revision of the model to use"),
    ] = DEFAULT_FLORENCE2_MODEL_REVISION,
    device: Annotated[
        str,
        typer.Option("--device", help="Device to use for evaluation"),
    ] = DEVICE,
    cache_dir: Annotated[
        Optional[str],
        typer.Option("--cache_dir", help="Directory to cache the model"),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option("--batch_size", help="Batch size for evaluation"),
    ] = 4,
    num_workers: Annotated[
        int,
        typer.Option("--num_workers", help="Number of workers for data loading"),
    ] = 0,
    val_num_workers: Annotated[
        Optional[int],
        typer.Option("--val_num_workers", help="Number of workers for validation data loading"),
    ] = None,
    output_dir: Annotated[
        str,
        typer.Option("--output_dir", help="Directory to save output files"),
    ] = "./evaluation/florence-2",
    metrics: Annotated[
        list[str],
        typer.Option("--metrics", help="List of metrics to track during evaluation"),
    ] = [],
) -> None:
    metric_objects = parse_metrics(metrics)
    config = TrainingConfiguration(
        dataset=dataset,
        model_id=model_id,
        revision=revision,
        device=torch.device(device),
        cache_dir=cache_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        val_num_workers=val_num_workers,
        output_dir=output_dir,
        metrics=metric_objects,
    )
    typer.echo(typer.style(text="Evaluation configuration", fg=typer.colors.BRIGHT_GREEN, bold=True))
    rich.print(dataclasses.asdict(config))
    florence2_evaluate(config=config)
