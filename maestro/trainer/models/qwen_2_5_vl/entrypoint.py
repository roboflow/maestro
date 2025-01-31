import dataclasses
from typing import Annotated, Optional

import rich
import typer

from maestro.trainer.models.qwen_2_5_vl.checkpoints import (
    DEFAULT_QWEN2_5_VL_MODEL_ID,
    DEFAULT_QWEN2_5_VL_MODEL_REVISION,
)
from maestro.trainer.models.qwen_2_5_vl.core import Qwen25VLConfiguration
from maestro.trainer.models.qwen_2_5_vl.core import train as qwen_2_5_vl_train

qwen_2_5_vl_app = typer.Typer(help="Fine-tune and evaluate Qwen2.5-VL model")


@qwen_2_5_vl_app.command(
    help="Train Qwen2.5-VL model", context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def train(
    dataset: Annotated[
        str,
        typer.Option("--dataset", help="Path to the dataset in Roboflow JSONL format"),
    ],
    model_id: Annotated[
        str,
        typer.Option("--model_id", help="Identifier for the Qwen2.5-VL model from HuggingFace Hub"),
    ] = DEFAULT_QWEN2_5_VL_MODEL_ID,
    revision: Annotated[
        str,
        typer.Option("--revision", help="Model revision to use"),
    ] = DEFAULT_QWEN2_5_VL_MODEL_REVISION,
    device: Annotated[
        str,
        typer.Option("--device", help="Device to use for training"),
    ] = "auto",
    optimization_strategy: Annotated[
        str,
        typer.Option("--optimization_strategy", help="Optimization strategy: lora, qlora, or none"),
    ] = "lora",
    epochs: Annotated[
        int,
        typer.Option("--epochs", help="Number of training epochs"),
    ] = 10,
    lr: Annotated[
        float,
        typer.Option("--lr", help="Learning rate for training"),
    ] = 2e-4,
    batch_size: Annotated[
        int,
        typer.Option("--batch_size", help="Training batch size"),
    ] = 4,
    accumulate_grad_batches: Annotated[
        int,
        typer.Option("--accumulate_grad_batches", help="Number of batches to accumulate for gradient updates"),
    ] = 8,
    val_batch_size: Annotated[
        Optional[int],
        typer.Option("--val_batch_size", help="Validation batch size"),
    ] = None,
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
        typer.Option("--output_dir", help="Directory to store training outputs"),
    ] = "./training/qwen_2_5_vl",
    metrics: Annotated[
        list[str],
        typer.Option("--metrics", help="List of metrics to track during training"),
    ] = [],
    system_message: Annotated[
        Optional[str],
        typer.Option("--system_message", help="System message used during data loading"),
    ] = None,
    min_pixels: Annotated[
        int,
        typer.Option("--min_pixels", help="Minimum number of pixels for input images"),
    ] = 256 * 28 * 28,
    max_pixels: Annotated[
        int,
        typer.Option("--max_pixels", help="Maximum number of pixels for input images"),
    ] = 1280 * 28 * 28,
    max_new_tokens: Annotated[
        int,
        typer.Option("--max_new_tokens", help="Maximum number of new tokens generated during inference"),
    ] = 1024,
    random_seed: Annotated[
        Optional[int],
        typer.Option("--random_seed", help="Random seed for ensuring reproducibility. If None, no seed is set"),
    ] = None,
) -> None:
    config = Qwen25VLConfiguration(
        dataset=dataset,
        model_id=model_id,
        revision=revision,
        device=device,
        optimization_strategy=optimization_strategy,  # type: ignore
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        accumulate_grad_batches=accumulate_grad_batches,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        val_num_workers=val_num_workers,
        output_dir=output_dir,
        metrics=metrics,
        system_message=system_message,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        max_new_tokens=max_new_tokens,
        random_seed=random_seed,
    )
    typer.echo(typer.style(text="Training configuration", fg=typer.colors.BRIGHT_GREEN, bold=True))
    rich.print(dataclasses.asdict(config))
    qwen_2_5_vl_train(config=config)
