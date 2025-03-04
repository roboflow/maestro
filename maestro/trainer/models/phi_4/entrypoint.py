import dataclasses
from typing import Annotated, Optional

import rich
import typer

from maestro.trainer.models.phi_4.checkpoints import (
    DEFAULT_PHI_4_MODEL_ID,
    DEFAULT_PHI_4_MODEL_REVISION,
)
from maestro.trainer.models.phi_4.core import Phi4Configuration
from maestro.trainer.models.phi_4.core import train as phi_4_train

phi_4_app = typer.Typer(help="Fine-tune and evaluate Phi-4 multimodal model")


@phi_4_app.command(
    help="Train Phi-4 model", context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def train(
    dataset: Annotated[
        str,
        typer.Option(
            "--dataset",
            help="Local path or Roboflow identifier. If not found locally, it will be resolved (and downloaded) "
            "automatically",
        ),
    ],
    model_id: Annotated[
        str, typer.Option("--model_id", help="Identifier for the Phi-4 model")
    ] = DEFAULT_PHI_4_MODEL_ID,
    revision: Annotated[str, typer.Option("--revision", help="Model revision to use")] = DEFAULT_PHI_4_MODEL_REVISION,
    device: Annotated[str, typer.Option("--device", help="Device to use for training")] = "auto",
    optimization_strategy: Annotated[
        str, typer.Option("--optimization_strategy", help="Optimization strategy: lora, qlora, or none")
    ] = "lora",
    cache_dir: Annotated[
        Optional[str], typer.Option("--cache_dir", help="Directory to cache the model weights locally")
    ] = None,
    use_flash_attention: Annotated[
        bool, typer.Option("--use_flash_attention", help="Use Flash Attention 2 for faster training")
    ] = True,
    epochs: Annotated[int, typer.Option("--epochs", help="Number of training epochs")] = 10,
    lr: Annotated[float, typer.Option("--lr", help="Learning rate for training")] = 1e-5,
    batch_size: Annotated[int, typer.Option("--batch_size", help="Training batch size")] = 4,
    accumulate_grad_batches: Annotated[
        int, typer.Option("--accumulate_grad_batches", help="Number of batches to accumulate for gradient updates")
    ] = 8,
    val_batch_size: Annotated[Optional[int], typer.Option("--val_batch_size", help="Validation batch size")] = None,
    num_workers: Annotated[int, typer.Option("--num_workers", help="Number of workers for data loading")] = 0,
    val_num_workers: Annotated[
        Optional[int], typer.Option("--val_num_workers", help="Number of workers for validation data loading")
    ] = None,
    output_dir: Annotated[
        str, typer.Option("--output_dir", help="Directory to store training outputs")
    ] = "./training/phi_4",
    metrics: Annotated[list[str], typer.Option("--metrics", help="List of metrics to track during training")] = [],
    system_message: Annotated[
        Optional[str], typer.Option("--system_message", help="System message to include in prompts")
    ] = None,
    max_new_tokens: Annotated[
        int, typer.Option("--max_new_tokens", help="Maximum number of new tokens generated during inference")
    ] = 512,
    random_seed: Annotated[
        Optional[int],
        typer.Option("--random_seed", help="Random seed for ensuring reproducibility. If None, no seed is set"),
    ] = None,
) -> None:
    config = Phi4Configuration(
        dataset=dataset,
        model_id=model_id,
        revision=revision,
        device=device,
        optimization_strategy=optimization_strategy,  # type: ignore
        cache_dir=cache_dir,
        use_flash_attention=use_flash_attention,
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
        max_new_tokens=max_new_tokens,
        random_seed=random_seed,
    )
    typer.echo(typer.style(text="Training configuration", fg=typer.colors.BRIGHT_GREEN, bold=True))
    rich.print(dataclasses.asdict(config))
    phi_4_train(config=config)
