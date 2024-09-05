import dataclasses
from typing import Optional, Annotated

import torch
import typer
import rich

from maestro.trainer.models.florence_2.entities import (
    TrainingConfiguration,
    DEFAULT_FLORENCE2_MODEL_ID,
    DEFAULT_FLORENCE2_MODEL_REVISION,
    DEVICE,
)

from maestro.trainer.models.florence_2.training import train as train_fun

florence_2_app = typer.Typer(help="Fine-tune and evaluate Florence 2 model")


@florence_2_app.command(
    help="Train Florence 2 model", context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def train(
    dataset_location: Annotated[
        str,
        typer.Option("--dataset_location", help="Path to directory with dataset"),
    ],
    model_id_or_path: Annotated[
        str,
        typer.Option("--model_id_or_path", help="Model to be used or path to your checkpoint"),
    ] = DEFAULT_FLORENCE2_MODEL_ID,
    revision: Annotated[
        str,
        typer.Option("--revision", help="Revision of Florence2 HF repository"),
    ] = DEFAULT_FLORENCE2_MODEL_REVISION,
    device: Annotated[
        str,
        typer.Option("--device", help="CUDA device ID to be used (in format: 'cuda:0')"),
    ] = DEVICE,
    transformers_cache_dir: Annotated[
        Optional[str],
        typer.Option("--transformers_cache_dir", help="Cache dir for HF weights"),
    ] = None,
    training_epochs: Annotated[
        int,
        typer.Option("--training_epochs", help="Number of training epochs"),
    ] = 10,
    optimiser: Annotated[
        str,
        typer.Option("--optimiser", help="Optimiser to be used"),
    ] = "adamw",
    learning_rate: Annotated[
        float,
        typer.Option("--learning_rate", help="Learning rate"),
    ] = 1e-5,
    lr_scheduler: Annotated[
        str,
        typer.Option("--lr_scheduler", help="LR scheduler"),
    ] = "linear",
    train_batch_size: Annotated[
        int,
        typer.Option("--train_batch_size", help="Batch size for training"),
    ] = 4,
    test_batch_size: Annotated[
        Optional[int],
        typer.Option(
            "--train_batch_size", help="Batch size for validation and test. If not given - train will be used."
        ),
    ] = None,
    loaders_workers: Annotated[
        int,
        typer.Option("--loaders_workers", help="Number of loaders workers. 0 = # of CPU"),
    ] = 0,
    test_loaders_workers: Annotated[
        Optional[int],
        typer.Option(
            "--test_loaders_workers",
            help="Number of workers for test and val loaders. If not given - train will be used.",
        ),
    ] = None,
    lora_r: Annotated[
        int,
        typer.Option("--lora_r", help="Value of Lora R"),
    ] = 8,
    lora_alpha: Annotated[
        int,
        typer.Option("--lora_alpha", help="Value of Lora Alpha"),
    ] = 8,
    lora_dropout: Annotated[
        float,
        typer.Option("--lora_dropout", help="Value of Lora Dropout"),
    ] = 0.05,
    bias: Annotated[
        str,
        typer.Option("--bias", help="Value of Lora Bias"),
    ] = "none",
    use_rslora: Annotated[
        bool,
        typer.Option(
            "--use_rslora/--no_use_rslora",
            help="Boolean flag to decide if rslora to be used",
        ),
    ] = True,
    init_lora_weights: Annotated[
        str,
        typer.Option("--init_lora_weights", help="Lora weights initialisation"),
    ] = "gaussian",
    training_dir: Annotated[
        str,
        typer.Option("--training_dir", help="Path to directory where training outputs should be preserved"),
    ] = "./training/florence-2",
    max_checkpoints_to_keep: Annotated[
        int,
        typer.Option("--max_checkpoints_to_keep", help="Max checkpoints to keep"),
    ] = 3,
    num_samples_to_visualise: Annotated[
        int,
        typer.Option("--num_samples_to_visualise", help="Number of samples to visualise"),
    ] = 64,
) -> None:
    configuration = TrainingConfiguration(
        dataset_location=dataset_location,
        model_id_or_path=model_id_or_path,
        revision=revision,
        device=torch.device(device),
        transformers_cache_dir=transformers_cache_dir,
        training_epochs=training_epochs,
        optimiser=optimiser,  # type: ignore
        learning_rate=learning_rate,
        lr_scheduler=lr_scheduler,  # type: ignore
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        loaders_workers=loaders_workers,
        test_loaders_workers=test_loaders_workers,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,  # type: ignore
        use_rslora=use_rslora,
        init_lora_weights=init_lora_weights,  # type: ignore
        training_dir=training_dir,
        max_checkpoints_to_keep=max_checkpoints_to_keep,
        num_samples_to_visualise=num_samples_to_visualise,
    )
    typer.echo(typer.style("Training configuration", fg=typer.colors.BRIGHT_GREEN, bold=True))
    rich.print(dataclasses.asdict(configuration))
    train_fun(configuration=configuration)


@florence_2_app.command(help="Evaluate Florence 2 model")
def evaluate() -> None:
    pass
