# import dataclasses
# from typing import Optional, Annotated

# import rich
# import torch
# import typer

# from maestro.trainer.models.florence_2.checkpoints import DEFAULT_FLORENCE2_MODEL_ID, \
#     DEFAULT_FLORENCE2_MODEL_REVISION, DEVICE
# from maestro.trainer.models.florence_2.core import TrainingConfiguration
# from maestro.trainer.models.florence_2.core import train as train_fun

# florence_2_app = typer.Typer(help="Fine-tune and evaluate Florence 2 model")


# @florence_2_app.command(
#     help="Train Florence 2 model", context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
# )
# def train(
#     dataset_location: Annotated[
#         str,
#         typer.Option("--dataset_location", help="Path to directory with dataset"),
#     ],
#     model_id_or_path: Annotated[
#         str,
#         typer.Option("--model_id_or_path", help="Model to be used or path to your checkpoint"),
#     ] = DEFAULT_FLORENCE2_MODEL_ID,
#     revision: Annotated[
#         str,
#         typer.Option("--revision", help="Revision of Florence2 HF repository"),
#     ] = DEFAULT_FLORENCE2_MODEL_REVISION,
#     device: Annotated[
#         str,
#         typer.Option("--device", help="CUDA device ID to be used (in format: 'cuda:0')"),
#     ] = DEVICE,
#     transformers_cache_dir: Annotated[
#         Optional[str],
#         typer.Option("--transformers_cache_dir", help="Cache dir for HF weights"),
#     ] = None,
#     training_epochs: Annotated[
#         int,
#         typer.Option("--training_epochs", help="Number of training epochs"),
#     ] = 10,
#     optimiser: Annotated[
#         str,
#         typer.Option("--optimiser", help="Optimiser to be used"),
#     ] = "adamw",
#     learning_rate: Annotated[
#         float,
#         typer.Option("--learning_rate", help="Learning rate"),
#     ] = 1e-5,
#     lr_scheduler: Annotated[
#         str,
#         typer.Option("--lr_scheduler", help="LR scheduler"),
#     ] = "linear",
#     train_batch_size: Annotated[
#         int,
#         typer.Option("--train_batch_size", help="Batch size for training"),
#     ] = 4,
#     test_batch_size: Annotated[
#         Optional[int],
#         typer.Option(
#             "--train_batch_size", help="Batch size for validation and test. If not given - train will be used."
#         ),
#     ] = None,
#     loaders_workers: Annotated[
#         int,
#         typer.Option("--loaders_workers", help="Number of loaders workers. 0 = # of CPU"),
#     ] = 0,
#     test_loaders_workers: Annotated[
#         Optional[int],
#         typer.Option(
#             "--test_loaders_workers",
#             help="Number of workers for test and val loaders. If not given - train will be used.",
#         ),
#     ] = None,
#     lora_r: Annotated[
#         int,
#         typer.Option("--lora_r", help="Value of Lora R"),
#     ] = 8,
#     lora_alpha: Annotated[
#         int,
#         typer.Option("--lora_alpha", help="Value of Lora Alpha"),
#     ] = 8,
#     lora_dropout: Annotated[
#         float,
#         typer.Option("--lora_dropout", help="Value of Lora Dropout"),
#     ] = 0.05,
#     bias: Annotated[
#         str,
#         typer.Option("--bias", help="Value of Lora Bias"),
#     ] = "none",
#     use_rslora: Annotated[
#         bool,
#         typer.Option(
#             "--use_rslora/--no_use_rslora",
#             help="Boolean flag to decide if rslora to be used",
#         ),
#     ] = True,
#     init_lora_weights: Annotated[
#         str,
#         typer.Option("--init_lora_weights", help="Lora weights initialisation"),
#     ] = "gaussian",
#     training_dir: Annotated[
#         str,
#         typer.Option("--training_dir", help="Path to directory where training outputs should be preserved"),
#     ] = "./training/florence-2",
#     max_checkpoints_to_keep: Annotated[
#         int,
#         typer.Option("--max_checkpoints_to_keep", help="Max checkpoints to keep"),
#     ] = 3,
#     num_samples_to_visualise: Annotated[
#         int,
#         typer.Option("--num_samples_to_visualise", help="Number of samples to visualise"),
#     ] = 64,
# ) -> None:
#     configuration = TrainingConfiguration(
#         dataset_location=dataset_location,
#         model_id_or_path=model_id_or_path,
#         revision=revision,
#         device=torch.device(device),
#         transformers_cache_dir=transformers_cache_dir,
#         training_epochs=training_epochs,
#         optimiser=optimiser,  # type: ignore
#         learning_rate=learning_rate,
#         lr_scheduler=lr_scheduler,  # type: ignore
#         train_batch_size=train_batch_size,
#         test_batch_size=test_batch_size,
#         loaders_workers=loaders_workers,
#         test_loaders_workers=test_loaders_workers,
#         lora_r=lora_r,
#         lora_alpha=lora_alpha,
#         lora_dropout=lora_dropout,
#         bias=bias,  # type: ignore
#         use_rslora=use_rslora,
#         init_lora_weights=init_lora_weights,  # type: ignore
#         training_dir=training_dir,
#         max_checkpoints_to_keep=max_checkpoints_to_keep,
#         num_samples_to_visualise=num_samples_to_visualise,
#     )
#     typer.echo(typer.style("Training configuration", fg=typer.colors.BRIGHT_GREEN, bold=True))
#     rich.print(dataclasses.asdict(configuration))
#     train_fun(configuration=configuration)


# @florence_2_app.command(help="Evaluate Florence 2 model")
# def evaluate() -> None:
#     pass

import typer
from typing import Optional, List, Union, Literal
from maestro.trainer.models.florence_2.core import TrainingConfiguration, train as train_florence2

app = typer.Typer()

@app.command()
def florence2(
    mode: str = typer.Option(..., help="Mode to run: train or evaluate"),
    dataset_path: str = typer.Option(..., help="Path to the dataset used for training"),
    model_id: str = typer.Option(None, help="Identifier for the Florence-2 model"),
    revision: str = typer.Option(None, help="Revision of the model to use"),
    device: str = typer.Option(None, help="Device to use for training"),
    cache_dir: Optional[str] = typer.Option(None, help="Directory to cache the model"),
    epochs: int = typer.Option(10, help="Number of training epochs"),
    optimizer: str = typer.Option("adamw", help="Optimizer to use for training"),
    lr: float = typer.Option(1e-5, help="Learning rate for the optimizer"),
    lr_scheduler: str = typer.Option("linear", help="Learning rate scheduler"),
    batch_size: int = typer.Option(4, help="Batch size for training"),
    val_batch_size: Optional[int] = typer.Option(None, help="Batch size for validation"),
    num_workers: int = typer.Option(0, help="Number of workers for data loading"),
    val_num_workers: Optional[int] = typer.Option(None, help="Number of workers for validation data loading"),
    lora_r: int = typer.Option(8, help="Rank of the LoRA update matrices"),
    lora_alpha: int = typer.Option(8, help="Scaling factor for the LoRA update"),
    lora_dropout: float = typer.Option(0.05, help="Dropout probability for LoRA layers"),
    bias: str = typer.Option("none", help="Which bias to train"),
    use_rslora: bool = typer.Option(True, help="Whether to use RSLoRA"),
    init_lora_weights: str = typer.Option("gaussian", help="How to initialize LoRA weights"),
    output_dir: str = typer.Option("./training/florence-2", help="Directory to save output files"),
    metrics: List[str] = typer.Option([], help="List of metrics to track during training")
):
    """Main entry point for Florence-2 model."""
    if mode == "train":
        train(
            dataset_path=dataset_path,
            model_id=model_id,
            revision=revision,
            device=device,
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
            metrics=metrics
        )
    elif mode == "evaluate":
        evaluate()
    else:
        typer.echo(f"Unknown mode: {mode}")
        raise typer.Exit(code=1)

def train(**kwargs):
    """Train a Florence-2 model."""
    # Filter out None values
    config_overrides = {k: v for k, v in kwargs.items() if v is not None}
    
    # Create configuration with overrides
    config = TrainingConfiguration(**config_overrides)
    
    train_florence2(config)

def evaluate():
    """Evaluate a Florence-2 model."""
    typer.echo("Evaluation not implemented yet.")

if __name__ == "__main__":
    app()
