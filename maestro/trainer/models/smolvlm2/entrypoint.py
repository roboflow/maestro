import dataclasses
import json
from typing import Annotated, Any, Optional

import rich
import typer

from maestro.trainer.logger import get_maestro_logger
from maestro.trainer.models.smolvlm2.checkpoints import DEFAULT_SMOLVLM2_MODEL_ID, DEFAULT_SMOLVLM2_MODEL_REVISION
from maestro.trainer.models.smolvlm2.core import SmolVLM2Configuration
from maestro.trainer.models.smolvlm2.core import train as smolvlm2_train

logger = get_maestro_logger()
smolvlm2_app = typer.Typer(help="Fine-tune and evaluate SmolVLM2 model")


@smolvlm2_app.command(
    help="Train SmolVLM2 model",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
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
        str, typer.Option("--model_id", help="Identifier for the SmolVLM2 model")
    ] = DEFAULT_SMOLVLM2_MODEL_ID,
    revision: Annotated[
        str, typer.Option("--revision", help="Model revision to use")
    ] = DEFAULT_SMOLVLM2_MODEL_REVISION,
    device: Annotated[str, typer.Option("--device", help="Device to use for training")] = "auto",
    optimization_strategy: Annotated[
        str, typer.Option("--optimization_strategy", help="Optimization strategy: lora, freeze, or none")
    ] = "lora",
    cache_dir: Annotated[
        Optional[str], typer.Option("--cache_dir", help="Directory to cache the model weights locally")
    ] = None,
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
    ] = "./training/smolvlm2",
    metrics: Annotated[list[str], typer.Option("--metrics", help="List of metrics to track during training")] = [],
    max_new_tokens: Annotated[
        int,
        typer.Option("--max_new_tokens", help="Maximum number of new tokens generated during inference"),
    ] = 1024,
    random_seed: Annotated[
        Optional[int],
        typer.Option("--random_seed", help="Random seed for ensuring reproducibility. If None, no seed is set"),
    ] = None,
    peft_advanced_params: Annotated[
        Optional[str],
        typer.Option("--peft_advanced_params", help="custom LoRA config. If None, default LoRA config is set"),
    ] = None,
) -> None:
    def parse_lora_params(param_str) -> dict[str, Any]:
        parsed_params = json.loads(param_str)
        if not isinstance(parsed_params, dict):
            raise TypeError("Parsed JSON is not a dictionary")
        return parsed_params

    if peft_advanced_params is not None:
        try:
            peft_advanced_params_dict = parse_lora_params(peft_advanced_params)
            logger.info(f"Parsed LoRA parameters: {peft_advanced_params_dict}")
        except json.JSONDecodeError:
            logger.exception("Failed to parse JSON")
            raise
        except TypeError:
            logger.exception("Invalid LoRA parameter format")
            raise

    config = SmolVLM2Configuration(
        dataset=dataset,
        model_id=model_id,
        revision=revision,
        device=device,
        optimization_strategy=optimization_strategy,  # type: ignore
        cache_dir=cache_dir,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        accumulate_grad_batches=accumulate_grad_batches,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        val_num_workers=val_num_workers,
        output_dir=output_dir,
        metrics=metrics,
        max_new_tokens=max_new_tokens,
        random_seed=random_seed,
        peft_advanced_params=peft_advanced_params_dict,
    )
    typer.echo(typer.style("Training configuration", fg=typer.colors.BRIGHT_GREEN, bold=True))
    rich.print(dataclasses.asdict(config))
    smolvlm2_train(config=config)










# from pathlib import Path
# from typing import Optional, Union

# import torch
# import typer

# from .inference import SmolVLM2Inference

# smolvlm2_app = typer.Typer()


# class SmolVLM2:
#     """Main entrypoint for SmolVLM2 model."""

#     def __init__(
#         self,
#         model_name: str = "smol-ai/smolvlm2-500m",
#         device: str = "cuda" if torch.cuda.is_available() else "cpu",
#         **kwargs,
#     ):
#         """Initialize SmolVLM2 model."""
#         self.inference = SmolVLM2Inference(model_name=model_name, device=device, **kwargs)

#     def generate(
#         self, images: Union[str, list[str]], prompt: Optional[str] = None, max_new_tokens: int = 512, **kwargs
#     ) -> dict:
#         """
#         Generate text from images.

#         Args:
#             images: Path(s) to image(s)
#             prompt: Optional prompt to guide generation
#             max_new_tokens: Maximum number of tokens to generate
#             **kwargs: Additional generation parameters

#         Returns:
#             Dictionary containing generated text and other outputs
#         """
#         return self.inference.generate(images=images, prompt=prompt, max_new_tokens=max_new_tokens, **kwargs)


# @smolvlm2_app.command(name="info", help="Get information about the SmolVLM2 model")
# def info() -> None:
#     """Get information about the SmolVLM2 model."""
#     try:
#         model = SmolVLM2()
#         info = model.inference.get_model_info()
#         typer.echo(f"Model Name: {info['model_name']}")
#         typer.echo(f"Model Size: {info['model_size']}")
#         typer.echo(f"Device: {info['device']}")
#         typer.echo(f"Tokenizer: {info['tokenizer']}")
#     except Exception as e:
#         typer.echo(f"Error retrieving model info: {e!s}", err=True)
#         raise typer.Exit(code=1)


# @smolvlm2_app.command(name="predict", help="Run inference on one or more images")
# def predict(
#     image: list[Path] = typer.Option(..., "--image", "-i", help="Path to image(s) for prediction"),
#     prompt: Optional[str] = typer.Option(None, "--prompt", "-p", help="Optional prompt to guide generation"),
#     max_new_tokens: int = typer.Option(512, "--max-new-tokens", help="Maximum new tokens to generate"),
#     output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path to save results"),
# ) -> None:
#     """Run inference on images using SmolVLM2."""
#     try:
#         model = SmolVLM2()
#         result = model.generate(images=[str(img) for img in image], prompt=prompt, max_new_tokens=max_new_tokens)

#         if output:
#             import json

#             with open(output, "w") as f:
#                 json.dump(result, f, indent=2)
#             typer.echo(f"Results saved to {output}")
#         else:
#             typer.echo(f"Generated text: {result['text']}")

#     except Exception as e:
#         typer.echo(f"Error during prediction: {e!s}", err=True)
#         raise typer.Exit(code=1)


# @smolvlm2_app.command(name="train", help="Fine-tune the SmolVLM2 model")
# def train(
#     dataset: Path = typer.Option(..., "--dataset", "-d", help="Path to dataset directory or file"),
#     epochs: int = typer.Option(10, "--epochs", "-e", help="Number of training epochs"),
#     batch_size: int = typer.Option(4, "--batch-size", "-b", help="Training batch size"),
#     optimization_strategy: str = typer.Option(
#         "qlora", "--optimization-strategy", "-o", help="Optimization strategy (qlora, lora, freeze_vision)"
#     ),
#     metrics: list[str] = typer.Option(["edit_distance"], "--metrics", "-m", help="Metrics to evaluate during training"),
#     output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Directory to save trained model"),
# ) -> None:
#     """Fine-tune the SmolVLM2 model on a dataset."""
#     try:
#         typer.echo("Starting SmolVLM2 fine-tuning...")

#         if output_dir is None:
#             import tempfile

#             output_dir = Path(tempfile.mkdtemp())
#             typer.echo(f"No output directory specified, using temporary directory: {output_dir}")

#         # Create configuration for training
#         config = {
#             "dataset": str(dataset),
#             "epochs": epochs,
#             "batch_size": batch_size,
#             "optimization_strategy": optimization_strategy,
#             "metrics": metrics,
#             "output_dir": str(output_dir),
#         }

#         # Import the train function here to avoid circular imports
#         from .core import train as train_model

#         results = train_model(config)

#         typer.echo(f"Training complete! Model saved to {output_dir}")
#         typer.echo(f"Final metrics: {results.get('metrics', {})}")

#     except Exception as e:
#         typer.echo(f"Error during training: {e!s}", err=True)
#         raise typer.Exit(code=1)
