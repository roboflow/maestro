from pathlib import Path
from typing import Optional, Union

import torch
import typer

from .inference import SmolVLM2Inference

smolvlm2_app = typer.Typer()

class SmolVLM2:
    """Main entrypoint for SmolVLM2 model."""

    def __init__(
        self,
        model_name: str = "smol-ai/smolvlm2-500m",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        """Initialize SmolVLM2 model."""
        self.inference = SmolVLM2Inference(model_name=model_name, device=device, **kwargs)

    def generate(
        self,
        images: Union[str, list[str]],
        prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        **kwargs
    ) -> dict:
        """
        Generate text from images.

        Args:
            images: Path(s) to image(s)
            prompt: Optional prompt to guide generation
            max_new_tokens: Maximum number of tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            Dictionary containing generated text and other outputs
        """
        return self.inference.generate(
            images=images,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            **kwargs
        )

@smolvlm2_app.command(name="info", help="Get information about the SmolVLM2 model")
def info() -> None:
    """Get information about the SmolVLM2 model."""
    try:
        model = SmolVLM2()
        info = model.inference.get_model_info()
        typer.echo(f"Model Name: {info['model_name']}")
        typer.echo(f"Model Size: {info['model_size']}")
        typer.echo(f"Device: {info['device']}")
        typer.echo(f"Tokenizer: {info['tokenizer']}")
    except Exception as e:
        typer.echo(f"Error retrieving model info: {e!s}", err=True)
        raise typer.Exit(code=1)

@smolvlm2_app.command(name="predict", help="Run inference on one or more images")
def predict(
    image: list[Path] = typer.Option(
        ..., "--image", "-i", help="Path to image(s) for prediction"
    ),
    prompt: Optional[str] = typer.Option(
        None, "--prompt", "-p", help="Optional prompt to guide generation"
    ),
    max_new_tokens: int = typer.Option(
        512, "--max-new-tokens", help="Maximum new tokens to generate"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file path to save results"
    ),
) -> None:
    """Run inference on images using SmolVLM2."""
    try:
        model = SmolVLM2()
        result = model.generate(
            images=[str(img) for img in image],
            prompt=prompt,
            max_new_tokens=max_new_tokens
        )

        if output:
            import json
            with open(output, "w") as f:
                json.dump(result, f, indent=2)
            typer.echo(f"Results saved to {output}")
        else:
            typer.echo(f"Generated text: {result['text']}")

    except Exception as e:
        typer.echo(f"Error during prediction: {e!s}", err=True)
        raise typer.Exit(code=1)

@smolvlm2_app.command(name="train", help="Fine-tune the SmolVLM2 model")
def train(
    dataset: Path = typer.Option(
        ..., "--dataset", "-d", help="Path to dataset directory or file"
    ),
    epochs: int = typer.Option(
        10, "--epochs", "-e", help="Number of training epochs"
    ),
    batch_size: int = typer.Option(
        4, "--batch-size", "-b", help="Training batch size"
    ),
    optimization_strategy: str = typer.Option(
        "qlora", "--optimization-strategy", "-o",
        help="Optimization strategy (qlora, lora, freeze_vision)"
    ),
    metrics: list[str] = typer.Option(
        ["edit_distance"], "--metrics", "-m", help="Metrics to evaluate during training"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", help="Directory to save trained model"
    ),
) -> None:
    """Fine-tune the SmolVLM2 model on a dataset."""
    try:
        typer.echo("Starting SmolVLM2 fine-tuning...")

        if output_dir is None:
            import tempfile
            output_dir = Path(tempfile.mkdtemp())
            typer.echo(f"No output directory specified, using temporary directory: {output_dir}")

        # Create configuration for training
        config = {
            "dataset": str(dataset),
            "epochs": epochs,
            "batch_size": batch_size,
            "optimization_strategy": optimization_strategy,
            "metrics": metrics,
            "output_dir": str(output_dir)
        }

        # Import the train function here to avoid circular imports
        from .core import train as train_model

        results = train_model(config)

        typer.echo(f"Training complete! Model saved to {output_dir}")
        typer.echo(f"Final metrics: {results.get('metrics', {})}")

    except Exception as e:
        typer.echo(f"Error during training: {e!s}", err=True)
        raise typer.Exit(code=1)
