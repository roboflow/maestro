"""
Example script demonstrating how to enable early stopping in Maestro models.
This is useful to prevent overfitting and reduce training time when model
performance on the validation set has stopped improving.
"""

from maestro.trainer.models.florence_2.core import Florence2Configuration
from maestro.trainer.models.florence_2.core import train as train_florence
from maestro.trainer.models.paligemma_2.core import PaliGemma2Configuration
from maestro.trainer.models.paligemma_2.core import train as train_paligemma
from maestro.trainer.models.qwen_2_5_vl.core import Qwen25VLConfiguration
from maestro.trainer.models.qwen_2_5_vl.core import train as train_qwen


# Example with Florence-2 model
def train_florence_with_early_stopping():
    """Train a Florence-2 model with early stopping enabled"""
    config = Florence2Configuration(
        dataset="path/to/your/dataset",  # Replace with your dataset path
        epochs=20,  # Set a larger number of epochs
        early_stopping=True,  # Enable early stopping
        early_stopping_patience=3,  # Stop after 3 epochs without improvement
        early_stopping_threshold=0.01,  # Minimum change to be considered as improvement
        early_stopping_monitor="val_loss",  # Metric to monitor (default: val_loss)
    )

    train_florence(config)


# Example with PaliGemma-2 model
def train_paligemma_with_early_stopping():
    """Train a PaliGemma-2 model with early stopping enabled"""
    config = PaliGemma2Configuration(
        dataset="path/to/your/dataset",  # Replace with your dataset path
        epochs=20,  # Set a larger number of epochs
        early_stopping=True,  # Enable early stopping
        early_stopping_patience=5,  # Stop after 5 epochs without improvement
        early_stopping_threshold=0.001,  # More sensitive to small improvements
        early_stopping_monitor="val_loss",  # Metric to monitor
    )

    train_paligemma(config)


# Example with Qwen2.5-VL model
def train_qwen_with_early_stopping():
    """Train a Qwen2.5-VL model with early stopping enabled"""
    config = Qwen25VLConfiguration(
        dataset="path/to/your/dataset",  # Replace with your dataset path
        epochs=20,  # Set a larger number of epochs
        early_stopping=True,  # Enable early stopping
        early_stopping_patience=3,  # Stop after 3 epochs without improvement
        early_stopping_threshold=0.01,  # Minimum change to be considered as improvement
        early_stopping_monitor="val_loss",  # Metric to monitor
    )

    train_qwen(config)


if __name__ == "__main__":
    # Choose one of the training functions to run
    train_florence_with_early_stopping()
    # train_paligemma_with_early_stopping()
    # train_qwen_with_early_stopping()
