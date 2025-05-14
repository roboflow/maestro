import os
import shutil
from typing import Callable

import lightning
from lightning.pytorch.callbacks import Callback, EarlyStopping

from maestro.trainer.common.training import MaestroTrainer, TModel, TProcessor


class SaveCheckpoint(Callback):
    def __init__(self, result_path: str, save_model_callback: Callable[[str, TProcessor, TModel], None]):
        self.result_path = result_path
        self.save_model_callback = save_model_callback

    def on_train_epoch_end(self, trainer: lightning.Trainer, pl_module: MaestroTrainer):
        checkpoint_path = f"{self.result_path}/latest"
        if os.path.exists(checkpoint_path):
            shutil.rmtree(checkpoint_path)
        self.save_model_callback(checkpoint_path, pl_module.processor, pl_module.model)
        print(f"Saved latest checkpoint to {checkpoint_path}")

        # TODO: Get current metric value from trainer
        # TODO: Compare with best value and save if better
        # TODO: Save best model to {self.result_path}/best if metric improved

    def on_train_end(self, trainer: lightning.Trainer, pl_module: MaestroTrainer):
        pass


class EarlyStoppingCallback(EarlyStopping):
    """
    Early stopping callback for PyTorch Lightning trainers.

    This callback stops training when a monitored metric has stopped improving.

    Attributes:
        monitor (str): Quantity to be monitored. Default is 'val_loss'.
        min_delta (float): Minimum change in monitored quantity to qualify as improvement.
        patience (int): Number of validation epochs with no improvement after which training will be stopped.
        mode (str): One of 'min', 'max'. In 'min' mode, training will stop when the quantity monitored
                    has stopped decreasing; in 'max' mode it will stop when the quantity monitored
                    has stopped increasing. Default is 'min'.
        verbose (bool): Whether to print progress messages.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        min_delta: float = 0.0,
        patience: int = 3,
        verbose: bool = True,
        mode: str = "min",
    ):
        super().__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
        )
