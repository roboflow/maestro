import os
import shutil
from typing import Callable

import lightning
from lightning.pytorch.callbacks import Callback

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
