from typing import Callable

import lightning as L
from lightning.pytorch.callbacks import Callback

from maestro.trainer.common.training import MaestroTrainer, TProcessor, TModel


class SaveCheckpoint(Callback):
    def __init__(self, result_path: str, save_model_callback: Callable[[str, TProcessor, TModel], None]):
        self.result_path = result_path
        self.save_model_callback = save_model_callback
        self.epoch = 0

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: MaestroTrainer):
        checkpoint_path = f"{self.result_path}/{self.epoch}"
        self.save_model_callback(checkpoint_path, pl_module.processor, pl_module.model)
        print(f"Saved checkpoint to {checkpoint_path}")
        self.epoch += 1

    def on_train_end(self, trainer: L.Trainer, pl_module: MaestroTrainer):
        checkpoint_path = f"{self.result_path}/latest"
        self.save_model_callback(checkpoint_path, pl_module.processor, pl_module.model)
        print(f"Saved checkpoint to {checkpoint_path}")
