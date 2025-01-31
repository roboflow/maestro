from typing import TypeVar

import lightning
from torch.utils.data import DataLoader
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin

TProcessor = TypeVar("TProcessor", bound=ProcessorMixin)
TModel = TypeVar("TModel", bound=PreTrainedModel)


class MaestroTrainer(lightning.LightningModule):
    def __init__(self, processor: TProcessor, model: TModel, train_loader: DataLoader, valid_loader: DataLoader):
        super().__init__()
        self.processor = processor
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader
