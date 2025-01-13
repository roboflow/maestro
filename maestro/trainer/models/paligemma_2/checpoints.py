import os
import torch

from maestro.trainer.common.configuration.env import CUDA_DEVICE_ENV, DEFAULT_CUDA_DEVICE

DEFAULT_PALIGEMMA2_MODEL_ID ="google/paligemma2-3b-pt-448"
DEVICE = torch.device("cpu") if not torch.cuda.is_available() else os.getenv(CUDA_DEVICE_ENV, DEFAULT_CUDA_DEVICE)