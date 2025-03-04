import os
from enum import Enum
from typing import Optional

import torch
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoProcessor, GenerationConfig

from maestro.trainer.common.utils.device import parse_device_spec

DEFAULT_QWEN2_5_VL_MODEL_ID = "microsoft/Phi-4-multimodal-instruct"
DEFAULT_QWEN2_5_VL_MODEL_REVISION = "refs/heads/main"