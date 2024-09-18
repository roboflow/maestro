import os
import random
from typing import Optional

import numpy as np
import torch

from maestro.trainer.common.configuration.env import DEFAULT_SEED, SEED_ENV


def make_it_reproducible(
    seed: Optional[int] = None,
    disable_cudnn_benchmark: bool = True,
    avoid_non_deterministic_algorithms: bool = True,
) -> None:
    if seed is None:
        seed = int(os.getenv(SEED_ENV, DEFAULT_SEED))
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.default_rng(seed)
    if avoid_non_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
    if disable_cudnn_benchmark:
        torch.backends.cudnn.benchmark = False
