import random
from typing import Optional

import numpy as np
import torch


def ensure_reproducibility(
    seed: Optional[int] = None,
    disable_cudnn_benchmark: bool = True,
    avoid_non_deterministic_algorithms: bool = True,
) -> None:
    """
    Sets seeds and configuration options to improve experiment reproducibility.

    This function ensures that random number generation is controlled for
    Python's `random` module, NumPy, and PyTorch when a seed is provided.
    It also configures CUDA settings to reduce sources of non-determinism
    when using GPUs.

    Args:
        seed (Optional[int]):
            The random seed to use. If `None`, no seeding is applied, and
            the behavior remains stochastic.
        disable_cudnn_benchmark (bool):
            If `True`, disables cuDNN benchmarking. This can improve reproducibility
            by preventing cuDNN from selecting the fastest algorithm dynamically,
            which may introduce variability across runs.
        avoid_non_deterministic_algorithms (bool):
            If `True`, enforces deterministic algorithms in PyTorch by calling
            `torch.use_deterministic_algorithms(True)`. This helps ensure consistent
            results across runs but may impact performance by disabling certain
            optimizations.

    Returns:
        None
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)  # noqa: NPY002

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    if avoid_non_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)

    if disable_cudnn_benchmark:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
