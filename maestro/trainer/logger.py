import logging
import os

from transformers.utils.logging import disable_progress_bar, enable_progress_bar


def configure_logging() -> None:
    """Configure global logging settings for PyTorch Lightning and Transformers.

    Sets up logging based on environment variables:
        MAESTRO_LIGHTNING_LOG_LEVEL: Sets Lightning's logging level (default: "INFO")
        MAESTRO_TRANSFORMERS_PROGRESS: Controls Transformers progress bar (1=enabled, 0=disabled)

    Example:
        import os
        from maestro.trainer import configure_logging
        os.environ["MAESTRO_LIGHTNING_LOG_LEVEL"] = "DEBUG"
        os.environ["MAESTRO_TRANSFORMERS_PROGRESS"] = "1"
        configure_logging()
    """
    lightning_level = os.getenv("MAESTRO_LIGHTNING_LOG_LEVEL", "INFO")
    set_lightning_logging(lightning_level)

    if os.getenv("MAESTRO_TRANSFORMERS_PROGRESS", "") == "1":
        set_transformers_progress(True)
    else:
        set_transformers_progress(False)


def set_lightning_logging(level: str) -> None:
    """Set PyTorch Lightning logging level while preserving transformers state.

    Args:
        level (str): Logging level (e.g., "INFO", "DEBUG", "WARNING", "ERROR")

    Example:
        from maestro.trainer import set_lightning_logging
        set_lightning_logging("DEBUG")
    """
    pytorch_lightning_logging = logging.getLogger("pytorch_lightning")
    cuda_log = logging.getLogger("lightning.pytorch.accelerators.cuda")
    rank_zero = logging.getLogger("lightning.pytorch.utilities.rank_zero")

    pytorch_lightning_logging.setLevel(getattr(logging, level))
    cuda_log.setLevel(getattr(logging, level))
    rank_zero.setLevel(getattr(logging, level))


def set_transformers_progress(status: bool) -> None:
    """Control visibility of Transformers progress bars.

    Args:
        status (bool): True to enable progress bars, False to disable

    Example:
        from maestro.trainer import set_transformers_progress
        set_transformers_progress(True)  # Enable progress bars
        set_transformers_progress(False)  # Disable progress bars
    """
    if status:
        enable_progress_bar()
    else:
        disable_progress_bar()
