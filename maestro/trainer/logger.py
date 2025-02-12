import logging
import os
import sys

from transformers.utils.logging import disable_progress_bar, enable_progress_bar


def get_maestro_logger(name: str = "maestro", level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured logger for the Maestro project.
    Adjust log levels, handlers, and formatters as needed.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        formatter = logging.Formatter("[%(levelname)s] %(name)s - %(message)s")
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.propagate = False

    return logger


def configure_logging() -> None:
    """Configure global logging settings for PyTorch Lightning and Transformers.

    Sets up logging based on environment variables:
        MAESTRO_LIGHTNING_LOG_LEVEL: Sets Lightning's logging level (default: "INFO")
        MAESTRO_TRANSFORMERS_PROGRESS: Controls Transformers progress bar (1=enabled, 0=disabled)

    Example:
        import os
        os.environ["MAESTRO_LIGHTNING_LOG_LEVEL"] = "DEBUG"
        os.environ["MAESTRO_TRANSFORMERS_PROGRESS"] = "1"
        from maestro.trainer import configure_logging
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

    lightning_logging = logging.getLogger("lightning")
    pytorch_lightning_logging = logging.getLogger("pytorch_lightning")
    cuda_logging = logging.getLogger("lightning.pytorch.accelerators.cuda")
    rank_zero_logging = logging.getLogger("lightning.pytorch.utilities.rank_zero")

    lightning_logging.setLevel(getattr(logging, level))
    pytorch_lightning_logging.setLevel(getattr(logging, level))
    cuda_logging.setLevel(getattr(logging, level))
    rank_zero_logging.setLevel(getattr(logging, level))


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
