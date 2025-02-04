from .logger import configure_logging, set_lightning_logging, set_transformers_progress

# Configure default logging settings on import
configure_logging()

__all__ = ["set_lightning_logging", "set_transformers_progress"]
