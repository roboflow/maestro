import os
from glob import glob


def create_new_run_directory(base_output_dir: str) -> str:
    """Creates a new numbered directory for the current training run.

    Args:
        base_output_dir (str): The base directory where all run directories are stored.

    Returns:
        str: The path to the newly created run directory.
    """
    base_output_dir = os.path.abspath(base_output_dir)
    existing_run_dirs = [d for d in glob(os.path.join(base_output_dir, "*")) if os.path.isdir(d)]
    new_run_number = len(existing_run_dirs) + 1
    new_run_dir = os.path.join(base_output_dir, str(new_run_number))
    os.makedirs(new_run_dir, exist_ok=True)
    return new_run_dir
