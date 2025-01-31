import os
from glob import glob


def create_new_run_directory(base_output_dir: str) -> str:
    """Creates a new sequentially numbered run directory.

    This function ensures that a new directory is created inside `base_output_dir`,
    following a numeric naming convention (e.g., `1`, `2`, `3`, etc.). It scans
    existing directories in `base_output_dir`, determines the highest existing
    number, and creates a new directory with an incremented number.

    Args:
        base_output_dir (str):
            The base directory where the new run directory will be created.
            If it does not exist, it will be created.

    Returns:
        str: The absolute path to the newly created run directory.
    """
    base_output_dir = os.path.abspath(base_output_dir)
    os.makedirs(base_output_dir, exist_ok=True)

    existing_run_dirs = [d for d in glob(os.path.join(base_output_dir, "*")) if os.path.isdir(d)]
    existing_numbers = []
    for dir_path in existing_run_dirs:
        try:
            dir_name = os.path.basename(dir_path)
            existing_numbers.append(int(dir_name))
        except ValueError:
            continue

    new_run_number = max(existing_numbers, default=0) + 1
    new_run_dir = os.path.join(base_output_dir, str(new_run_number))
    os.makedirs(new_run_dir, exist_ok=True)
    return new_run_dir
