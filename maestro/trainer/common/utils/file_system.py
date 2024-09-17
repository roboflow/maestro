import json
import os
from glob import glob
from typing import Union


def read_jsonl(path: str) -> list[dict]:
    file_lines = read_file(
        path=path,
        split_lines=True,
    )
    return [json.loads(line) for line in file_lines]


def read_file(
    path: str,
    split_lines: bool = False,
    strip_white_spaces: bool = False,
    line_separator: str = "\n",
) -> Union[str, list[str]]:
    with open(path) as f:
        file_content = f.read()
    if strip_white_spaces:
        file_content = file_content.strip()
    if not split_lines:
        return file_content
    lines = file_content.split(line_separator)
    if not strip_white_spaces:
        return lines
    return [line.strip() for line in lines]


def save_json(path: str, content: dict) -> None:
    ensure_parent_dir_exists(path=path)
    with open(path, "w") as f:
        json.dump(content, f, indent=4)


def ensure_parent_dir_exists(path: str) -> None:
    parent_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent_dir, exist_ok=True)


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
