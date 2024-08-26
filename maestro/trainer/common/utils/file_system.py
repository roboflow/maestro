import json
from typing import Union, List


def read_jsonl(path: str) -> List[dict]:
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
) -> Union[str, List[str]]:
    with open(path, "r") as f:
        file_content = f.read()
    if strip_white_spaces:
        file_content = file_content.strip()
    if not split_lines:
        return file_content
    lines = file_content.split(line_separator)
    if not strip_white_spaces:
        return lines
    return [line.strip() for line in lines]
