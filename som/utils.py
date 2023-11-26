import re
from typing import List

from som.core import Mode


def extract_marks_in_brackets(text: str, mode: Mode) -> List[str]:
    """
    Extracts all marks enclosed in square brackets from a given string, based on the
    specified mode.

    Args:
        text (str): The string to be searched.
        mode (Mode): The mode to determine the type of marks to extract (NUMERIC or
            ALPHABETIC).

    Returns:
        List[str]: A list of marks found within square brackets.
    """
    if mode == Mode.NUMERIC:
        pattern = r'\[(\d+)\]'
    elif mode == Mode.ALPHABETIC:
        pattern = r'\[([A-Za-z]+)\]'

    marks = re.findall(pattern, text)
    return marks
