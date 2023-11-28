import re
from typing import List, Dict

import numpy as np
import supervision as sv

from multimodalmaestro.primitives import MarkMode


def extract_marks_in_brackets(text: str, mode: MarkMode) -> List[str]:
    """
    Extracts all unique marks enclosed in square brackets from a given string, based
        on the specified mode. Duplicates are removed and the results are sorted in
        descending order.

    Args:
        text (str): The string to be searched.
        mode (MarkMode): The mode to determine the type of marks to extract (NUMERIC or
            ALPHABETIC).

    Returns:
        List[str]: A list of unique marks found within square brackets, sorted in
            descending order.
    """
    if mode == MarkMode.NUMERIC:
        pattern = r'\[(\d+)\]'
    elif mode == MarkMode.ALPHABETIC:
        pattern = r'\[([A-Za-z]+)\]'
    else:
        raise ValueError(f"Unknown mode: {mode}")

    found_marks = re.findall(pattern, text)
    unique_marks = set(found_marks)

    if mode == MarkMode.NUMERIC:
        return sorted(unique_marks, key=int, reverse=False)
    else:
        return sorted(unique_marks, reverse=False)


def extract_relevant_masks(
    text: str,
    detections: sv.Detections
) -> Dict[str, np.ndarray]:
    """
    Extracts relevant masks from the detections based on marks found in the given text.

    Args:
        text (str): The string containing marks in square brackets to be searched for.
        detections (sv.Detections): An object containing detection information,
            including masks indexed by numeric identifiers.

    Returns:
        Dict[str, np.ndarray]: A dictionary where each key is a mark found in the text,
            and each value is the corresponding mask from detections.
    """
    marks = extract_marks_in_brackets(text=text, mode=MarkMode.NUMERIC)
    return {
        mark: detections.mask[int(mark)]
        for mark
        in marks
    }
