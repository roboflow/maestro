from typing import Optional, Tuple, List, Callable

import re
import numpy as np
import supervision as sv

from maestro.pipelines.base import BasePromptCreator, BaseResponseProcessor


def extract_mark_ids(text: str) -> List[str]:
    """
    Extracts all unique marks enclosed in square brackets from a given string.
        Duplicates are removed and the results are sorted in descending order.

    Args:
        text (str): The string to be searched.

    Returns:
        List[str]: A list of unique marks found within square brackets, sorted in
            descending order.
    """
    pattern = r'\[(\d+)\]'
    found_marks = re.findall(pattern, text)
    unique_marks = set(found_marks)
    return sorted(unique_marks, key=int, reverse=False)


def default_annotate(image: np.ndarray, marks: sv.Detections) -> np.ndarray:
    h, w, _ = image.shape
    line_thickness = sv.calculate_dynamic_line_thickness(resolution_wh=(w, h))
    mask_annotator = sv.MaskAnnotator(
        color_lookup=sv.ColorLookup.INDEX, opacity=0.4)
    polygon_annotator = sv.PolygonAnnotator(
        color_lookup=sv.ColorLookup.INDEX, thickness=line_thickness)

    annotated_image = image.copy()
    annotated_image = mask_annotator.annotate(
        scene=annotated_image, detections=marks)
    return polygon_annotator.annotate(
        scene=annotated_image, detections=marks)


class SamPromptCreator(BasePromptCreator):
    def __init__(self, device: str):
        self.device = device

    def create(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, sv.Detections]:
        pass


class SamResponseProcessor(BaseResponseProcessor):

    def __init__(
        self,
        annotate: Callable[[np.ndarray, sv.Detections], np.ndarray] = default_annotate,
    ) -> None:
        self.annotate = annotate

    def process(self, text: str, marks: sv.Detections) -> sv.Detections:
        mark_ids = extract_mark_ids(text=text)
        mark_ids = np.array(mark_ids, dtype=int)
        return marks[mark_ids]

    def visualize(
        self,
        text: str,
        image: np.ndarray,
        marks: sv.Detections
    ) -> np.ndarray:
        marks = self.process(text=text, marks=marks)
        return self.annotate(image, marks)
