import re
from typing import Optional

import numpy as np


def result_to_detections_formatter(
    text: str, resolution_wh: tuple[int, int], classes: Optional[list[str]] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Converts SmolVLM_2 text output into detection format.

    SmolVLM_2 outputs text in a format like:
    "a person standing in front of a car [x1, y1, x2, y2]"

    Args:
        text: SmolVLM_2 output text
        resolution_wh: Target image resolution (width, height)
        classes: Optional list of valid class names

    Returns:
        Tuple of (boxes, class_ids) where:
        - boxes is a float32 array of shape (N, 4) with xyxy coordinates
        - class_ids is an int32 array of shape (N,) with class IDs
    """
    # Extract bounding boxes using regex
    box_pattern = r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
    matches = re.finditer(box_pattern, text)

    boxes_list = []
    class_ids_list = []

    # Create class mapping if provided
    if classes is not None:
        name_to_index = {cls_name: idx for idx, cls_name in enumerate(classes)}
    else:
        name_to_index = None

    for match in matches:
        x_min, y_min, x_max, y_max = map(float, match.groups())

        # Extract class name from text before the box
        text_before = text[: match.start()].strip()
        class_name = text_before.split()[-1] if text_before else "unknown"

        if name_to_index is not None:
            if class_name not in name_to_index:
                continue
            current_class_id = name_to_index[class_name]
        else:
            current_class_id = -1

        boxes_list.append([x_min, y_min, x_max, y_max])
        class_ids_list.append(current_class_id)

    boxes = np.array(boxes_list, dtype=np.float32).reshape(-1, 4)
    class_ids = np.array(class_ids_list, dtype=np.int32)

    return boxes, class_ids


def detections_to_text_formatter(
    xyxy: np.ndarray, class_id: np.ndarray, classes: list[str], resolution_wh: tuple[int, int]
) -> str:
    """Converts detections to SmolVLM_2 text format.

    Args:
        xyxy: Bounding boxes in xyxy format
        class_id: Class IDs for each box
        classes: List of class names
        resolution_wh: Image resolution (width, height)

    Returns:
        Formatted text string for SmolVLM_2
    """
    text_parts = []

    for i in range(len(xyxy)):
        cls_name = classes[class_id[i]]
        x_min, y_min, x_max, y_max = map(int, xyxy[i])
        box_text = f"{cls_name} [{x_min}, {y_min}, {x_max}, {y_max}]"
        text_parts.append(box_text)

    return " ".join(text_parts)


def format_prompt_for_detection(
    prompt: str,
    xyxy: Optional[np.ndarray] = None,
    class_id: Optional[np.ndarray] = None,
    classes: Optional[list[str]] = None,
    resolution_wh: Optional[tuple[int, int]] = None,
) -> str:
    """Formats a prompt for object detection with SmolVLM_2.

    Args:
        prompt: Base prompt
        xyxy: Optional bounding boxes
        class_id: Optional class IDs
        classes: Optional class names        resolution_wh: Optional image resolution

    Returns:
        Formatted prompt string
    """
    if all(x is not None for x in [xyxy, class_id, classes, resolution_wh]):
        # Type-cast to the expected types before passing to formatter
        detection_text = detections_to_text_formatter(
            xyxy,
            class_id if class_id is not None else [],
            classes if classes is not None else [],
            resolution_wh if resolution_wh is not None else (0, 0),
        )
        return f"{prompt} {detection_text}"
    return prompt
