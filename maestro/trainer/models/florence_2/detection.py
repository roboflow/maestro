import re

import numpy as np

PATTERN = re.compile(r"(\w+(?:\s+\w+)*)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>")


def result_to_detections_formatter(
    text: str, resolution_wh: tuple[int, int], classes: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Converts Florence-2-compatible text describing bounding boxes into NumPy arrays.

    Each bounding box is represented by:
    <class_name><loc_x_min><loc_y_min><loc_x_max><loc_y_max>
    with coordinates in the [0..1000] range.

    These coordinates are normalized by dividing by 1000, then scaled to the
    width/height specified by `resolution_wh`. If `classes` is provided, class names
    map to their IDs based on the order in `classes` and any class names not found in
    `classes` are skipped. If `classes` is not provided, all detected boxes are included
    with a default class ID of -1.

    Args:
        text (str): Florence-2-compatible text string with bounding box data.
        resolution_wh (tuple[int, int]): A (width, height) representing the target image resolution in pixels.
        classes (list[str] | None): A list of valid class names where the index of each class corresponds
            to its class ID. If None, each detection is assigned a class ID of -1.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            A tuple where:
            - The first element is a float32 array of shape (N, 4), containing xyxy bounding boxes scaled to
                `resolution_wh`.
            - The second element is an int32 array of shape (N,), containing the class IDs corresponding to each
                bounding box.
    """
    if classes is not None:
        name_to_index = {cls_name: idx for idx, cls_name in enumerate(classes)}
    else:
        name_to_index = None

    matches = PATTERN.finditer(text)
    boxes_list = []
    class_ids_list = []

    for match in matches:
        class_name, x_min, y_min, x_max, y_max = match.groups()
        if name_to_index is not None:
            if class_name not in name_to_index:
                continue
            current_class_id = name_to_index[class_name]
        else:
            current_class_id = -1

        x_min = float(x_min)
        y_min = float(y_min)
        x_max = float(x_max)
        y_max = float(y_max)

        boxes_list.append([x_min, y_min, x_max, y_max])
        class_ids_list.append(current_class_id)

    boxes = np.array(boxes_list, dtype=np.float32).reshape(-1, 4)

    if boxes.shape[0] > 0:
        boxes /= 1000.0
        boxes[:, 0::2] *= resolution_wh[0]
        boxes[:, 1::2] *= resolution_wh[1]

    class_ids = np.array(class_ids_list, dtype=np.int32)

    return boxes, class_ids


def detections_to_suffix_formatter(
    xyxy: np.ndarray, class_id: np.ndarray, classes: list[str], resolution_wh: tuple[int, int]
) -> str:
    """Generates Florence-2-compatible text describing bounding boxes.

    Each bounding box is assumed to be in pixel coordinates (xyxy) scaled
    to `resolution_wh`. Coordinates are normalized to [0..1], then scaled
    to [0..1000] and rounded to integers. The text format is:
    <class_name><loc_x_min><loc_y_min><loc_x_max><loc_y_max>

    Args:
        xyxy (np.ndarray): A float32 NumPy array of shape (N, 4) containing
            bounding boxes in xyxy format, scaled to `resolution_wh`.
        class_id (np.ndarray): An int32 array of shape (N,) with class IDs
            corresponding to each bounding box.
        classes (list[str]): A list of valid class names where the index
            corresponds to the class ID.
        resolution_wh (tuple[int, int]): (width, height) of the target image
            in pixels.

    Returns:
        str: A single string containing all bounding boxes formatted for
        Florence-2 compatibility.
    """
    width, height = resolution_wh
    text_parts = []

    for i in range(len(xyxy)):
        cls_name = classes[class_id[i]]
        x_min, y_min, x_max, y_max = xyxy[i]
        x_min /= width
        x_max /= width
        y_min /= height
        y_max /= height
        x_min_int = int(round(x_min * 1000))
        x_max_int = int(round(x_max * 1000))
        y_min_int = int(round(y_min * 1000))
        y_max_int = int(round(y_max * 1000))
        box_text = f"{cls_name}<loc_{x_min_int}><loc_{y_min_int}><loc_{x_max_int}><loc_{y_max_int}>"
        text_parts.append(box_text)

    return "".join(text_parts)


def detections_to_prefix_formatter(
    xyxy: np.ndarray, class_id: np.ndarray, classes: list[str], resolution_wh: tuple[int, int]
) -> str:
    return "<OD>"
