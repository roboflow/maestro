import numpy as np
from qwen_vl_utils import smart_resize

# Qwen2.5-VL's vision encoder uses a 14px patch with a 2x2 spatial merge, so the
# resized side lengths must be multiples of 14 * 2. qwen-vl-utils dropped the default
# for `smart_resize(factor=...)` in 0.0.13; passing it explicitly works on every version.
QWEN_2_5_VL_IMAGE_FACTOR = 28


def detections_to_suffix_formatter(
    xyxy: np.ndarray,
    class_id: np.ndarray,
    classes: list[str],
    resolution_wh: tuple[int, int],
    min_pixels: int,
    max_pixels: int,
    image_factor: int = QWEN_2_5_VL_IMAGE_FACTOR,
) -> str:
    """Formats detections as the JSON suffix Qwen2.5-VL is trained to emit.

    Boxes are rescaled from the source image resolution to the resolution the
    processor will actually feed the model, so the coordinates in the training
    target match what the model sees.

    Args:
        xyxy (np.ndarray): Boxes in `(N, 4)` `xyxy` format, in source-image pixels.
        class_id (np.ndarray): Class index per box, shape `(N,)`.
        classes (list[str]): Class names, indexed by `class_id`.
        resolution_wh (tuple[int, int]): Source image `(width, height)`.
        min_pixels (int): Lower bound on the resized pixel count.
        max_pixels (int): Upper bound on the resized pixel count.
        image_factor (int): Side lengths are rounded to a multiple of this value.
            Defaults to the Qwen2.5-VL vision encoder's patch size times its
            spatial merge size.

    Returns:
        str: A fenced JSON block of `bbox_2d` / `label` objects.
    """
    image_w, image_h = resolution_wh
    input_h, input_w = smart_resize(
        height=image_h, width=image_w, factor=image_factor, min_pixels=min_pixels, max_pixels=max_pixels
    )

    xyxy = xyxy / [image_w, image_h, image_w, image_h]
    xyxy = xyxy * [input_w, input_h, input_w, input_h]
    xyxy = xyxy.astype(int)

    detection_lines = []
    for cid, box in zip(class_id, xyxy):
        label = classes[int(cid)]
        bbox_str = ", ".join(str(num) for num in box.tolist())
        line = f'\t{{"bbox_2d": [{bbox_str}], "label": "{label}"}}'
        detection_lines.append(line)

    joined_detections = ",\n".join(detection_lines)
    formatted_str = f"```json\n[\n{joined_detections}\n]\n```"
    return formatted_str


def detections_to_prefix_formatter(
    xyxy: np.ndarray, class_id: np.ndarray, classes: list[str], resolution_wh: tuple[int, int]
) -> str:
    return "Outline the position of " + ", ".join(classes) + ". Output all the coordinates in JSON format."
