import numpy as np
import pytest

# `qwen_vl_utils` ships with the optional `qwen_2_5_vl` extra. Without this guard the
# import below aborts collection for the whole run, taking unrelated suites with it.
pytest.importorskip("qwen_vl_utils", reason="requires the optional `qwen_2_5_vl` extra")

from maestro.trainer.models.qwen_2_5_vl.detection import (
    QWEN_2_5_VL_IMAGE_FACTOR,
    detections_to_suffix_formatter,
)

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28


@pytest.mark.parametrize(
    ("xyxy", "class_id", "classes", "resolution_wh", "image_factor", "expected"),
    [
        # 1. Single box on a 640x480 image. smart_resize gives (h=476, w=644), so the
        #    box is scaled by 644/640 horizontally and 476/480 vertically.
        (
            np.array([[10.0, 20.0, 110.0, 120.0]]),
            np.array([0]),
            ["cat", "dog"],
            (640, 480),
            QWEN_2_5_VL_IMAGE_FACTOR,
            '```json\n[\n\t{"bbox_2d": [10, 19, 110, 119], "label": "cat"}\n]\n```',
        ),
        # 2. Two boxes, second class -> both scaled, labels resolved by class_id.
        (
            np.array([[0.0, 0.0, 640.0, 480.0], [320.0, 240.0, 640.0, 480.0]]),
            np.array([0, 1]),
            ["cat", "dog"],
            (640, 480),
            QWEN_2_5_VL_IMAGE_FACTOR,
            '```json\n[\n\t{"bbox_2d": [0, 0, 644, 476], "label": "cat"},\n'
            '\t{"bbox_2d": [322, 238, 644, 476], "label": "dog"}\n]\n```',
        ),
        # 3. Larger source resolution -> (h=756, w=1036).
        (
            np.array([[0.0, 0.0, 1024.0, 768.0]]),
            np.array([0]),
            ["cat"],
            (1024, 768),
            QWEN_2_5_VL_IMAGE_FACTOR,
            '```json\n[\n\t{"bbox_2d": [0, 0, 1036, 756], "label": "cat"}\n]\n```',
        ),
        # 4. A non-default factor is honoured: 640x480 at factor 56 gives (h=504, w=616).
        (
            np.array([[0.0, 0.0, 640.0, 480.0]]),
            np.array([0]),
            ["cat"],
            (640, 480),
            56,
            '```json\n[\n\t{"bbox_2d": [0, 0, 616, 504], "label": "cat"}\n]\n```',
        ),
        # 5. No detections -> an empty JSON block rather than an error.
        (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            ["cat"],
            (640, 480),
            QWEN_2_5_VL_IMAGE_FACTOR,
            "```json\n[\n\n]\n```",
        ),
    ],
)
def test_detections_to_suffix_formatter(
    xyxy: np.ndarray,
    class_id: np.ndarray,
    classes: list[str],
    resolution_wh: tuple[int, int],
    image_factor: int,
    expected: str,
) -> None:
    result = detections_to_suffix_formatter(
        xyxy=xyxy,
        class_id=class_id,
        classes=classes,
        resolution_wh=resolution_wh,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        image_factor=image_factor,
    )
    assert result == expected


def test_resized_side_lengths_are_multiples_of_the_image_factor() -> None:
    """The whole point of `factor`: Qwen2.5-VL cannot patch a side it cannot divide."""
    result = detections_to_suffix_formatter(
        xyxy=np.array([[0.0, 0.0, 640.0, 480.0]]),
        class_id=np.array([0]),
        classes=["cat"],
        resolution_wh=(640, 480),
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    x1, y1, x2, y2 = (int(value) for value in result.split("[")[2].split("]")[0].split(","))
    assert x2 % QWEN_2_5_VL_IMAGE_FACTOR == 0
    assert y2 % QWEN_2_5_VL_IMAGE_FACTOR == 0
