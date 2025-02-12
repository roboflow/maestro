from contextlib import ExitStack

import numpy as np
import pytest

from maestro.trainer.models.florence_2.detection import detections_to_suffix_formatter, result_to_detections_formatter


@pytest.mark.parametrize(
    ("text", "classes", "resolution_wh", "expected_boxes", "expected_class_ids", "exception"),
    [
        # 1. Empty text -> no boxes, no class IDs
        (
            "",
            ["cat", "dog"],
            (640, 480),
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            ExitStack(),
        ),
        # 2. Single valid box -> scaled coordinates
        (
            "cat<loc_100><loc_200><loc_400><loc_300>",
            ["cat", "dog"],
            (640, 480),
            np.array([[64.0, 96.0, 256.0, 144.0]], dtype=np.float32),
            np.array([0], dtype=np.int32),
            ExitStack(),
        ),
        # 3. Mixed valid/invalid classes -> skip unknown
        (
            "cat<loc_0><loc_100><loc_200><loc_300>"
            "bird<loc_500><loc_500><loc_600><loc_600>"
            "dog<loc_50><loc_50><loc_1000><loc_1000>",
            ["cat", "dog"],
            (100, 200),
            np.array(
                [
                    [0.0, 20.0, 20.0, 60.0],
                    [5.0, 10.0, 100.0, 200.0],
                ],
                dtype=np.float32,
            ),
            np.array([0, 1], dtype=np.int32),
            ExitStack(),
        ),
        # 4. Boundary coordinates -> full image box
        (
            "dog<loc_0><loc_0><loc_1000><loc_1000>",
            ["cat", "dog"],
            (800, 600),
            np.array([[0.0, 0.0, 800.0, 600.0]], dtype=np.float32),
            np.array([1], dtype=np.int32),
            ExitStack(),
        ),
        # 5. Unknown classes -> empty output
        (
            "lion<loc_100><loc_200><loc_300><loc_400>tiger<loc_200><loc_300><loc_400><loc_500>",
            ["cat", "dog"],
            (640, 480),
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            ExitStack(),
        ),
        # 6. Malformatted string -> no matches
        (
            "cat<loc_200><loc_400><loc_300>",
            ["cat", "dog"],
            (640, 480),
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            ExitStack(),
        ),
        # 7. Some classes known, some unknown -> only valid boxes are returned
        (
            "cat<loc_100><loc_200><loc_400><loc_300>"
            "unknown<loc_10><loc_20><loc_50><loc_60>"
            "dog<loc_0><loc_0><loc_1000><loc_1000>",
            ["cat", "dog"],
            (640, 480),
            np.array(
                [
                    [64.0, 96.0, 256.0, 144.0],
                    [0.0, 0.0, 640.0, 480.0],
                ],
                dtype=np.float32,
            ),
            np.array([0, 1], dtype=np.int32),
            ExitStack(),
        ),
        # 8. Partially malformatted string -> one valid box returned
        (
            "cat<loc_100><loc_200><loc_400>dog<loc_0><loc_0><loc_1000><loc_1000>",
            ["cat", "dog"],
            (640, 480),
            np.array([[0.0, 0.0, 640.0, 480.0]], dtype=np.float32),
            np.array([1], dtype=np.int32),
            ExitStack(),
        ),
        # 9. Single valid box with no classes provided -> default class id -1
        (
            "cat<loc_100><loc_200><loc_400><loc_300>",
            None,
            (640, 480),
            np.array([[64.0, 96.0, 256.0, 144.0]], dtype=np.float32),
            np.array([-1], dtype=np.int32),
            ExitStack(),
        ),
        # 10. Multiple boxes with no classes provided -> all detections included with class id -1
        (
            "cat<loc_0><loc_100><loc_200><loc_300>"
            "bird<loc_500><loc_500><loc_600><loc_600>"
            "dog<loc_50><loc_50><loc_1000><loc_1000>",
            None,
            (100, 200),
            np.array(
                [
                    [0.0, 20.0, 20.0, 60.0],
                    [50.0, 100.0, 60.0, 120.0],
                    [5.0, 10.0, 100.0, 200.0],
                ],
                dtype=np.float32,
            ),
            np.array([-1, -1, -1], dtype=np.int32),
            ExitStack(),
        ),
    ],
)
def test_result_to_detections_formatter(
    text: str,
    classes: list[str] | None,
    resolution_wh: tuple[int, int],
    expected_boxes: np.ndarray,
    expected_class_ids: np.ndarray,
    exception: ExitStack,
) -> None:
    with exception:
        boxes, class_ids = result_to_detections_formatter(text, resolution_wh, classes)
        assert boxes.shape == expected_boxes.shape
        assert class_ids.shape == expected_class_ids.shape
        np.testing.assert_allclose(boxes, expected_boxes, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(class_ids, expected_class_ids)


@pytest.mark.parametrize(
    ("xyxy", "class_id", "classes", "resolution_wh", "expected_text", "exception"),
    [
        # 1. Empty arrays -> blank text
        (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            ["cat", "dog"],
            (200, 200),
            "",
            ExitStack(),
        ),
        # 2. Single box -> simple normalization
        (
            np.array([[50, 50, 100, 100]], dtype=np.float32),
            np.array([0], dtype=np.int32),
            ["cat", "dog"],
            (200, 200),
            "cat<loc_250><loc_250><loc_500><loc_500>",
            ExitStack(),
        ),
        # 3. Multiple boxes -> concatenated text
        (
            np.array(
                [
                    [0.0, 0.0, 50.0, 50.0],
                    [100.0, 100.0, 200.0, 200.0],
                ],
                dtype=np.float32,
            ),
            np.array([0, 1], dtype=np.int32),
            ["cat", "dog"],
            (200, 200),
            "cat<loc_0><loc_0><loc_250><loc_250>dog<loc_500><loc_500><loc_1000><loc_1000>",
            ExitStack(),
        ),
        # 4. Full boundary -> entire image
        (
            np.array([[0.0, 0.0, 200.0, 200.0]], dtype=np.float32),
            np.array([1], dtype=np.int32),
            ["cat", "dog"],
            (200, 200),
            "dog<loc_0><loc_0><loc_1000><loc_1000>",
            ExitStack(),
        ),
        # 5. Invalid class ID -> IndexError
        (
            np.array([[0.0, 0.0, 50.0, 50.0]], dtype=np.float32),
            np.array([2], dtype=np.int32),
            ["cat", "dog"],
            (200, 200),
            None,
            pytest.raises(IndexError),
        ),
    ],
)
def test_detections_to_suffix_formatter(
    xyxy: np.ndarray,
    class_id: np.ndarray,
    classes: list[str],
    resolution_wh: tuple[int, int],
    expected_text: str,
    exception: ExitStack,
) -> None:
    with exception:
        result = detections_to_suffix_formatter(xyxy, class_id, classes, resolution_wh)
        if expected_text is not None:
            assert result == expected_text
