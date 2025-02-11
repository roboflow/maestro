from contextlib import ExitStack

import numpy as np
import pytest

from maestro.trainer.models.florence_2.detection import boxes_to_text, text_to_boxes


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
            "catloc_100<loc_200><loc__400><loc_300>",
            ["cat", "dog"],
            (640, 480),
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            ExitStack(),
        ),
    ],
)
def test_text_to_boxes(
    text: str,
    classes: list[str],
    resolution_wh: tuple[int, int],
    expected_boxes: np.ndarray,
    expected_class_ids: np.ndarray,
    exception: ExitStack,
) -> None:
    with exception:
        boxes, class_ids = text_to_boxes(text, classes, resolution_wh)
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
def test_boxes_to_text(
    xyxy: np.ndarray,
    class_id: np.ndarray,
    classes: list[str],
    resolution_wh: tuple[int, int],
    expected_text: str,
    exception: ExitStack,
) -> None:
    with exception:
        result = boxes_to_text(xyxy, class_id, classes, resolution_wh)
        if expected_text is not None:
            assert result == expected_text
