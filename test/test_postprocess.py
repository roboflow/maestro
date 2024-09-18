import pytest

from maestro.postprocessing.text import extract_marks_in_brackets
from maestro.primitives import MarkMode


@pytest.mark.parametrize(
    "text, mode, expected_result",
    [
        ("[1]", MarkMode.NUMERIC, ["1"]),
        ("lorem ipsum [1] dolor sit amet", MarkMode.NUMERIC, ["1"]),
        ("[1] lorem ipsum [2] dolor sit amet", MarkMode.NUMERIC, ["1", "2"]),
        ("[1] lorem ipsum [1] dolor sit amet", MarkMode.NUMERIC, ["1"]),
        ("[2] lorem ipsum [1] dolor sit amet", MarkMode.NUMERIC, ["1", "2"]),
        ("[A]", MarkMode.ALPHABETIC, ["A"]),
        ("lorem ipsum [A] dolor sit amet", MarkMode.ALPHABETIC, ["A"]),
        ("[A] lorem ipsum [B] dolor sit amet", MarkMode.ALPHABETIC, ["A", "B"]),
        ("[A] lorem ipsum [A] dolor sit amet", MarkMode.ALPHABETIC, ["A"]),
        ("[B] lorem ipsum [A] dolor sit amet", MarkMode.ALPHABETIC, ["A", "B"]),
        ("[1] lorem ipsum [A] dolor sit amet", MarkMode.NUMERIC, ["1"]),
        ("[1] lorem ipsum [A] dolor sit amet", MarkMode.ALPHABETIC, ["A"]),
    ],
)
def test_extract_marks_in_brackets(text: str, mode: MarkMode, expected_result: list[str]) -> None:
    result = extract_marks_in_brackets(text=text, mode=mode)
    assert result == expected_result
