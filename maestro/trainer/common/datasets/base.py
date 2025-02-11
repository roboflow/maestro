from typing import Any, Protocol

import supervision as sv
from PIL import Image


class BaseDetectionDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> tuple[Image.Image, sv.Detections]: ...


class BaseVLDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> tuple[Image.Image, dict[str, Any]]: ...
