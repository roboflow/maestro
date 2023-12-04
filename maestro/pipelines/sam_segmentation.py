from typing import Optional, Tuple, List

import numpy as np
import supervision as sv

from maestro.pipelines.base import BasePromptCreator, BaseResponseProcessor


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

    def process(self, text: str) -> List[str]:
        pass

    def extract(self, text: str, marks: sv.Detections) -> sv.Detections:
        pass

    def visualize(
        self,
        text: str,
        image: np.ndarray,
        marks: sv.Detections
    ) -> np.ndarray:
        pass
