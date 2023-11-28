from typing import Union

import numpy as np
import supervision as sv
from PIL import Image
from transformers import pipeline, SamModel, SamProcessor


class SegmentAnythingMarker:

    def __init__(self, device: str, model_name: str = "facebook/sam-vit-huge"):
        self.model = SamModel.from_pretrained(model_name).to(device)
        self.image_processor = SamProcessor.from_pretrained(model_name).to(device)
        self.pipeline = pipeline(
            task='mask-generation',
            model=self.model,
            image_processor=self.image_processor)

    def mark(self, image: Union[np.ndarray, Image]) -> sv.Detections:
        # image in BGR
        pass

    def guided_mark(
        self,
        image: Union[np.ndarray, Image],
        mask: np.ndarray
    ) -> sv.Detections:
        pass
