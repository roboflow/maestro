import numpy as np

import supervision as sv

from transformers import SamModel, pipeline, SamProcessor


class SegmentAnythingMarker:

    def __init__(self, device: str, model_name: str = "facebook/sam-vit-huge"):
        self.model = SamModel.from_pretrained(model_name)
        self.image_processor = SamProcessor.from_pretrained(model_name)
        self.pipeline = pipeline(
            task='mask-generation',
            model=self.model,
            image_processor=self.image_processor)

    def mark(self) -> sv.Detections:
        pass

    def guided_mark(self, mask: np.ndarray) -> sv.Detections:
        pass
