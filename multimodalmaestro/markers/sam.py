import cv2
import numpy as np
import supervision as sv
from PIL import Image
from transformers import pipeline, SamModel, SamProcessor, SamImageProcessor
from typing import Union

from multimodalmaestro.postprocessing.mask import masks_to_marks


class SegmentAnythingMarkGenerator:
    """
    A class for performing image segmentation using a specified model.

    Parameters:
        device (str): The device to run the model on (e.g., 'cpu', 'cuda').
        model_name (str): The name of the model to be loaded. Defaults to
                          'facebook/sam-vit-huge'.
    """
    def __init__(self, device: str = 'cpu', model_name: str = "facebook/sam-vit-huge"):
        self.model = SamModel.from_pretrained(model_name).to(device)
        self.processor = SamProcessor.from_pretrained(model_name)
        self.image_processor = SamImageProcessor.from_pretrained(model_name)
        self.pipeline = pipeline(
            task="mask-generation",
            model=self.model,
            image_processor=self.image_processor,
            device=device)

    def generate(self, image: np.ndarray) -> sv.Detections:
        """
        Generate image segmentation marks.

        Parameters:
            image (np.ndarray): The image to be marked in BGR format.

        Returns:
            sv.Detections: An object containing the segmentation masks and their
                corresponding bounding box coordinates.
        """
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        outputs = self.pipeline(image, points_per_batch=64)
        masks = np.array(outputs['masks'])
        return masks_to_marks(masks=masks)
