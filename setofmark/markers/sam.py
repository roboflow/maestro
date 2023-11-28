from typing import Union

import numpy as np
import supervision as sv
from PIL import Image
from transformers import pipeline, SamModel, SamProcessor, SamImageProcessor


class SegmentAnythingMarker:
    """
    A class for performing image segmentation using a specified model.

    Parameters:
        device (str): The device to run the model on (e.g., 'cpu', 'cuda').
        model_name (str): The name of the model to be loaded. Defaults to
                          'facebook/sam-vit-huge'.
    """
    def __init__(self, device: str, model_name: str = "facebook/sam-vit-huge"):
        self.model = SamModel.from_pretrained(model_name).to(device)
        self.processor = SamProcessor.from_pretrained(model_name)
        self.image_processor = SamImageProcessor.from_pretrained(model_name)
        self.pipeline = pipeline(
            task="mask-generation",
            model=self.model,
            image_processor=self.image_processor,
            device=device)

    def mark(self, image: Union[np.ndarray, Image.Image]) -> sv.Detections:
        """
        Marks an image for segmentation.

        Parameters:
            image (Union[np.ndarray, Image]): The image to be marked. Can be a
                numpy array or a PIL image.

        Returns:
            sv.Detections: An object containing the segmentation masks and their
                corresponding bounding box coordinates.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        outputs = self.pipeline(image, points_per_batch=64)
        masks = np.array(outputs['masks'])
        return sv.Detections(
            mask=masks,
            xyxy=sv.mask_to_xyxy(masks=masks)
        )

    # def guided_mark(
    #     self,
    #     image: Union[np.ndarray, Image],
    #     mask: np.ndarray
    # ) -> sv.Detections:
    #     """
    #     Performs guided marking on an image for segmentation.
    #
    #     Returns:
    #         sv.Detections: An object containing the segmentation masks and their
    #             corresponding bounding box coordinates.
    #     """
    #     masks = []
    #     for polygon in sv.mask_to_polygons(mask.astype(bool)):
    #         random_point_indexes = np.random.choice(
    #             polygon.shape[0],
    #             size=5,
    #             replace=True)
    #         input_point = polygon[random_point_indexes]
    #         input_label = np.ones(5)
    #         mask = predictor.predict(
    #             point_coords=input_point,
    #             point_labels=input_label,
    #             multimask_output=False,
    #         )[0][0]
    #         masks.append(mask)
    #     masks = np.array(masks, dtype=bool)
    #     return sv.Detections(
    #         xyxy=sv.mask_to_xyxy(masks),
    #         mask=masks
    #     )
