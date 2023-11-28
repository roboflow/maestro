import numpy as np
import supervision as sv


class Visualizer:

    def __init__(
        self,
        line_thickness: int = 2,
        mask_opacity: float = 0.1,
        text_scale: float = 0.6
    ) -> None:
        self.box_annotator = sv.BoundingBoxAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            thickness=line_thickness)
        self.mask_annotator = sv.MaskAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            opacity=mask_opacity)
        self.polygon_annotator = sv.PolygonAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            thickness=line_thickness)
        self.label_annotator = sv.LabelAnnotator(
            color=sv.Color.black(),
            text_color=sv.Color.white(),
            color_lookup=sv.ColorLookup.INDEX,
            text_position=sv.Position.CENTER_OF_MASS,
            text_scale=text_scale)

    def visualize(
        self,
        image: np.ndarray,
        detections: sv.Detections,
        with_box: bool,
        with_mask: bool,
        with_polygon: bool,
        with_label: bool
    ) -> np.ndarray:
        annotated_image = image.copy()
        if with_box:
            annotated_image = self.box_annotator.annotate(
                scene=annotated_image, detections=detections)
        if with_mask:
            annotated_image = self.mask_annotator.annotate(
                scene=annotated_image, detections=detections)
        if with_polygon:
            annotated_image = self.polygon_annotator.annotate(
                scene=annotated_image, detections=detections)
        if with_label:
            labels = list(map(str, range(len(detections))))
            annotated_image = self.label_annotator.annotate(
                scene=annotated_image, detections=detections, labels=labels)
        return annotated_image
