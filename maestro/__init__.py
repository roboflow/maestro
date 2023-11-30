import importlib.metadata as importlib_metadata

try:
    # This will read version from pyproject.toml
    __version__ = importlib_metadata.version(__package__ or __name__)
except importlib_metadata.PackageNotFoundError:
    __version__ = "development"

from maestro.lmms.gpt4 import prompt_image
from maestro.markers.sam import SegmentAnythingMarkGenerator
from maestro.postprocessing.mask import (
    compute_mask_iou_vectorized,
    mask_non_max_suppression,
    filter_masks_by_relative_area,
    adjust_mask_features_by_relative_area,
    FeatureType,
    masks_to_marks,
    refine_marks
)
from maestro.postprocessing.text import (
    extract_marks_in_brackets,
    extract_relevant_masks
)
from maestro.visualizers import MarkVisualizer
from maestro.primitives import MarkMode
