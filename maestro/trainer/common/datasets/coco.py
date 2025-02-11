import os
from typing import ClassVar

from PIL import Image
from supervision.dataset.formats.coco import load_coco_annotations
from supervision.detection.core import Detections
from torch.utils.data import Dataset

from maestro.trainer.logger import get_maestro_logger

logger = get_maestro_logger()


class COCODataset(Dataset):
    REQUIRED_KEYS: ClassVar[list[str]] = ["images", "annotations", "categories"]

    def __init__(self, annotations_path: str, images_directory_path: str) -> None:
        self.images_directory_path = images_directory_path
        self.entries = self._load_entries(annotations_path, images_directory_path)

    @classmethod
    def _load_entries(cls, annotations_path: str, images_dir: str) -> list[tuple[str, Detections]]:
        if not os.path.isfile(annotations_path):
            logger.warning(f"Annotations file does not exist: '{annotations_path}'")
            return []

        try:
            classes, images, annotation_dict = load_coco_annotations(
                images_directory_path=images_dir, annotations_path=annotations_path, force_masks=False
            )
        except Exception as e:
            logger.warning(f"Could not parse annotations file '{annotations_path}': {e}")
            return []

        total_images = len(images)
        skipped_count = 0
        empty_detections_count = 0
        valid_entries: list[tuple[str, Detections]] = []

        for image_path in images:
            if not os.path.exists(image_path):
                skipped_count += 1
                logger.warning(f"Skipping file: image file not found '{image_path}'")
                continue

            detections = annotation_dict[image_path]
            if detections.xyxy.shape[0] == 0:
                empty_detections_count += 1

            valid_entries.append((image_path, detections))

        loaded_count = total_images - skipped_count
        if total_images > 0:
            logger.info(
                f"Loaded {loaded_count} valid entries out of {total_images} from '{annotations_path}'. "
                f"Skipped {skipped_count}. Found {empty_detections_count} entries with empty detections."
            )
        else:
            logger.warning(f"No images found in '{annotations_path}'.")

        return valid_entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[Image.Image, Detections]:
        if idx >= len(self.entries):
            raise IndexError(f"Index {idx} is out of range.")
        image_path, detections = self.entries[idx]
        image = Image.open(image_path).convert("RGB")
        return image, detections
