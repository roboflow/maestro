from enum import Enum

import cv2
import numpy as np


class FeatureType(Enum):
    """
    An enumeration to represent the types of features for mask adjustment in image
    segmentation.
    """
    ISLAND = 'ISLAND'
    HOLE = 'HOLE'

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


def compute_mask_iou_vectorized(masks: np.ndarray) -> np.ndarray:
    """
    Vectorized computation of the Intersection over Union (IoU) for all pairs of masks.

    Parameters:
        masks (np.ndarray): A 3D numpy array with shape `(N, H, W)`, where `N` is the
            number of masks, `H` is the height, and `W` is the width.

    Returns:
        np.ndarray: A 2D numpy array of shape `(N, N)` where each element `[i, j]` is
            the IoU between masks `i` and `j`.

    Raises:
        ValueError: If any of the masks is found to be empty.
    """
    if np.any(masks.sum(axis=(1, 2)) == 0):
        raise ValueError(
            "One or more masks are empty. Please filter out empty masks before using "
            "`compute_iou_vectorized` function."
        )

    masks_bool = masks.astype(bool)
    masks_flat = masks_bool.reshape(masks.shape[0], -1)
    intersection = np.logical_and(masks_flat[:, None], masks_flat[None, :]).sum(axis=2)
    union = np.logical_or(masks_flat[:, None], masks_flat[None, :]).sum(axis=2)
    iou_matrix = intersection / union
    return iou_matrix


def mask_non_max_suppression(masks: np.ndarray, iou_threshold: float) -> np.ndarray:
    """
    Performs Non-Max Suppression on a set of masks by prioritizing larger masks and
        removing smaller masks that overlap significantly.

    When the IoU between two masks exceeds the specified threshold, the smaller mask
    (in terms of area) is discarded. This process is repeated for each pair of masks,
    effectively filtering out masks that are significantly overlapped by larger ones.

    Parameters:
        masks (np.ndarray): A 3D numpy array with shape `(N, H, W)`, where `N` is the
            number of masks, `H` is the height, and `W` is the width.
        iou_threshold (float): The IoU threshold for determining significant overlap.

    Returns:
        np.ndarray: A 3D numpy array of filtered masks.
    """
    num_masks = masks.shape[0]
    areas = masks.sum(axis=(1, 2))
    sorted_idx = np.argsort(-areas)
    keep_mask = np.ones(num_masks, dtype=bool)
    iou_matrix = compute_mask_iou_vectorized(masks)
    for i in range(num_masks):
        if not keep_mask[sorted_idx[i]]:
            continue

        overlapping_masks = iou_matrix[sorted_idx[i]] > iou_threshold
        overlapping_masks[sorted_idx[i]] = False
        keep_mask[sorted_idx] = np.logical_and(
            keep_mask[sorted_idx],
            ~overlapping_masks)

    return masks[keep_mask]


def filter_masks_by_relative_area(
    masks: np.ndarray,
    minimum_area: float,
    maximum_area: float
) -> np.ndarray:
    """
    Filters masks based on their relative area within the total area of each mask.

    Parameters:
        masks (np.ndarray): A 3D numpy array with shape `(N, H, W)`, where `N` is the
            number of masks, `H` is the height, and `W` is the width.
        minimum_area (float): The minimum relative area threshold. Must be between `0`
            and `1`.
        maximum_area (float): The maximum relative area threshold. Must be between `0`
            and `1`.

    Returns:
        np.ndarray: A 3D numpy array containing masks that fall within the specified
            relative area range.

    Raises:
        ValueError: If `minimum_area` or `maximum_area` are outside the `0` to `1`
            range, or if `minimum_area` is greater than `maximum_area`.
    """

    if not (isinstance(masks, np.ndarray) and masks.ndim == 3):
        raise ValueError("Input must be a 3D numpy array.")

    if not (0 <= minimum_area <= 1) or not (0 <= maximum_area <= 1):
        raise ValueError("`minimum_area` and `maximum_area` must be between 0 and 1.")

    if minimum_area > maximum_area:
        raise ValueError("`minimum_area` must be less than or equal to `maximum_area`.")

    total_area = masks.shape[1] * masks.shape[2]
    relative_areas = masks.sum(axis=(1, 2)) / total_area
    return masks[(relative_areas >= minimum_area) & (relative_areas <= maximum_area)]


def adjust_mask_features_by_relative_area(
    mask: np.ndarray,
    area_threshold: float,
    feature_type: FeatureType = FeatureType.ISLAND
) -> np.ndarray:
    """
    Adjusts a mask by removing small islands or filling small holes based on a relative
    area threshold.

    !!! warning

        Running this function on a mask with small islands may result in empty masks.

    Parameters:
        mask (np.ndarray): A 2D numpy array with shape `(H, W)`, where `H` is the
            height, and `W` is the width.
        area_threshold (float): Threshold for relative area to remove or fill features.
        feature_type (FeatureType): Type of feature to adjust (`ISLAND` for removing
            islands, `HOLE` for filling holes).

    Returns:
        np.ndarray: A 2D numpy array containing mask.
    """
    height, width = mask.shape
    total_area = width * height

    mask = np.uint8(mask * 255)
    operation = (
        cv2.RETR_EXTERNAL
        if feature_type == FeatureType.ISLAND
        else cv2.RETR_CCOMP
    )
    contours, _ = cv2.findContours(mask, operation, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        relative_area = area / total_area
        if relative_area < area_threshold:
            cv2.drawContours(
                image=mask,
                contours=[contour],
                contourIdx=-1,
                color=(0 if feature_type == FeatureType.ISLAND else 255),
                thickness=-1
            )
    return np.where(mask > 0, 1, 0).astype(bool)
