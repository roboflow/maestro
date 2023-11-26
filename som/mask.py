import numpy as np


def compute_iou_vectorized(masks: np.ndarray) -> np.ndarray:
    """
    Vectorized computation of the Intersection over Union (IoU) for all pairs of masks.

    Parameters:
        masks (np.ndarray): A 3D numpy array of shape (N, H, W).

    Returns:
        np.ndarray: A 2D array of shape (N, N) where each element [i, j] is the IoU
            between masks i and j.
    """
    flat_masks = masks.reshape(masks.shape[0], -1)
    intersections = np.dot(flat_masks, flat_masks.T)
    area_sum = flat_masks.sum(axis=1).reshape(-1, 1) + flat_masks.sum(axis=1)
    unions = area_sum - intersections
    iou_matrix = np.where(unions != 0, intersections / unions, 0)
    return iou_matrix


def mask_non_max_suppression(masks: np.ndarray, iou_threshold: float) -> np.ndarray:
    """
    Performs Non-Max Suppression on a set of masks by prioritizing larger masks and
        removing smaller masks that overlap significantly.

    Parameters:
        masks (np.ndarray): A 3D numpy array with shape (N, H, W), where N is the number
            of masks.
        iou_threshold (float): The IoU threshold for determining significant overlap.

    Returns:
        np.ndarray: A 3D numpy array of filtered masks.
    """
    num_masks = masks.shape[0]
    areas = masks.sum(axis=(1, 2))
    sorted_idx = np.argsort(-areas)
    keep_mask = np.ones(num_masks, dtype=bool)
    iou_matrix = compute_iou_vectorized(masks)

    for i in range(num_masks):
        if not keep_mask[sorted_idx[i]]:
            continue

        overlapping_masks = iou_matrix[sorted_idx[i]] > iou_threshold
        overlapping_masks[sorted_idx[i]] = False
        keep_mask[sorted_idx] = np.logical_and(keep_mask[sorted_idx], ~overlapping_masks)

    return masks[keep_mask]