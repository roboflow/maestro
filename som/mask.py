import numpy as np


def compute_iou_vectorized(masks: np.ndarray) -> np.ndarray:
    """
    Vectorized computation of the Intersection over Union (IoU) for all pairs of masks.

    Parameters:
        masks (np.ndarray): A 3D numpy array of shape (N, H, W).

    Returns:
        np.ndarray: A 2D array of shape (N, N) where each element [i, j] is the IoU
            between masks i and j.

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
