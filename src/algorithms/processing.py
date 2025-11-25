"""Pre-processing utilities for mesh generation pipelines."""

from __future__ import annotations

import numpy as np


def coarsen_label_volume(
    label_volume: np.ndarray,
    *,
    canonical_labels: tuple[int, ...] = (1, 2, 4),
    max_groups: int = 3,
) -> np.ndarray:
    """Compress verbose atlas-style labels into a small, indexed set.

    When BraTS-style labels (1/2/4) are present we leave the volume untouched so
    that downstream metrics continue to reference the clinically meaningful
    classes. Otherwise we fall back to quantile binning of the non-zero label
    IDs, which converts thousands of fragment IDs into at most ``max_groups``
    stable categories. This keeps semantic smoothing tractable and ensures label
    transitions still align with anatomical boundaries.
    """

    if label_volume.ndim != 3:
        raise ValueError("label_volume must be 3D")

    data = np.asarray(label_volume)
    nonzero = np.unique(data[data > 0])

    if nonzero.size == 0 or max_groups < 1:
        return data

    if canonical_labels:
        canonical_found = any(lbl in nonzero for lbl in canonical_labels)
        if canonical_found:
            return data

    if nonzero.size <= max_groups:
        return data

    quantiles = np.linspace(0.0, 1.0, max_groups + 1)[1:-1]
    thresholds = np.quantile(nonzero.astype(np.float64), quantiles)
    thresholds = np.unique(thresholds)

    collapsed = np.zeros_like(data, dtype=np.int16)
    nz_mask = data > 0

    if thresholds.size == 0:
        collapsed[nz_mask] = 1
        return collapsed

    collapsed[nz_mask] = np.digitize(data[nz_mask], thresholds, right=False) + 1
    return collapsed


def map_labels_to_vertices(label_volume: np.ndarray, affine: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Map segmentation labels to mesh vertices via nearest-neighbor sampling.

    Parameters
    ----------
    label_volume:
        Raw NIfTI data array shaped (H, W, D) containing integer tissue IDs.
    affine:
        4x4 affine matrix mapping voxel indices to world coordinates.
    verts:
        Mesh vertices shaped (N, 3) expressed in world coordinates.

    Returns
    -------
    np.ndarray
        Array of length N with the label assigned to each vertex.
    """
    if label_volume.ndim != 3:
        raise ValueError("label_volume must be a 3D array")
    if affine.shape != (4, 4):
        raise ValueError("affine must be a 4x4 matrix")
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("verts must be shaped (N, 3)")

    label_data = np.asarray(label_volume)
    verts_world = np.asarray(verts, dtype=np.float64)

    try:
        inv_affine = np.linalg.inv(affine)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Affine matrix is singular and cannot be inverted") from exc

    ones = np.ones((verts_world.shape[0], 1), dtype=np.float64)
    verts_h = np.hstack([verts_world, ones])
    voxel_coords = (verts_h @ inv_affine.T)[:, :3]

    shape = np.array(label_data.shape, dtype=np.float64)
    inside = (
        np.all(np.isfinite(voxel_coords), axis=1)
        & (voxel_coords[:, 0] >= -0.5)
        & (voxel_coords[:, 1] >= -0.5)
        & (voxel_coords[:, 2] >= -0.5)
        & (voxel_coords[:, 0] <= shape[0] - 0.5)
        & (voxel_coords[:, 1] <= shape[1] - 0.5)
        & (voxel_coords[:, 2] <= shape[2] - 0.5)
    )

    if inside.mean() < 0.5:
        # Many vertices are outside the volume, which often happens when verts
        # are already in voxel space (scaled by spacing). Fall back to dividing
        # by inferred voxel sizes to recover voxel coordinates directly.
        voxel_sizes = np.linalg.norm(affine[:3, :3], axis=0)
        voxel_sizes[voxel_sizes == 0] = 1.0
        voxel_coords = verts_world / voxel_sizes

    voxel_indices = np.rint(voxel_coords).astype(np.int64)
    max_indices = np.array(label_data.shape) - 1
    min_indices = np.zeros(3, dtype=np.int64)
    voxel_indices = np.clip(voxel_indices, min_indices, max_indices)

    labels = label_data[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]]
    return labels.astype(np.int16, copy=False)
