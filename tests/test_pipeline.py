"""Lightweight smoke tests for the mesh smoothing pipeline.

Why this file exists:
- `Makefile` runs `python tests/test_pipeline.py`
- `setup.py` exposes an optional console entrypoint: `mesh-test=tests.test_pipeline:main`

These tests are intentionally fast and data-free (no BraTS download required).
They validate that the core smoothing functions run and return correctly-shaped outputs.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Allow running this file directly via `python tests/test_pipeline.py`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _make_unit_cube_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Return a simple triangulated cube surface mesh.

    - 8 vertices
    - 12 triangles

    This is non-degenerate and sufficient for smoothing/curvature computations.
    """

    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    faces = np.array(
        [
            # bottom (z=0)
            [0, 1, 2],
            [0, 2, 3],
            # top (z=1)
            [4, 6, 5],
            [4, 7, 6],
            # front (y=0)
            [0, 5, 1],
            [0, 4, 5],
            # back (y=1)
            [3, 2, 6],
            [3, 6, 7],
            # left (x=0)
            [0, 3, 7],
            [0, 7, 4],
            # right (x=1)
            [1, 5, 6],
            [1, 6, 2],
        ],
        dtype=np.int64,
    )

    return verts, faces


def _assert_finite_array(name: str, arr: np.ndarray) -> None:
    if not np.isfinite(arr).all():
        bad = np.argwhere(~np.isfinite(arr))[:10]
        raise AssertionError(f"{name} contains non-finite values at indices: {bad.tolist()}")


def test_smoothing_algorithms_smoke() -> None:
    # Import here to keep module import lightweight.
    from src.algorithms.smoothing import laplacian_smoothing, taubin_smoothing
    from src.algorithms.novel_algorithms_efficient import (
        geodesic_heat_smoothing,
        information_theoretic_smoothing,
        anisotropic_tensor_smoothing,
    )

    verts, faces = _make_unit_cube_mesh()

    lap = laplacian_smoothing(verts, faces, iterations=2, lambda_val=0.5)
    assert lap.shape == verts.shape
    _assert_finite_array("laplacian", lap)

    tau = taubin_smoothing(verts, faces, iterations=2, lambda_val=0.5, mu_val=-0.53)
    assert tau.shape == verts.shape
    _assert_finite_array("taubin", tau)

    geo, geo_info = geodesic_heat_smoothing(verts, faces, iterations=2)
    assert geo.shape == verts.shape
    assert isinstance(geo_info, dict)
    _assert_finite_array("geodesic_heat", geo)

    info, info_meta = information_theoretic_smoothing(verts, faces, iterations=2)
    assert info.shape == verts.shape
    assert isinstance(info_meta, dict)
    _assert_finite_array("information_theoretic", info)

    aniso, aniso_meta = anisotropic_tensor_smoothing(verts, faces, iterations=2)
    assert aniso.shape == verts.shape
    assert isinstance(aniso_meta, dict)
    _assert_finite_array("anisotropic_tensor", aniso)


def main() -> int:
    """CLI entrypoint used by `make test` / `mesh-test`."""

    test_smoothing_algorithms_smoke()
    print("OK: core smoothing algorithms executed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
