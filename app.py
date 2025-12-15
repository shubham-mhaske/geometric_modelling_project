"""Streamlit demo app (clean + demo-first).

Goal: A stable, uncluttered demo for the project pipeline:
BraTS mask → Marching Cubes mesh → smoothing → metrics + visualization.

Design constraints:
- macOS + Streamlit can crash if native VTK windows are created off the main thread.
- Therefore: no `pv.Plotter().show()` or anything that opens a window. We use Plotly WebGL.
"""

from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal

import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import pyvista as pv
import streamlit as st
from skimage import measure

from src.algorithms import smoothing
from src.algorithms.novel_algorithms_efficient import (
    anisotropic_tensor_smoothing,
    geodesic_heat_smoothing,
    information_theoretic_smoothing,
)


# ---- PyVista configuration (safe: no windows) ----
pv.OFF_SCREEN = True


APP_VERSION = "2025-12-14b"


APP_BG = "#f7f8fc"
CARD_BG = "#ffffff"
PLOT_SCENE_BG = "#ffffff"  # Plotly 3D scene background (set back to white)
CARD_BORDER = "rgba(15, 23, 42, 0.10)"
TEXT = "rgba(15, 23, 42, 0.92)"
MUTED = "rgba(15, 23, 42, 0.62)"
ACCENT = "#b8860b"  # darkgoldenrod-ish


@dataclass(frozen=True)
class PatientFiles:
    mask: str
    t1n: Optional[str] = None
    t1c: Optional[str] = None
    t2w: Optional[str] = None
    t2f: Optional[str] = None


def _inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        /* IMPORTANT: Do NOT override icon fonts globally, or Streamlit's dropdown icons
           can render as the literal ligature text (e.g., 'keyboard_arrow_down'). */
        html, body, p, h1, h2, h3, h4, h5, h6, label, input, textarea, button {{
            font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
        }}
        .material-icons, .material-symbols-outlined {{
            font-family: 'Material Icons' !important;
        }}
        #MainMenu, footer, header {{ visibility: hidden; }}

        [data-testid="stAppViewContainer"] {{
            background:
              radial-gradient(1200px 700px at 10% 0%, rgba(184,134,11,0.09) 0%, rgba(0,0,0,0) 55%),
              radial-gradient(900px 600px at 85% 10%, rgba(128,0,0,0.06) 0%, rgba(0,0,0,0) 60%),
              {APP_BG};
            color: {TEXT};
        }}

        [data-testid="stSidebar"] {{
            background: #ffffff !important;
            border-right: 1px solid {CARD_BORDER};
        }}

        /* Avoid forcing colors on everything; let BaseWeb widgets render correctly */
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {{
            color: {TEXT} !important;
        }}

        /* Fix label spacing so it doesn't collide with dropdown widgets */
        [data-testid="stSidebar"] label {{
            display: block;
            margin-bottom: 0.25rem;
            font-weight: 600;
        }}

        .hero {{
            text-align: left;
            margin: 0.4rem 0 0.9rem 0;
        }}
        .byline {{
            display: inline-block;
            font-size: 0.82rem;
            font-weight: 500;
            letter-spacing: 0.06em;
            color: rgba(15, 23, 42, 0.55);
            margin-bottom: 0.45rem;
        }}
        .byline .heart {{
            opacity: 0.55;
            margin-left: 0.35rem;
        }}
        .hero h1 {{
            font-size: 2.0rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin: 0;
            color: {TEXT};
        }}
        .hero p {{
            margin: 0.35rem 0 0 0;
            color: {MUTED};
            font-size: 0.98rem;
        }}

        .kpi {{
            background: {CARD_BG};
            border: 1px solid {CARD_BORDER};
            border-radius: 14px;
            padding: 0.85rem 0.9rem;
        }}
        .kpi .v {{
            font-size: 1.35rem;
            font-weight: 800;
            color: {ACCENT};
            line-height: 1.1;
        }}
        .kpi .l {{
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: {MUTED};
            margin-top: 0.25rem;
        }}
        .subtle {{ color: {MUTED}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _plotly_lighting(preset: str) -> dict:
    p = (preset or "soft").strip().lower()
    if p in {"crisp", "high", "high-contrast"}:
        # More specular to help see "staircase"/faceting during demos.
        return dict(ambient=0.46, diffuse=0.92, specular=0.58, roughness=0.22, fresnel=0.26)
    # Bright baseline for a light UI, with enough specular to read surface steps.
    # Note: Plotly Mesh3d is effectively single-light; we add a separate fill-lit trace elsewhere.
    return dict(ambient=0.76, diffuse=0.74, specular=0.22, roughness=0.56, fresnel=0.08)


def _faces_reversed(faces: np.ndarray) -> np.ndarray:
    f = np.asarray(faces, dtype=np.int32)
    if f.ndim != 2 or f.shape[1] != 3:
        return f
    # Reverse winding (swap j/k) so normals flip.
    return f[:, [0, 2, 1]]


def _inset_vertices(verts: np.ndarray, scale: float = 0.999) -> np.ndarray:
    """Slightly scale a mesh toward its centroid.

    Used to avoid coplanar z-fighting when we draw an inward-facing duplicate mesh
    for improved interior lighting.
    """
    v = np.asarray(verts, dtype=np.float32)
    if v.size == 0:
        return v
    c = v.mean(axis=0)
    s = float(scale)
    return (c + (v - c) * s).astype(np.float32)


def _context_lighting() -> dict:
    """Matte lighting for the translucent context mesh (reduces shimmer during rotation)."""
    return dict(ambient=0.85, diffuse=0.35, specular=0.02, roughness=0.95, fresnel=0.02)


def _fill_lighting() -> dict:
    """Very even lighting used for the "fill light" rendering pass.

    Plotly Mesh3d doesn't support multiple light sources. We approximate a second light by
    layering a duplicate trace. To avoid depth-sorting shimmer that can look like the mesh
    is "rotating" relative to itself, keep this pass mostly ambient and non-specular.
    """
    return dict(ambient=1.00, diffuse=0.18, specular=0.00, roughness=1.00, fresnel=0.00)


def _colorscale_greys():
    # Use a neutral, print-friendly grayscale for detail shading.
    return "Greys"


def _detail_colorscale(style: str):
    """Subtle, demo-friendly palette for geometry/detail shading.

    Keep it low-saturation so it reads like "shading" not "heatmap".
    """
    s = (style or "neutral").strip().lower()
    if s in {"cool", "cool blue", "blue", "ocean"}:
        # Slate → ink (avoid starting at pure white; the app background is very light).
        return [
            [0.0, "#e2e8f0"],
            [0.55, "#94a3b8"],
            [1.0, "#0f172a"],
        ]
    return _colorscale_greys()


def _geometry_colors(style: str) -> Tuple[str, str]:
    """(original_color, smoothed_color) for plain geometry mode."""
    s = (style or "neutral").strip().lower()
    if s in {"cool", "cool blue", "blue", "ocean"}:
        # Subtle cool tones with a bit more contrast against the light UI.
        return "rgba(100, 116, 139, 0.92)", "rgba(15, 23, 42, 0.96)"
    # Neutral (default)
    return "rgba(148, 163, 184, 0.88)", "rgba(71, 85, 105, 0.90)"


def _scene_cube_bounds(v: np.ndarray, padding: float = 0.06):
    center = v.mean(axis=0)
    ranges = v.max(axis=0) - v.min(axis=0)
    max_range = float(ranges.max()) if ranges.size else 1.0
    half = max_range / 2.0 + max_range * float(padding)
    return center, half


def _mesh_fingerprint(verts: np.ndarray, faces: np.ndarray) -> str:
    """Short mesh signature to verify that selection changes actually change the mesh."""
    v = np.asarray(verts, dtype=np.float32)
    if v.size == 0:
        return "empty"
    mn = v.min(axis=0)
    mx = v.max(axis=0)
    ext = mx - mn
    c = v.mean(axis=0)
    return (
        f"V={len(v):,} F={len(faces):,} "
        f"bbox=({ext[0]:.1f},{ext[1]:.1f},{ext[2]:.1f})mm "
        f"ctr=({c[0]:.1f},{c[1]:.1f},{c[2]:.1f})"
    )


def _camera_preset(name: str) -> dict:
    n = (name or "isometric").strip().lower()
    if n == "front":
        eye = dict(x=0.0, y=2.2, z=0.0)
    elif n == "top":
        eye = dict(x=0.0, y=0.0, z=2.4)
    elif n == "side":
        eye = dict(x=2.4, y=0.0, z=0.0)
    else:
        eye = dict(x=1.7, y=1.7, z=1.2)
    return dict(
        eye=eye,
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        projection=dict(type="perspective"),
    )


@st.cache_resource
def index_patients(root: str = "data") -> Dict[str, PatientFiles]:
    """Discover BraTS folders and available modalities.

    Expected structure: data/data/BraTS-*/...nii.gz
    """
    patients: Dict[str, PatientFiles] = {}
    data_root = os.path.join(root, "data")
    for folder in sorted(glob.glob(os.path.join(data_root, "BraTS-*"))):
        pid = os.path.basename(folder)
        files = {"mask": None, "t1n": None, "t1c": None, "t2w": None, "t2f": None}
        for f in os.listdir(folder):
            fp = os.path.join(folder, f)
            if "mask.nii" in f:
                files["mask"] = fp
            elif "t1n.nii" in f and "voided" not in f:
                files["t1n"] = fp
            elif "t1c.nii" in f:
                files["t1c"] = fp
            elif "t2w.nii" in f:
                files["t2w"] = fp
            elif "t2f.nii" in f:
                files["t2f"] = fp
        if files["mask"]:
            patients[pid] = PatientFiles(**files)  # type: ignore[arg-type]
    return patients


@st.cache_data
def load_nifti(path: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    nii = nib.load(path)
    vol = np.asarray(nii.get_fdata(), dtype=np.float32)
    zooms = nii.header.get_zooms()[:3]
    return vol, (float(zooms[0]), float(zooms[1]), float(zooms[2]))


@st.cache_data
def load_labels(path: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Load a label/mask volume as int16."""
    nii = nib.load(path)
    vol = np.asarray(nii.get_fdata(), dtype=np.int16)
    zooms = nii.header.get_zooms()[:3]
    return vol, (float(zooms[0]), float(zooms[1]), float(zooms[2]))


@st.cache_data
def mri_path_to_surface_mesh(
    mri_path: str,
    *,
    downsample: int = 2,
    threshold_percentile: float = 25.0,
    sigma: float = 1.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a coarse outer surface from an MRI volume.

    This is intended for a **visual context** mesh and for optional "smooth whole brain" demos.
    It is a heuristic surface (not a clinically accurate skull extraction).
    """
    vol, zooms = load_nifti(mri_path)
    vol = np.asarray(vol, dtype=np.float32)

    ds = int(max(1, downsample))
    if ds > 1:
        vol = vol[::ds, ::ds, ::ds]
        zooms = (zooms[0] * ds, zooms[1] * ds, zooms[2] * ds)

    nz = vol[vol > 0]
    if nz.size == 0:
        raise ValueError("MRI volume appears empty (all zeros).")

    # BraTS MRIs are commonly skull-stripped; a true skull surface isn't present.
    # For a stable context mesh, use the nonzero foreground by default when percentile <= 0.
    if float(threshold_percentile) <= 0.0:
        mask = vol > 0
    else:
        thr = float(np.percentile(nz, float(threshold_percentile)))
        mask = vol > thr

    from skimage.filters import gaussian
    from skimage.morphology import ball, binary_closing, remove_small_holes, remove_small_objects

    mask = binary_closing(mask, footprint=ball(2))
    mask = remove_small_objects(mask, min_size=8000)
    mask = remove_small_holes(mask, area_threshold=8000)
    smooth = gaussian(mask.astype(np.float32), sigma=float(sigma), preserve_range=True)

    verts, faces, _, _ = measure.marching_cubes(
        smooth,
        level=0.5,
        spacing=(zooms[0], zooms[1], zooms[2]),
    )
    if len(verts) > 500:
        verts, faces = _largest_component(verts, faces)
    return verts.astype(np.float32), faces.astype(np.int32)


def _largest_component(verts: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Keep largest connected component (PyVista connectivity, window-free)."""
    try:
        faces_padded = np.hstack(
            [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces.astype(np.int64)]
        )
        poly = pv.PolyData(verts, faces_padded)
        largest = poly.connectivity(extraction_mode='largest')
        v = np.asarray(largest.points)
        f = np.asarray(largest.faces).reshape(-1, 4)[:, 1:4]
        return v, f
    except Exception:
        return verts, faces


def _select_region_mask(labels: np.ndarray, region: str) -> np.ndarray:
    region = (region or "all").strip().lower()
    mapping = {
        "all": [1, 2, 4],
        "core": [1, 4],
        "enhancing": [4],
        "edema": [2],
        "necrotic": [1],
    }
    use = mapping.get(region, [1, 2, 4])
    m = np.isin(labels, use)
    if int(m.sum()) == 0:
        m = labels != 0
    return m


@st.cache_data
def mask_path_to_mesh(
    mask_path: str,
    region: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert label volume to mesh (verts, faces, vertex_labels).

    Returns:
    - verts: (N, 3)
    - faces: (M, 3)
    - vertex_labels: (N,)
    """
    labels, zooms = load_labels(mask_path)

    from skimage.filters import gaussian
    from skimage.morphology import ball, binary_closing

    region_mask = _select_region_mask(labels, region)
    region_mask = binary_closing(region_mask, footprint=ball(1))
    smooth = gaussian(region_mask.astype(np.float32), sigma=0.55, preserve_range=True)

    if float(smooth.max()) <= 0.0:
        raise ValueError("No tumor voxels found for this region.")

    verts, faces, _, _ = measure.marching_cubes(
        smooth, level=0.5, spacing=(zooms[0], zooms[1], zooms[2])
    )

    if len(verts) > 100:
        verts, faces = _largest_component(verts, faces)

    # Map vertex to nearest voxel label (fast + good enough for coloring)
    voxel = (verts / np.array(zooms)).astype(np.int32)
    voxel = np.clip(voxel, 0, np.array(labels.shape, dtype=np.int32) - 1)
    vertex_labels = labels[voxel[:, 0], voxel[:, 1], voxel[:, 2]].astype(np.int16)

    return verts.astype(np.float32), faces.astype(np.int32), vertex_labels


def _colorscale_continuous(name: str):
    n = (name or "Cividis").strip().lower()
    # Use built-in Plotly continuous scales by name
    if n in {"cividis", "viridis", "plasma", "magma", "inferno", "turbo", "ice"}:
        return n
    return "cividis"


def _colorscale_diverging(name: str):
    n = (name or "RdBu").strip().lower()
    # Plotly accepts these by name
    if n in {"rdbu", "rdylbu", "picnic", "portland"}:
        return {"rdbu": "RdBu", "rdylbu": "RdYlBu", "picnic": "Picnic", "portland": "Portland"}[n]
    return "RdBu"


def _colorscale_magnitude():
    """High-contrast sequential palette for nonnegative evidence (mm, |Δκ|, etc.).

    Avoid dark blues (hard to see on a light UI). Use a warm, print-friendly ramp.
    """

    return [
        [0.0, "#fff7ec"],  # very light peach
        [0.25, "#fee8c8"],
        [0.55, "#fdbb84"],
        [0.80, "#fc8d59"],
        [1.0, "#b30000"],  # deep red
    ]


def _colorscale_signed():
    """Diverging palette for signed evidence (negative ↔ positive) without dark blues."""

    return [
        [0.0, "#166534"],  # green (negative)
        [0.25, "#86efac"],
        [0.50, "#f8fafc"],  # near-white at zero
        [0.75, "#fdba74"],
        [1.0, "#9a3412"],  # orange/brown (positive)
    ]


def _pick_mri_path(files: PatientFiles, preferred: str) -> Optional[str]:
    p = (preferred or "t1n").strip().lower()
    cand = getattr(files, p, None)
    if cand is not None:
        return cand
    return files.t1n or files.t1c or files.t2w or files.t2f


def _labels_to_bins(labels: np.ndarray) -> np.ndarray:
    """Map BraTS labels {0,1,2,4} to compact bins {0,1,2,3} for stable discrete coloring."""
    lab = np.asarray(labels, dtype=np.int16)
    out = np.zeros_like(lab, dtype=np.int16)
    out[lab == 1] = 1
    out[lab == 2] = 2
    out[lab == 4] = 3
    return out


def _label_colorscale_discrete():
    # 0: background-ish, 1: necrotic red, 2: edema blue, 3: enhancing amber
    return [
        [0.0, "#64748b"],
        [0.333333, "#ef4444"],
        [0.666666, "#2563eb"],
        [1.0, "#f59e0b"],
    ]


def _robust_range(x: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> Tuple[float, float]:
    arr = np.asarray(x, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(arr, lo))
    vmax = float(np.percentile(arr, hi))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def compute_curvature(
    verts: np.ndarray,
    faces: np.ndarray,
    kind: Literal["mean", "gaussian"] = "mean",
) -> np.ndarray:
    """Compute curvature per-vertex using PyVista/VTK (no windows)."""
    faces_padded = np.hstack(
        [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces.astype(np.int64)]
    )
    poly = pv.PolyData(verts, faces_padded)
    curv_type = "mean" if kind == "mean" else "gaussian"
    curv = poly.curvature(curv_type=curv_type)
    return np.asarray(curv, dtype=np.float32)


def apply_smoothing(
    verts: np.ndarray,
    faces: np.ndarray,
    algo: str,
    iterations: int,
) -> Tuple[np.ndarray, float]:
    algo = (algo or "none").strip().lower()
    if algo == "none" or iterations <= 0:
        return verts.copy(), 0.0

    v = verts.copy()
    t0 = time.time()
    if algo == "laplacian":
        v = smoothing.laplacian_smoothing(v, faces, iterations)
    elif algo == "taubin":
        v = smoothing.taubin_smoothing(v, faces, iterations)
    elif algo == "geodesic_heat":
        v, _ = geodesic_heat_smoothing(v, faces, iterations=iterations)
    elif algo == "anisotropic_tensor":
        v, _ = anisotropic_tensor_smoothing(v, faces, iterations=iterations)
    elif algo == "information_theoretic":
        v, _ = information_theoretic_smoothing(v, faces, iterations=iterations)
    else:
        v = smoothing.taubin_smoothing(v, faces, iterations)
    dt = time.time() - t0
    return v.astype(np.float32), float(dt)


def _poly_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    faces_padded = np.hstack(
        [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces.astype(np.int64)]
    )
    poly = pv.PolyData(verts, faces_padded)
    return float(poly.volume)


def mesh_figure(
    verts_a: np.ndarray,
    faces_a: np.ndarray,
    labels: np.ndarray,
    verts_b: np.ndarray,
    faces_b: np.ndarray,
    disp: np.ndarray,
    show_wireframe: bool,
    lighting_preset: str,
    camera: str,
) -> go.Figure:
    lighting = _plotly_lighting(lighting_preset)
    center, half = _scene_cube_bounds(np.vstack([verts_a, verts_b]))

    # Original colored by label
    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=verts_a[:, 0],
            y=verts_a[:, 1],
            z=verts_a[:, 2],
            i=faces_a[:, 0],
            j=faces_a[:, 1],
            k=faces_a[:, 2],
            intensity=labels.astype(np.float32),
            colorscale=[
                [0.0, "#404052"],
                [0.33, "#ef4444"],
                [0.66, "#3b82f6"],
                [1.0, "#fbbf24"],
            ],
            opacity=0.98,
            lighting=lighting,
            lightposition=dict(x=950, y=1200, z=1600),
            flatshading=False,
            hoverinfo="skip",
            name="Original",
            showscale=False,
        )
    )

    # Smoothed colored by displacement
    disp_n = disp
    if disp_n.size and float(disp_n.max()) > 0:
        disp_n = (disp_n - float(disp_n.min())) / (float(disp_n.max()) - float(disp_n.min()) + 1e-9)
    fig.add_trace(
        go.Mesh3d(
            x=verts_b[:, 0],
            y=verts_b[:, 1],
            z=verts_b[:, 2],
            i=faces_b[:, 0],
            j=faces_b[:, 1],
            k=faces_b[:, 2],
            intensity=disp_n.astype(np.float32),
            colorscale=_colorscale_magnitude(),
            opacity=1.0,
            lighting=lighting,
            lightposition=dict(x=950, y=1200, z=1600),
            flatshading=False,
            hoverinfo="skip",
            name="Smoothed",
            showscale=False,
        )
    )

    if show_wireframe:
        fig.add_trace(
            go.Scatter3d(
                x=verts_b[:, 0],
                y=verts_b[:, 1],
                z=verts_b[:, 2],
                mode="markers",
                marker=dict(size=1, color="rgba(255,255,255,0.06)"),
                hoverinfo="skip",
                name="Wire",
            )
        )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=APP_BG,
        plot_bgcolor=APP_BG,
        showlegend=False,
        height=620,
        scene=dict(
            xaxis=dict(visible=False, range=[center[0] - half, center[0] + half]),
            yaxis=dict(visible=False, range=[center[1] - half, center[1] + half]),
            zaxis=dict(visible=False, range=[center[2] - half, center[2] + half]),
            bgcolor=PLOT_SCENE_BG,
            aspectmode="cube",
            camera=_camera_preset(camera),
        ),
    )
    return fig


def mesh_single_figure(
    verts: np.ndarray,
    faces: np.ndarray,
    *,
    intensity: Optional[np.ndarray] = None,
    colorscale=None,
    cmin: Optional[float] = None,
    cmax: Optional[float] = None,
    showscale: bool = False,
    colorbar_title: Optional[str] = None,
    color: Optional[str] = None,
    title: Optional[str] = None,
    show_wireframe: bool = False,
    lighting_preset: str = "soft",
    camera: str = "isometric",
    height: int = 560,
    two_sided: bool = True,
    two_lights: bool = True,
    flatshading: bool = False,
) -> go.Figure:
    lighting = _plotly_lighting(lighting_preset)
    center, half = _scene_cube_bounds(verts)

    key_light = dict(x=1100, y=1300, z=1700)
    fill_light = dict(x=-1100, y=-900, z=1400)

    kwargs = dict(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=1.0,
        lighting=lighting,
        lightposition=key_light,
        flatshading=bool(flatshading),
        hoverinfo="skip",
    )
    if intensity is not None and colorscale is not None:
        kwargs.update(
            intensity=np.asarray(intensity, dtype=np.float32),
            colorscale=colorscale,
            showscale=bool(showscale),
        )
        if colorbar_title and bool(showscale):
            # NOTE: Mesh3d colorbar does not support `titleside` in some Plotly versions.
            # Keep it compatible: just set a title.
            kwargs["colorbar"] = dict(
                title=dict(text=str(colorbar_title), font=dict(color=TEXT, size=12)),
                tickfont=dict(color=TEXT, size=11),
                outlinewidth=0,
                thickness=16,
            )
        if cmin is not None:
            kwargs["cmin"] = float(cmin)
        if cmax is not None:
            kwargs["cmax"] = float(cmax)
    elif color is not None:
        kwargs.update(color=color)
    else:
        kwargs.update(color="#ff6b6b")

    fig = go.Figure()
    # Primary surface (outward faces)
    fig.add_trace(go.Mesh3d(**kwargs))

    # Approximate a second light source by layering a subtle duplicate trace lit from
    # another direction. We offset it slightly to avoid coplanar z-fighting.
    if bool(two_lights):
        kwargs_fill = dict(kwargs)
        kwargs_fill.pop("colorbar", None)
        v_out = _inset_vertices(verts, scale=1.00035)
        kwargs_fill.update(
            x=v_out[:, 0],
            y=v_out[:, 1],
            z=v_out[:, 2],
            lightposition=fill_light,
            # Keep this pass subtle and stable (less transparency popping during rotation).
            opacity=0.22,
            lighting=_fill_lighting(),
            showscale=False,
        )
        fig.add_trace(go.Mesh3d(**kwargs_fill))

    # Two-sided lighting hack: add a tiny inset, reversed-winding duplicate.
    # This makes interior views readable (camera inside the mesh) without coplanar z-fighting.
    if bool(two_sided):
        kwargs_in = dict(kwargs)
        kwargs_in.pop("colorbar", None)
        v_in = _inset_vertices(verts, scale=0.999)
        f_in = _faces_reversed(faces)
        kwargs_in.update(
            x=v_in[:, 0],
            y=v_in[:, 1],
            z=v_in[:, 2],
            i=f_in[:, 0],
            j=f_in[:, 1],
            k=f_in[:, 2],
            # Light the interior from the opposite direction to improve inside views.
            lightposition=fill_light,
            lighting=_fill_lighting(),
            showscale=False,
        )
        # If we are using a scalar field, mirror its values onto the inset trace.
        # Keep colorbar only on the primary trace.
        if intensity is not None and colorscale is not None:
            kwargs_in["intensity"] = np.asarray(intensity, dtype=np.float32)
            kwargs_in["colorscale"] = colorscale
            if cmin is not None:
                kwargs_in["cmin"] = float(cmin)
            if cmax is not None:
                kwargs_in["cmax"] = float(cmax)

        fig.add_trace(go.Mesh3d(**kwargs_in))

    if show_wireframe:
        fig.add_trace(
            go.Scatter3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                mode="markers",
                marker=dict(size=1, color="rgba(15,23,42,0.08)"),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        margin=dict(l=0, r=0, t=28 if title else 0, b=0),
        paper_bgcolor=APP_BG,
        plot_bgcolor=APP_BG,
        showlegend=False,
        height=int(height),
        font=dict(color=TEXT),
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=14, color=TEXT)) if title else None,
        scene=dict(
            xaxis=dict(visible=False, range=[center[0] - half, center[0] + half]),
            yaxis=dict(visible=False, range=[center[1] - half, center[1] + half]),
            zaxis=dict(visible=False, range=[center[2] - half, center[2] + half]),
            bgcolor=PLOT_SCENE_BG,
            aspectmode="cube",
            camera=_camera_preset(camera),
        ),
    )
    return fig


def mri_slice_figure(
    vol: np.ndarray,
    mask: np.ndarray,
    plane: str,
    idx: int,
    show_overlay: bool,
) -> go.Figure:
    plane = (plane or "axial").strip().lower()
    if plane == "coronal":
        sl = vol[:, idx, :]
        ml = mask[:, idx, :]
    elif plane == "sagittal":
        sl = vol[idx, :, :]
        ml = mask[idx, :, :]
    else:
        sl = vol[:, :, idx]
        ml = mask[:, :, idx]

    sl = sl.T
    ml = ml.T
    vmin, vmax = (0.0, 1.0)
    if np.any(sl > 0):
        vmin, vmax = np.percentile(sl[sl > 0], [2, 98])

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=sl,
            colorscale="gray",
            zmin=float(vmin),
            zmax=float(vmax),
            showscale=False,
            hoverinfo="skip",
        )
    )
    if show_overlay and np.any(ml > 0):
        fig.add_trace(
            go.Heatmap(
                z=ml,
                colorscale=[
                    [0.0, "rgba(0,0,0,0)"],
                    [0.25, "rgba(239,68,68,0.55)"],
                    [0.50, "rgba(59,130,246,0.55)"],
                    [1.0, "rgba(251,191,36,0.60)"],
                ],
                showscale=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=APP_BG,
        plot_bgcolor=APP_BG,
        height=380,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x"),
    )
    return fig


def _kpi(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="kpi">
          <div class="v">{value}</div>
          <div class="l">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="High-Fidelity Mesh Smoothing",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()

    st.markdown(
        """
        <div class="hero">
                    <div class="byline">Shubham Mhaske <span class="heart">|</span> Geometric Modelling</div>
                    <h1>High-Fidelity Mesh Smoothing</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    patients = index_patients("data")
    if not patients:
        st.error("No BraTS cases found under `data/data/BraTS-*`. Run the data download step first.")
        st.stop()

    # ---------------- Sidebar (minimal) ----------------
    with st.sidebar:
        st.markdown("## Controls")

        st.caption(f"App version: **{APP_VERSION}**")
        if st.button("Clear cache", help="If selections seem stuck, clear Streamlit caches and rerun."):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

        pid = st.selectbox("Patient", sorted(patients.keys()))

        mesh_target = st.selectbox(
            "Mesh target",
            ["Tumor (from mask)", "Head/brain surface (from MRI)"],
            index=0,
            help="Tumor is the project focus; MRI surface is for optional whole-surface smoothing demos.",
        )

        region = "core"
        if mesh_target == "Tumor (from mask)":
            # Only show region choices that exist for this patient (avoids the "does nothing" feel).
            labels_dbg, _ = load_labels(patients[pid].mask)
            region_counts = {
                "core": int(np.count_nonzero(np.isin(labels_dbg, [1, 4]))),
                "all": int(np.count_nonzero(np.isin(labels_dbg, [1, 2, 4]))),
                "enhancing": int(np.count_nonzero(labels_dbg == 4)),
                "edema": int(np.count_nonzero(labels_dbg == 2)),
                "necrotic": int(np.count_nonzero(labels_dbg == 1)),
            }
            region_options = [k for k in ["core", "all", "enhancing", "edema", "necrotic"] if region_counts[k] > 0]
            if len(region_options) <= 1:
                region = region_options[0] if region_options else "all"
                st.caption(f"Tumor region: **{region}**")
            else:
                region = st.selectbox(
                    "Tumor region",
                    region_options,
                    index=0,
                    help="Only non-empty regions for this patient are shown.",
                )

        algo_label = st.selectbox(
            "Smoothing",
            [
                "Anisotropic Tensor",
                "Geodesic Heat",
                "Info-Theoretic",
                "Taubin",
                "Laplacian",
                "None",
            ],
            index=0,
        )
        algo_key = {
            "Anisotropic Tensor": "anisotropic_tensor",
            "Geodesic Heat": "geodesic_heat",
            "Info-Theoretic": "information_theoretic",
            "Taubin": "taubin",
            "Laplacian": "laplacian",
            "None": "none",
        }[algo_label]
        iterations = 0
        if algo_key != "none":
            iterations = st.slider("Iterations", 1, 30, 10)

        st.markdown("---")
        viz_options = [
            "Geometry (single color)",
            "Displacement (magnitude)",
            "Curvature change (|Δ|)",
            "Curvature change (signed)",
        ]
        if mesh_target == "Tumor (from mask)":
            viz_options.append("Tumor labels")

        view_mode = st.selectbox(
            "Visualization",
            viz_options,
            index=0,
            help="Default is a single-color mesh for clear shape/smoothness comparison. Use the other modes for metric-based color evidence.",
        )

        geometry_style = "Neutral"
        if view_mode == "Geometry (single color)":
            geometry_style = st.selectbox(
                "Geometry style",
                ["Neutral", "Cool"],
                index=0,
                help="One extra subtle color option for the geometry view (same visualization, different tint).",
            )

        show_context = st.checkbox(
            "Show brain context",
            value=True,
            help="Shows a translucent MRI-derived surface behind the mesh (auto-disabled when the mesh target is already the MRI surface to avoid z-fighting).",
        )
        context_opacity = st.slider("Context opacity", 0.02, 0.22, 0.10)

        show_mri = st.checkbox(
            "Show MRI slice",
            value=False,
            help="Shows an MRI slice panel on the right. Turn it off to make the 3D viewports wider.",
        )

        show_overlay = False
        if show_mri:
            show_overlay = st.checkbox("Show MRI slice overlay", value=True)

        with st.expander("Advanced", expanded=False):
            enhance_detail = st.checkbox(
                "Enhance surface detail",
                value=True,
                help="Adds subtle curvature shading in Geometry mode so surface complexity and smoothing are easier to see.",
            )
            facet_shading = st.checkbox(
                "Facet shading",
                value=False,
                help="Flat triangle shading (helps show marching-cubes faceting / staircase artifacts).",
            )
            plot_height = st.slider(
                "3D plot height",
                min_value=420,
                max_value=860,
                value=560,
                step=20,
                help="Adjust the size of the 3D viewports.",
            )

        st.markdown("---")

    # ---------------- Load + process ----------------
    files = patients[pid]
    # Auto-pick MRI from the same patient folder (keeps datapoints correctly paired)
    mri_path = _pick_mri_path(files, "t1n")

    # Build main mesh depending on target
    if mesh_target == "Tumor (from mask)":
        try:
            verts, faces, vlabels = mask_path_to_mesh(files.mask, region)
        except Exception as e:
            st.error(f"Tumor mesh generation failed: {e}")
            st.stop()
    else:
        if mri_path is None:
            st.error("No MRI modality found for this case; cannot build a surface mesh.")
            st.stop()
        try:
            verts, faces = mri_path_to_surface_mesh(mri_path, downsample=2, threshold_percentile=0.0, sigma=0.9)
            vlabels = np.zeros((verts.shape[0],), dtype=np.int16)
        except Exception as e:
            st.error(f"Surface mesh generation failed: {e}")
            st.stop()

    smooth_verts, dt = apply_smoothing(verts, faces, algo_key, int(iterations))
    disp = np.linalg.norm(smooth_verts - verts, axis=1)

    # Metrics (simple + demo-relevant)
    v0 = _poly_volume(verts, faces)
    v1 = _poly_volume(smooth_verts, faces)
    vol_delta_pct = ((v1 - v0) / (v0 + 1e-12)) * 100.0

    # ---------------- Layout ----------------
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        _kpi("Vertices", f"{len(verts):,}")
    with k2:
        _kpi("Triangles", f"{len(faces):,}")
    with k3:
        _kpi("Volume Δ", f"{vol_delta_pct:+.2f}%")
    with k4:
        _kpi("Avg disp", f"{float(disp.mean()):.2f} mm")
    with k5:
        _kpi("Time", f"{dt:.2f} s")

    st.caption(
        f"Selected: **{pid}** • **{mesh_target}**"
        + (f" • region **{region}**" if mesh_target == "Tumor (from mask)" else "")
        + f" • smoothing **{algo_label}**"
    )

    if show_mri:
        left, right = st.columns([2.25, 1.0], gap="large")
    else:
        left = st.container()
        right = None

    with left:
        cL, cR = st.columns(2, gap="medium")
        # Keep rendering stable with one good lighting preset and a standard camera.
        light_key = "soft"
        camera = "isometric"

        # Default if Advanced is collapsed (Streamlit only defines variables when widgets are created)
        plot_height = locals().get("plot_height", 560)
        enhance_detail = locals().get("enhance_detail", True)
        facet_shading = locals().get("facet_shading", False)
        geometry_style = locals().get("geometry_style", "Neutral")
        # Wireframe removed for a cleaner demo; keep disabled.
        show_wire = False

        # Precompute optional evidence signals
        curv_kind: Optional[Literal["mean", "gaussian"]] = "mean"  # single, stable default

        curv_o = curv_s = curv_delta = curv_signed = None
        need_curv = ("Curvature change" in view_mode) or (view_mode == "Geometry (single color)" and bool(enhance_detail))
        if need_curv:
            with st.spinner("Computing curvature…"):
                curv_o = compute_curvature(verts, faces, kind=(curv_kind or "mean"))
                curv_s = compute_curvature(smooth_verts, faces, kind=(curv_kind or "mean"))
                curv_delta = np.abs(curv_s - curv_o)
                curv_signed = (curv_s - curv_o)

        # Optional context surface mesh (shown behind both plots)
        ctx_verts = ctx_faces = None
        # Avoid drawing the same surface twice (WebGL z-fighting / flicker when rotating)
        show_context_effective = bool(show_context) and (mesh_target != "Head/brain surface (from MRI)")
        if show_context_effective and mri_path is not None:
            try:
                ctx_verts, ctx_faces = mri_path_to_surface_mesh(
                    mri_path,
                    downsample=3,
                    threshold_percentile=0.0,
                    sigma=0.85,
                )
            except Exception:
                ctx_verts = ctx_faces = None

        # Auto-show scales for metric/evidence modes
        show_colorbars = view_mode in {
            "Displacement (magnitude)",
            "Curvature change (|Δ|)",
            "Curvature change (signed)",
        }

        with cL:
            if view_mode == "Tumor labels" and mesh_target == "Tumor (from mask)":
                bins = _labels_to_bins(vlabels)
                fig_orig = mesh_single_figure(
                    verts,
                    faces,
                    intensity=bins,
                    colorscale=_label_colorscale_discrete(),
                    cmin=0,
                    cmax=3,
                    title="Original (tumor labels)",
                    show_wireframe=show_wire,
                    lighting_preset=light_key,
                    camera=camera.lower(),
                    showscale=False,
                    height=plot_height,
                    flatshading=bool(facet_shading),
                )
            elif view_mode == "Displacement (magnitude)":
                # IMPORTANT: displacement is defined *between* original and smoothed.
                # If we color both meshes by the same displacement array, they will look identical.
                # Instead, show a baseline (0) on the original and the true displacement on the smoothed.
                _, vmax = _robust_range(disp, 1, 99)
                fig_orig = mesh_single_figure(
                    verts,
                    faces,
                    intensity=np.zeros_like(disp, dtype=np.float32),
                    colorscale=_colorscale_magnitude(),
                    cmin=0.0,
                    cmax=vmax,
                    title="Original — displacement (baseline)",
                    show_wireframe=show_wire,
                    lighting_preset=light_key,
                    camera=camera.lower(),
                    showscale=False,
                    height=plot_height,
                    flatshading=bool(facet_shading),
                )
            elif view_mode.startswith("Curvature change"):
                # Keep the same scalar + same range across both plots so differences are readable.
                if view_mode.startswith("Curvature change (signed)"):
                    assert curv_signed is not None
                    finite = curv_signed[np.isfinite(curv_signed)]
                    vabs = float(np.percentile(np.abs(finite), 99)) if finite.size else 1.0
                    vabs = max(vabs, 1e-6)
                    fig_orig = mesh_single_figure(
                        verts,
                        faces,
                        intensity=np.zeros_like(curv_signed, dtype=np.float32),
                        colorscale=_colorscale_signed(),
                        cmin=-vabs,
                        cmax=vabs,
                        title="Original — Δ curvature (baseline)",
                        show_wireframe=show_wire,
                        lighting_preset=light_key,
                        camera=camera.lower(),
                        showscale=False,
                        height=plot_height,
                        flatshading=bool(facet_shading),
                    )
                else:
                    assert curv_delta is not None
                    # Δ magnitude is nonnegative; show baseline 0 on the original.
                    _, vmax = _robust_range(curv_delta, 1, 99)
                    fig_orig = mesh_single_figure(
                        verts,
                        faces,
                        intensity=np.zeros_like(curv_delta, dtype=np.float32),
                        colorscale=_colorscale_magnitude(),
                        cmin=0.0,
                        cmax=vmax,
                        title="Original — |Δ curvature| (baseline)",
                        show_wireframe=show_wire,
                        lighting_preset=light_key,
                        camera=camera.lower(),
                        showscale=False,
                        height=plot_height,
                        flatshading=bool(facet_shading),
                    )
            else:
                # Geometry view: optionally add subtle grayscale curvature shading so
                # surface complexity and smoothing differences are visible.
                if view_mode == "Geometry (single color)" and bool(enhance_detail) and curv_o is not None:
                    det_o = np.abs(np.asarray(curv_o, dtype=np.float32))
                    det_s = np.abs(np.asarray(curv_s, dtype=np.float32)) if curv_s is not None else det_o
                    det_all = np.concatenate([det_o[np.isfinite(det_o)], det_s[np.isfinite(det_s)]])
                    if det_all.size:
                        cmin_d, cmax_d = _robust_range(det_all, 5, 99)
                    else:
                        cmin_d, cmax_d = 0.0, 1.0
                    fig_orig = mesh_single_figure(
                        verts,
                        faces,
                        intensity=det_o,
                        colorscale=_detail_colorscale(geometry_style),
                        cmin=cmin_d,
                        cmax=cmax_d,
                        title="Original (geometry + detail shading)",
                        show_wireframe=show_wire,
                        lighting_preset=light_key,
                        camera=camera.lower(),
                        showscale=False,
                        height=plot_height,
                        flatshading=bool(facet_shading),
                    )
                else:
                    col_o, col_s = _geometry_colors(geometry_style)
                    fig_orig = mesh_single_figure(
                        verts,
                        faces,
                        color=col_o,
                        title="Original (geometry)",
                        show_wireframe=show_wire,
                        lighting_preset=light_key,
                        camera=camera.lower(),
                        showscale=False,
                        height=plot_height,
                        flatshading=bool(facet_shading),
                    )

            if ctx_verts is not None and ctx_faces is not None and len(ctx_verts) > 0 and len(ctx_faces) > 0:
                ctx_trace = go.Mesh3d(
                    x=ctx_verts[:, 0], y=ctx_verts[:, 1], z=ctx_verts[:, 2],
                    i=ctx_faces[:, 0], j=ctx_faces[:, 1], k=ctx_faces[:, 2],
                    color="rgba(100, 116, 139, 1.0)",
                    opacity=float(context_opacity),
                    lighting=_context_lighting(),
                    flatshading=False,
                    hoverinfo="skip",
                    showscale=False,
                )
                # Plotly does not allow arbitrary reassignment of fig.data when introducing new traces.
                # Add as a subtle overlay (low opacity) for context.
                fig_orig.add_trace(ctx_trace)
            st.plotly_chart(
                fig_orig,
                use_container_width=True,
                key=f"orig::{pid}::{mesh_target}::{region}::{view_mode}::{algo_key}::{int(iterations)}::{int(plot_height)}::{int(enhance_detail)}::{int(facet_shading)}",
            )

        with cR:
            if view_mode == "Tumor labels" and mesh_target == "Tumor (from mask)":
                bins = _labels_to_bins(vlabels)
                fig_smooth = mesh_single_figure(
                    smooth_verts,
                    faces,
                    intensity=bins,
                    colorscale=_label_colorscale_discrete(),
                    cmin=0,
                    cmax=3,
                    title=f"Smoothed ({algo_label}) — tumor labels",
                    show_wireframe=show_wire,
                    lighting_preset=light_key,
                    camera=camera.lower(),
                    showscale=False,
                    height=plot_height,
                    flatshading=bool(facet_shading),
                )
            elif view_mode == "Geometry (single color)":
                if bool(enhance_detail) and curv_s is not None and curv_o is not None:
                    det_o = np.abs(np.asarray(curv_o, dtype=np.float32))
                    det_s = np.abs(np.asarray(curv_s, dtype=np.float32))
                    det_all = np.concatenate([det_o[np.isfinite(det_o)], det_s[np.isfinite(det_s)]])
                    if det_all.size:
                        cmin_d, cmax_d = _robust_range(det_all, 5, 99)
                    else:
                        cmin_d, cmax_d = 0.0, 1.0
                    fig_smooth = mesh_single_figure(
                        smooth_verts,
                        faces,
                        intensity=det_s,
                        colorscale=_detail_colorscale(geometry_style),
                        cmin=cmin_d,
                        cmax=cmax_d,
                        title=f"Smoothed ({algo_label}) — geometry + detail shading",
                        show_wireframe=show_wire,
                        lighting_preset=light_key,
                        camera=camera.lower(),
                        showscale=False,
                        height=plot_height,
                        flatshading=bool(facet_shading),
                    )
                else:
                    col_o, col_s = _geometry_colors(geometry_style)
                    fig_smooth = mesh_single_figure(
                        smooth_verts,
                        faces,
                        color=col_s,
                        title=f"Smoothed ({algo_label}) — geometry",
                        show_wireframe=show_wire,
                        lighting_preset=light_key,
                        camera=camera.lower(),
                        showscale=False,
                        height=plot_height,
                        flatshading=bool(facet_shading),
                    )
            elif view_mode == "Displacement (magnitude)":
                # Displacement, but clamp range for readability
                _, vmax = _robust_range(disp, 1, 99)
                fig_smooth = mesh_single_figure(
                    smooth_verts,
                    faces,
                    intensity=disp,
                    colorscale=_colorscale_magnitude(),
                    cmin=0.0,
                    cmax=vmax,
                    title=f"Smoothed ({algo_label}) — displacement",
                    show_wireframe=show_wire,
                    lighting_preset=light_key,
                    camera=camera.lower(),
                    showscale=True,
                    colorbar_title="mm",
                    height=plot_height,
                    flatshading=bool(facet_shading),
                )
            elif view_mode.startswith("Curvature change (signed)"):
                assert curv_signed is not None
                finite = curv_signed[np.isfinite(curv_signed)]
                vabs = float(np.percentile(np.abs(finite), 99)) if finite.size else 1.0
                vabs = max(vabs, 1e-6)
                fig_smooth = mesh_single_figure(
                    smooth_verts,
                    faces,
                    intensity=curv_signed,
                    colorscale=_colorscale_signed(),
                    cmin=-vabs,
                    cmax=vabs,
                    title=f"Smoothed ({algo_label}) — Δ curvature (signed)",
                    show_wireframe=show_wire,
                    lighting_preset=light_key,
                    camera=camera.lower(),
                    showscale=True,
                    colorbar_title="Δκ",
                    height=plot_height,
                    flatshading=bool(facet_shading),
                )
            elif view_mode.startswith("Curvature change"):
                assert curv_delta is not None
                _, vmax = _robust_range(curv_delta, 1, 99)
                fig_smooth = mesh_single_figure(
                    smooth_verts,
                    faces,
                    intensity=curv_delta,
                    colorscale=_colorscale_magnitude(),
                    cmin=0.0,
                    cmax=vmax,
                    title=f"Smoothed ({algo_label}) — |Δ curvature|",
                    show_wireframe=show_wire,
                    lighting_preset=light_key,
                    camera=camera.lower(),
                    showscale=True,
                    colorbar_title="|Δκ|",
                    height=plot_height,
                    flatshading=bool(facet_shading),
                )
            else:
                # Fallback: displacement
                vmin, vmax = _robust_range(disp, 1, 99)
                fig_smooth = mesh_single_figure(
                    smooth_verts,
                    faces,
                    intensity=disp,
                    colorscale=_colorscale_magnitude(),
                    cmin=vmin,
                    cmax=vmax,
                    title=f"Smoothed ({algo_label}) — displacement",
                    show_wireframe=show_wire,
                    lighting_preset=light_key,
                    camera=camera.lower(),
                    showscale=True,
                    colorbar_title="mm",
                    height=plot_height,
                    flatshading=bool(facet_shading),
                )

            if ctx_verts is not None and ctx_faces is not None and len(ctx_verts) > 0 and len(ctx_faces) > 0:
                ctx_trace = go.Mesh3d(
                    x=ctx_verts[:, 0], y=ctx_verts[:, 1], z=ctx_verts[:, 2],
                    i=ctx_faces[:, 0], j=ctx_faces[:, 1], k=ctx_faces[:, 2],
                    color="rgba(100, 116, 139, 1.0)",
                    opacity=float(context_opacity),
                    lighting=_context_lighting(),
                    flatshading=False,
                    hoverinfo="skip",
                    showscale=False,
                )
                fig_smooth.add_trace(ctx_trace)
            st.plotly_chart(
                fig_smooth,
                use_container_width=True,
                key=f"smooth::{pid}::{mesh_target}::{region}::{view_mode}::{algo_key}::{int(iterations)}::{int(plot_height)}::{int(enhance_detail)}::{int(facet_shading)}",
            )

        st.caption("Rotate/zoom each view independently. Double-click inside a plot to reset.")

        # Evidence plots (kept in sync with the selected metric visualization)
        with st.expander("Evidence (distributions)", expanded=False):
            e1, e2 = st.columns(2)
            with e1:
                st.markdown("**Displacement distribution**")
                st.plotly_chart(
                    go.Figure(
                        data=[go.Histogram(x=disp, nbinsx=40, marker=dict(color=ACCENT, opacity=0.85))]
                    ).update_layout(
                        height=220,
                        margin=dict(l=0, r=0, t=10, b=0),
                        paper_bgcolor=APP_BG,
                        plot_bgcolor=CARD_BG,
                        xaxis_title="mm",
                        yaxis_title="count",
                        font=dict(color=TEXT),
                    ),
                    use_container_width=True,
                )
            with e2:
                if view_mode.startswith("Curvature change") and curv_delta is not None:
                    st.markdown("**|Δ curvature| distribution**")
                    st.plotly_chart(
                        go.Figure(
                            data=[go.Histogram(x=curv_delta, nbinsx=40, marker=dict(color="#2563eb", opacity=0.85))]
                        ).update_layout(
                            height=220,
                            margin=dict(l=0, r=0, t=10, b=0),
                            paper_bgcolor=APP_BG,
                            plot_bgcolor=CARD_BG,
                            xaxis_title="curvature",
                            yaxis_title="count",
                            font=dict(color=TEXT),
                        ),
                        use_container_width=True,
                    )
                else:
                    st.markdown("**Curvature evidence**")
                    st.caption("Switch Visualization to a curvature-change mode to see curvature distributions.")

    if right is not None:
        with right:
            st.markdown("### MRI slice")
            if mri_path is None:
                st.info("No MRI modality found for this case.")
            else:
                mvol, _ = load_nifti(mri_path)
                plane = st.radio("Plane", ["axial", "coronal", "sagittal"], horizontal=True)
                max_idx = (
                    mvol.shape[2] - 1
                    if plane == "axial"
                    else mvol.shape[1] - 1
                    if plane == "coronal"
                    else mvol.shape[0] - 1
                )
                idx = st.slider("Slice", 0, int(max_idx), int(max_idx // 2))
                st.plotly_chart(
                    mri_slice_figure(mvol, load_labels(files.mask)[0], plane, int(idx), show_overlay),
                    use_container_width=True,
                    key=f"mri::{pid}::{plane}::{int(idx)}::{int(show_overlay)}",
                )

    st.markdown(
        """
                <div style="margin-top: 1.2rem; color: rgba(15,23,42,0.45); font-size: 0.9rem;">
          CSCE 645 Geometric Modeling • Fall 2025 • Shubham Vikas Mhaske
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
