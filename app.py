"""
Brain Tumor 3D Mesh Analysis Demo
Interactive MRI + 3D Mesh Visualization with Tumor Overlays
"""

import os
import io
import json
import glob
import tempfile
import time

import streamlit as st
import nibabel as nib
import numpy as np
from skimage import measure
import pyvista as pv
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from src.algorithms import smoothing, simplification, metrics
from src.algorithms.processing import map_labels_to_vertices, coarsen_label_volume
from src.algorithms.novel_algorithms_efficient import (
    geodesic_heat_smoothing,
    anisotropic_tensor_smoothing,
    information_theoretic_smoothing
)

# PyVista configuration
pv.OFF_SCREEN = True
pv.global_theme.background = 'black'
pv.global_theme.font.color = 'white'

# ============================================================================
# DATA LOADING HELPERS
# ============================================================================

def find_patient_data(search_dir='data'):
    """Find all patient folders with complete MRI+mask data."""
    patients = {}
    
    # Search in data/data folder for BraTS cases
    data_folder = os.path.join(search_dir, 'data')
    if os.path.exists(data_folder):
        for patient_folder in glob.glob(os.path.join(data_folder, 'BraTS-*')):
            patient_id = os.path.basename(patient_folder)
            
            # Find all files for this patient
            files = {
                't1c': None, 't1n': None, 't2f': None, 't2w': None, 'mask': None
            }
            
            for f in os.listdir(patient_folder):
                fpath = os.path.join(patient_folder, f)
                if 't1c.nii' in f:
                    files['t1c'] = fpath
                elif 't1n.nii' in f and 'voided' not in f:
                    files['t1n'] = fpath
                elif 't2f.nii' in f:
                    files['t2f'] = fpath
                elif 't2w.nii' in f:
                    files['t2w'] = fpath
                elif 'mask.nii' in f:
                    files['mask'] = fpath
            
            # Only include if we have mask
            if files['mask']:
                patients[patient_id] = files
    
    return patients


def load_mri_volume(file_path):
    """Load MRI volume and return data + affine."""
    try:
        nii = nib.load(file_path)
        data = nii.get_fdata()
        return data, nii.affine, nii.header.get_zooms()
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return None, None, None


@st.cache_data
def load_patient_data(_patient_files):
    """Load all MRI sequences and mask for a patient."""
    patient_files = _patient_files
    data = {}
    
    for modality, fpath in patient_files.items():
        if fpath:
            vol, affine, zooms = load_mri_volume(fpath)
            if vol is not None:
                data[modality] = {
                    'volume': vol,
                    'affine': affine,
                    'zooms': zooms,
                    'path': fpath
                }
    
    return data


@st.cache_data
def process_nifti_to_mesh(_mask_data, tumor_region='all', _cache_key=None):
    """Convert tumor mask to 3D mesh."""
    try:
        label_volume = np.asarray(_mask_data['volume'], dtype=np.int16)
        zooms = _mask_data['zooms']
        affine = _mask_data['affine']
        
        # Select tumor region
        region_labels = {
            'all': [1, 2, 4],
            'core': [1, 4],
            'enhancing': [4],
            'edema': [2],
            'necrotic': [1]
        }
        labels_to_use = region_labels.get(tumor_region, [1, 2, 4])
        binary_mask = np.isin(label_volume, labels_to_use).astype(np.uint8)
        
        if binary_mask.sum() == 0:
            binary_mask = (label_volume != 0).astype(np.uint8)
            if binary_mask.sum() == 0:
                raise ValueError("No tumor found in mask")
        
        # Smooth mask
        from scipy.ndimage import gaussian_filter, binary_closing
        binary_mask = binary_closing(binary_mask, iterations=1).astype(np.uint8)
        smooth_mask = gaussian_filter(binary_mask.astype(float), sigma=0.5)
        
        # Generate mesh
        verts, faces, _, _ = measure.marching_cubes(smooth_mask, level=0.5, spacing=tuple(zooms[:3]))
        
        # Keep largest component
        if len(verts) > 100:
            verts, faces = keep_largest_component(verts, faces)
        
        faces_padded = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).astype(np.int64)
        mesh = pv.PolyData(verts, faces_padded)
        
        # Map vertices to tumor labels
        vertex_labels = map_vertex_to_tumor_region(verts, label_volume, affine, zooms[:3])
        
        return mesh, verts, faces, vertex_labels, label_volume
        
    except Exception as e:
        raise RuntimeError(f"Error processing mask: {e}")


def keep_largest_component(verts, faces):
    """Keep only largest connected mesh component."""
    try:
        import trimesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            largest = max(components, key=lambda m: len(m.vertices))
            return largest.vertices, largest.faces
        return verts, faces
    except:
        return verts, faces


def map_vertex_to_tumor_region(verts, label_volume, affine, zooms):
    """Map vertices to tumor region labels."""
    try:
        voxel_coords = (verts / np.array(zooms)).astype(int)
        voxel_coords = np.clip(voxel_coords, 0, np.array(label_volume.shape) - 1)
        labels = label_volume[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]]
        return labels.astype(int)
    except:
        return np.zeros(len(verts), dtype=int)


@st.cache_data
def create_brain_surface_mesh(_mri_data, _cache_key=None):
    """Create brain surface mesh from MRI volume (skull-stripped)."""
    try:
        volume = _mri_data['volume']
        zooms = _mri_data['zooms']
        
        # Simple brain extraction: threshold + morphology
        from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_erosion, binary_dilation
        
        # Normalize and threshold
        vol_norm = (volume - volume.min()) / (volume.max() - volume.min() + 1e-10)
        threshold = np.percentile(vol_norm[vol_norm > 0], 10) if np.any(vol_norm > 0) else 0.1
        brain_mask = vol_norm > threshold
        
        # Morphological operations to get smooth brain surface
        brain_mask = binary_erosion(brain_mask, iterations=2)
        brain_mask = binary_fill_holes(brain_mask)
        brain_mask = binary_dilation(brain_mask, iterations=2)
        
        # Smooth the mask
        smooth_mask = gaussian_filter(brain_mask.astype(float), sigma=1.5)
        
        # Generate mesh
        verts, faces, _, _ = measure.marching_cubes(smooth_mask, level=0.5, spacing=tuple(zooms[:3]))
        
        # Keep largest component (the brain)
        if len(verts) > 100:
            verts, faces = keep_largest_component(verts, faces)
        
        faces_padded = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).astype(np.int64)
        mesh = pv.PolyData(verts, faces_padded)
        
        return mesh, verts, faces
        
    except Exception as e:
        st.warning(f"Could not create brain surface: {e}")
        return None, None, None


# ============================================================================
# MRI VISUALIZATION
# ============================================================================

def create_mri_slice_view(volume, slice_idx, plane='axial', tumor_mask=None, overlay_alpha=0.4):
    """Create MRI slice with optional tumor overlay."""
    
    # Get slice based on plane
    if plane == 'axial':
        mri_slice = volume[:, :, slice_idx]
        if tumor_mask is not None:
            tumor_slice = tumor_mask[:, :, slice_idx]
    elif plane == 'coronal':
        mri_slice = volume[:, slice_idx, :]
        if tumor_mask is not None:
            tumor_slice = tumor_mask[:, slice_idx, :]
    else:  # sagittal
        mri_slice = volume[slice_idx, :, :]
        if tumor_mask is not None:
            tumor_slice = tumor_mask[slice_idx, :, :]
    
    # Normalize MRI
    mri_slice = mri_slice.T  # Flip for correct orientation
    vmin, vmax = np.percentile(mri_slice[mri_slice > 0], [2, 98]) if np.any(mri_slice > 0) else (0, 1)
    
    fig = go.Figure()
    
    # Add MRI as grayscale
    fig.add_trace(go.Heatmap(
        z=mri_slice,
        colorscale='gray',
        zmin=vmin,
        zmax=vmax,
        showscale=False,
        hoverinfo='skip'
    ))
    
    # Add tumor overlay if provided
    if tumor_mask is not None:
        tumor_slice = tumor_slice.T
        tumor_rgba = np.zeros((*tumor_slice.shape, 4))
        
        # Color code: 1=red (necrotic), 2=blue (edema), 4=yellow (enhancing)
        tumor_rgba[tumor_slice == 1] = [1, 0, 0, overlay_alpha]  # Necrotic - Red
        tumor_rgba[tumor_slice == 2] = [0, 0.5, 1, overlay_alpha]  # Edema - Blue
        tumor_rgba[tumor_slice == 4] = [1, 1, 0, overlay_alpha]  # Enhancing - Yellow
        
        # Create custom colorscale for overlay
        if np.any(tumor_slice > 0):
            fig.add_trace(go.Heatmap(
                z=tumor_slice,
                colorscale=[
                    [0.0, 'rgba(0,0,0,0)'],
                    [0.1, 'rgba(255,0,0,0.4)'],      # Necrotic
                    [0.4, 'rgba(0,128,255,0.4)'],    # Edema
                    [1.0, 'rgba(255,255,0,0.4)']     # Enhancing
                ],
                showscale=False,
                hoverinfo='skip'
            ))
    
    fig.update_layout(
        xaxis=dict(visible=False, showticklabels=False),
        yaxis=dict(visible=False, showticklabels=False, scaleanchor='x'),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='#0a0a1a',
        plot_bgcolor='#0a0a1a',
        height=400
    )
    
    return fig


def create_multimodal_view(patient_data, slice_idx, plane='axial', show_tumor=True):
    """Create side-by-side view of multiple MRI modalities."""
    
    modalities_order = ['t1n', 't1c', 't2w', 't2f']
    available_modalities = [m for m in modalities_order if m in patient_data]
    
    n_cols = len(available_modalities)
    if n_cols == 0:
        return None
    
    fig = make_subplots(
        rows=1, cols=n_cols,
        subplot_titles=[m.upper() for m in available_modalities],
        horizontal_spacing=0.02
    )
    
    tumor_mask = patient_data['mask']['volume'] if show_tumor and 'mask' in patient_data else None
    
    for idx, modality in enumerate(available_modalities, 1):
        volume = patient_data[modality]['volume']
        
        # Get slice
        if plane == 'axial':
            mri_slice = volume[:, :, slice_idx].T
            tumor_slice = tumor_mask[:, :, slice_idx].T if tumor_mask is not None else None
        elif plane == 'coronal':
            mri_slice = volume[:, slice_idx, :].T
            tumor_slice = tumor_mask[:, slice_idx, :].T if tumor_mask is not None else None
        else:  # sagittal
            mri_slice = volume[slice_idx, :, :].T
            tumor_slice = tumor_mask[slice_idx, :, :].T if tumor_mask is not None else None
        
        # Normalize
        vmin, vmax = np.percentile(mri_slice[mri_slice > 0], [2, 98]) if np.any(mri_slice > 0) else (0, 1)
        
        # Add MRI
        fig.add_trace(go.Heatmap(
            z=mri_slice,
            colorscale='gray',
            zmin=vmin,
            zmax=vmax,
            showscale=False,
            hoverinfo='skip'
        ), row=1, col=idx)
        
        # Add tumor overlay
        if tumor_slice is not None and np.any(tumor_slice > 0):
            fig.add_trace(go.Heatmap(
                z=tumor_slice,
                colorscale=[
                    [0.0, 'rgba(0,0,0,0)'],
                    [0.25, 'rgba(255,0,0,0.5)'],
                    [0.5, 'rgba(0,128,255,0.5)'],
                    [1.0, 'rgba(255,255,0,0.5)']
                ],
                showscale=False,
                hoverinfo='skip'
            ), row=1, col=idx)
    
    # Update all axes
    for i in range(1, n_cols + 1):
        fig.update_xaxes(visible=False, showticklabels=False, row=1, col=i)
        fig.update_yaxes(visible=False, showticklabels=False, scaleanchor=f'x{i}', row=1, col=i)
    
    fig.update_layout(
        paper_bgcolor='#0a0a1a',
        plot_bgcolor='#0a0a1a',
        margin=dict(l=0, r=0, t=40, b=0),
        height=400,
        font=dict(color='white', size=12)
    )
    
    return fig


# ============================================================================
# 3D MESH VISUALIZATION
# ============================================================================

def create_stunning_3d_mesh(verts, faces,
                              vertex_labels=None, displacement=None,
                              color_mode='tumor_region', title=None, show_wireframe=False):
    """Create a single, visually appealing 3D mesh for the tumor."""

    def _scene_bounds(v, padding=0.05):
        # Compute a cube bounding box to avoid clipping when rotating/zooming
        center = v.mean(axis=0)
        ranges = v.max(axis=0) - v.min(axis=0)
        max_range = ranges.max()
        pad = max_range * padding
        half = max_range / 2 + pad
        return center, half
    
    traces = []
    # Softer, balanced lighting to avoid harsh highlights and z-fighting feel
    lighting = dict(ambient=0.55, diffuse=0.65, specular=0.12, roughness=0.7, fresnel=0.05)
    
    # Add tumor mesh (solid, colored)
    intensity = None
    colorscale = None
    showscale = False
    tumor_color = '#ff6b6b'
    
    if color_mode == 'tumor_region' and vertex_labels is not None and len(vertex_labels) > 0:
        intensity = vertex_labels.astype(float)
        colorscale = [
            [0.0, '#555577'],
            [0.33, '#ef4444'],
            [0.66, '#3b82f6'],
            [1.0, '#fbbf24']
        ]
        showscale = False
        
    elif color_mode == 'displacement' and displacement is not None and len(displacement) > 0:
        if displacement.max() > 0:
            intensity = (displacement - displacement.min()) / (displacement.max() - displacement.min() + 1e-10)
            colorscale = [[0.0, '#1e3a5f'], [0.5, '#667eea'], [1.0, '#f5576c']]
            showscale = True
    
    # Tuned lighting for more realistic shading
    lighting = dict(ambient=0.35, diffuse=0.8, specular=0.25, roughness=0.45, fresnel=0.2)

    tumor_kwargs = dict(
        x=verts[:, 0], 
        y=verts[:, 1], 
        z=verts[:, 2],
        i=faces[:, 0], 
        j=faces[:, 1], 
        k=faces[:, 2],
        opacity=1.0,  # fully opaque to avoid seeing inner surfaces
        lighting=lighting,
        lightposition=dict(x=900, y=1400, z=1800),
        flatshading=False,
        name='Tumor',
        hoverinfo='skip'
    )
    
    if intensity is not None:
        tumor_kwargs['intensity'] = intensity
        tumor_kwargs['colorscale'] = colorscale
        tumor_kwargs['showscale'] = showscale
    else:
        tumor_kwargs['color'] = tumor_color
    
    traces.append(go.Mesh3d(**tumor_kwargs))

    if show_wireframe:
        traces.append(go.Scatter3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            mode='lines',
            line=dict(color='rgba(255, 255, 255, 0.2)', width=1),
            hoverinfo='none',
            name='Wireframe'
        ))

    fig = go.Figure(data=traces)
    
    center, half = _scene_bounds(verts)

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, showbackground=False, showgrid=False,
                      range=[center[0]-half, center[0]+half]),
            yaxis=dict(visible=False, showbackground=False, showgrid=False,
                      range=[center[1]-half, center[1]+half]),
            zaxis=dict(visible=False, showbackground=False, showgrid=False,
                      range=[center[2]-half, center[2]+half]),
            bgcolor='#0a0a1a',
            aspectmode='cube',
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                projection=dict(type='orthographic')
            )
        ),
        paper_bgcolor='#0a0a1a',
        plot_bgcolor='#0a0a1a',
        margin=dict(l=0, r=0, t=30 if title else 0, b=0),
        title=dict(
            text=title,
            font=dict(color='white', size=14),
            x=0.5,
            xanchor='center'
        ) if title else None,
        showlegend=False,
        height=550
    )
    
    return fig


def create_side_by_side_3d(tumor_orig_verts, tumor_orig_faces,
                           tumor_proc_verts, tumor_proc_faces,
                           vertex_labels=None, displacement=None):
    """Side-by-side comparison of original vs processed with brain context."""
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=['Original Tumor', 'Smoothed Tumor'],
        horizontal_spacing=0.02
    )
    
    # Shared lighting tuned for more realistic shading
    lighting = dict(ambient=0.35, diffuse=0.8, specular=0.25, roughness=0.45, fresnel=0.2)
    
    # Column 1: Original
    fig.add_trace(go.Mesh3d(
        x=tumor_orig_verts[:, 0], y=tumor_orig_verts[:, 1], z=tumor_orig_verts[:, 2],
        i=tumor_orig_faces[:, 0], j=tumor_orig_faces[:, 1], k=tumor_orig_faces[:, 2],
        intensity=vertex_labels.astype(float),
        colorscale=[
            [0.0, '#555577'],
            [0.33, '#ef4444'],
            [0.66, '#3b82f6'],
            [1.0, '#fbbf24']
        ],
        opacity=0.95,
        lighting=lighting, lightposition=dict(x=1200, y=1200, z=1500),
        flatshading=False, hoverinfo='skip'
    ), row=1, col=1)
    
    # Column 2: Processed
    # Color processed tumor by displacement if available
    if displacement is not None and len(displacement) > 0 and displacement.max() > 0:
        intensity = (displacement - displacement.min()) / (displacement.max() - displacement.min() + 1e-10)
        colorscale = [[0.0, '#1e3a5f'], [0.5, '#667eea'], [1.0, '#f5576c']]
        
        fig.add_trace(go.Mesh3d(
            x=tumor_proc_verts[:, 0], y=tumor_proc_verts[:, 1], z=tumor_proc_verts[:, 2],
            i=tumor_proc_faces[:, 0], j=tumor_proc_faces[:, 1], k=tumor_proc_faces[:, 2],
            intensity=intensity, colorscale=colorscale,
            opacity=1.0,
            lighting=lighting, lightposition=dict(x=900, y=1400, z=1800), flatshading=False,
            showscale=False, hoverinfo='skip'
        ), row=1, col=2)
    else:
        fig.add_trace(go.Mesh3d(
            x=tumor_proc_verts[:, 0], y=tumor_proc_verts[:, 1], z=tumor_proc_verts[:, 2],
            i=tumor_proc_faces[:, 0], j=tumor_proc_faces[:, 1], k=tumor_proc_faces[:, 2],
            color='#38ef7d', opacity=1.0,
            lighting=lighting, lightposition=dict(x=900, y=1400, z=1800),
            flatshading=False, hoverinfo='skip'
        ), row=1, col=2)
    
    # Shared bounds to prevent clipping/vanishing during interaction
    def _scene_bounds(v, padding=0.05):
        center = v.mean(axis=0)
        ranges = v.max(axis=0) - v.min(axis=0)
        max_range = ranges.max()
        pad = max_range * padding
        half = max_range / 2 + pad
        return center, half
    center, half = _scene_bounds(np.vstack([tumor_orig_verts, tumor_proc_verts]))

    camera = dict(
        eye=dict(x=1.5, y=1.5, z=1.2),
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        projection=dict(type='orthographic')
    )
    scene_common = dict(
        xaxis=dict(visible=False, showbackground=False, range=[center[0]-half, center[0]+half]),
        yaxis=dict(visible=False, showbackground=False, range=[center[1]-half, center[1]+half]),
        zaxis=dict(visible=False, showbackground=False, range=[center[2]-half, center[2]+half]),
        bgcolor='#0a0a1a',
        aspectmode='cube',
        aspectratio=dict(x=1, y=1, z=1),
        camera=camera
    )
    fig.update_layout(scene=scene_common, scene2=scene_common)
    
    fig.update_layout(
        paper_bgcolor='#0a0a1a',
        plot_bgcolor='#0a0a1a',
        margin=dict(l=0, r=0, t=30, b=0),
        height=550,
        font=dict(color='white', size=12),
        showlegend=False
    )
    
    return fig


# ============================================================================
# STYLING
# ============================================================================

def inject_enhanced_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif !important; }
    
    #MainMenu, footer, header { visibility: hidden; }
    
    .main .block-container {
        padding: 1.5rem 2.5rem;
        max-width: 100%;
    }
    
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3a 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a3a 100%) !important;
        border-right: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    h1, h2, h3, h4 {
        color: white !important;
        font-weight: 600 !important;
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #f093fb 50%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        text-align: center;
        color: rgba(255,255,255,0.5);
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: rgba(30, 30, 60, 0.6);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38ef7d;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: rgba(255,255,255,0.5);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }
    
    .section-header {
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
    }
    
    .legend-item {
        display: inline-block;
        margin: 0.3rem 0.8rem;
        font-size: 0.85rem;
        color: rgba(255,255,255,0.7);
    }
    
    .legend-box {
        display: inline-block;
        width: 16px;
        height: 16px;
        margin-right: 0.5rem;
        border-radius: 4px;
        vertical-align: middle;
    }
    
    div[data-testid="stExpander"] {
        background: rgba(30, 30, 60, 0.4);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
    }

    .stSelectbox label, .stSlider label, .stRadio label, .stCheckbox label {
        color: rgba(255,255,255,0.9) !important;
        font-weight: 500 !important;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(30, 30, 60, 0.4);
        border-radius: 8px 8px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: rgba(255, 255, 255, 0.7);
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(102, 126, 234, 0.2) !important;
        color: white !important;
        border-bottom: 2px solid #667eea !important;
    }
    
    /* Metric Card Hover Effect */
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
        transition: all 0.3s ease;
    }
    
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="Brain Tumor 3D Analysis",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    inject_enhanced_css()
    
    # Title
    st.markdown('<h1 class="hero-title">🧠 Brain Tumor MRI + 3D Mesh Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">An interactive tool for visualizing and analyzing 3D brain tumor meshes from MRI data.</p>', unsafe_allow_html=True)

    with st.sidebar:
        with st.expander("ℹ️ Project Info", expanded=False):
            st.markdown("""
            **Geometric Modeling Project**
            
            This application demonstrates advanced mesh smoothing algorithms applied to brain tumor segmentation data.
            
            **Features:**
            *   Multi-modal MRI visualization
            *   Real-time 3D mesh generation
            *   Comparative smoothing analysis
            *   Quantitative metrics
            """)
            
        with st.expander("📖 How to Use", expanded=False):
            st.markdown("""
            1.  **Select Patient:** Choose a case from the dropdown.
            2.  **Configure:** Select tumor region and smoothing algorithm.
            3.  **Analyze:** Use the tabs to switch between Dashboard, 3D Analysis, and MRI views.
            4.  **Export:** Download results from the Export section.
            """)

    # ========================================================================
    # SIDEBAR
    # ========================================================================
    with st.sidebar:
        st.markdown("### 📁 Patient Selection")
        
        patients = find_patient_data('data')
        
        if not patients:
            st.error("No patient data found in data/ folder")
            st.stop()
        
        patient_ids = sorted(patients.keys())
        selected_patient = st.selectbox(
            "Choose Patient",
            patient_ids,
            format_func=lambda x: f"🧠 {x}"
        )
        
        st.markdown("---")
        st.markdown("### 🎯 Tumor Region")
        tumor_region = st.selectbox(
            "Region",
            ["all", "core", "enhancing", "edema", "necrotic"],
            format_func=lambda x: {
                'all': '🧠 Whole Tumor',
                'core': '🔴 Tumor Core',
                'enhancing': '✨ Enhancing',
                'edema': '💧 Edema',
                'necrotic': '⚫ Necrotic'
            }[x]
        )
        
        st.markdown("---")
        st.markdown("### 🔬 Smoothing Algorithm")
        
        algorithms = {
            'None': 'No smoothing',
            'Laplacian': 'Classic uniform',
            'Taubin': 'Volume preserving',
            'Geodesic Heat': '🔥 Curvature-adaptive',
            'Anisotropic Tensor': '📐 82% better volume',
            'Info-Theoretic': '🧮 Entropy-based'
        }
        
        selected_algo = st.selectbox("Algorithm", list(algorithms.keys()), index=4)
        st.caption(algorithms[selected_algo])
        
        if selected_algo != 'None':
            iterations = st.slider("Iterations", 1, 30, 10)
        else:
            iterations = 0
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        
        show_tumor_overlay = st.checkbox("Show tumor overlay on MRI", value=True)
        show_wireframe = st.checkbox("Show mesh wireframe", value=False)
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    
    patient_files = patients[selected_patient]
    
    with st.spinner(f"Loading {selected_patient}..."):
        patient_data = load_patient_data(patient_files)
    
    if 'mask' not in patient_data:
        st.error("No tumor mask found for this patient")
        st.stop()
    
    # Process mesh
    cache_key = f"{selected_patient}_{tumor_region}"
    
    with st.spinner("Generating 3D mesh..."):
        mesh, verts, faces, vertex_labels, label_volume = process_nifti_to_mesh(
            patient_data['mask'], tumor_region, _cache_key=cache_key
        )
    
    original_verts = verts.copy()
    original_faces = faces.copy()
    original_volume = float(mesh.volume)
    
    # ========================================================================
    # APPLY SMOOTHING
    # ========================================================================
    
    processed_verts = original_verts.copy()
    processing_time = 0.0
    
    if selected_algo != 'None' and iterations > 0:
        with st.spinner(f"Applying {selected_algo}..."):
            start_time = time.time()
            
            if selected_algo == 'Laplacian':
                processed_verts = smoothing.laplacian_smoothing(processed_verts, original_faces, iterations)
            elif selected_algo == 'Taubin':
                processed_verts = smoothing.taubin_smoothing(processed_verts, original_faces, iterations)
            elif selected_algo == 'Geodesic Heat':
                processed_verts, _ = geodesic_heat_smoothing(processed_verts, original_faces, iterations=iterations)
            elif selected_algo == 'Anisotropic Tensor':
                processed_verts, _ = anisotropic_tensor_smoothing(processed_verts, original_faces, iterations=iterations)
            elif selected_algo == 'Info-Theoretic':
                processed_verts, _ = information_theoretic_smoothing(processed_verts, original_faces, iterations=iterations)
            
            processing_time = time.time() - start_time
    
    # Calculate metrics
    faces_padded = np.hstack([np.full((original_faces.shape[0], 1), 3, dtype=np.int64), original_faces]).astype(np.int64)
    processed_mesh = pv.PolyData(processed_verts, faces_padded)
    processed_volume = float(processed_mesh.volume)
    volume_change = ((processed_volume - original_volume) / original_volume) * 100
    
    displacement = np.linalg.norm(processed_verts - original_verts, axis=1) if processed_verts.shape == original_verts.shape else np.zeros(len(processed_verts))
    
    # ========================================================================
    # METRICS ROW
    # ========================================================================
    
    st.markdown('<div class="section-header">📊 Mesh Statistics</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(original_faces):,}</div>
            <div class="metric-label">Triangles</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(original_verts):,}</div>
            <div class="metric-label">Vertices</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        color = '#38ef7d' if abs(volume_change) < 0.1 else '#f59e0b' if abs(volume_change) < 0.5 else '#ef4444'
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {color};">{volume_change:+.2f}%</div>
            <div class="metric-label">Volume Δ</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{displacement.mean():.2f}</div>
            <div class="metric-label">Avg Disp (mm)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{processing_time:.2f}s</div>
            <div class="metric-label">Time</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ========================================================================
    # MAIN VISUALIZATION
    # ========================================================================
    
    st.markdown('<div class="section-header">🎯 Visualization & Analysis</div>', unsafe_allow_html=True)
    
    # Create tabs for different views
    tab_dashboard, tab_3d, tab_mri, tab_cross, tab_methodology = st.tabs([
        "📊 Dashboard", "🎨 3D Analysis", "📷 MRI Viewer", "🔪 Cross-Section", "📚 Methodology"
    ])
    
    # --- TAB 1: DASHBOARD (Integrated View) ---
    with tab_dashboard:
        # Tumor legend
        st.markdown("""
        <div style="text-align: center; margin: 1rem 0;">
            <span class="legend-item">
                <span class="legend-box" style="background: #ef4444;"></span>Necrotic Core (Label 1)
            </span>
            <span class="legend-item">
                <span class="legend-box" style="background: #3b82f6;"></span>Edema (Label 2)
            </span>
            <span class="legend-item">
                <span class="legend-box" style="background: #fbbf24;"></span>Enhancing Tumor (Label 4)
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Layout: MRI slices on top, 3D meshes on bottom
        st.markdown("#### 📷 MRI Slices (Multi-Modal)")
        
        # Get middle slice
        if patient_data:
            sample_vol = next(iter([v for k, v in patient_data.items() if k != 'mask'])).get('volume')
            if sample_vol is not None:
                mid_slice = sample_vol.shape[2] // 2
                
                slice_idx = st.slider("Slice Index", 0, sample_vol.shape[2] - 1, mid_slice, key='main_slider')
                
                # Multi-modal view
                fig_mri = create_multimodal_view(patient_data, slice_idx, plane='axial', show_tumor=show_tumor_overlay)
                if fig_mri:
                    st.plotly_chart(fig_mri, use_container_width=True, key="dashboard_mri")
        
        st.markdown("#### 🎨 3D Tumor Mesh")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("**Original Mesh**")
            fig_orig = create_stunning_3d_mesh(
                original_verts, original_faces,
                color_mode='tumor region',
                vertex_labels=vertex_labels,
                show_wireframe=show_wireframe
            )
            st.plotly_chart(fig_orig, use_container_width=True, key="dashboard_orig")
        
        with col_right:
            st.markdown(f"**{selected_algo} Smoothed**" if selected_algo != 'None' else "**No Smoothing Applied**")
            fig_proc = create_stunning_3d_mesh(
                processed_verts, original_faces,
                color_mode='displacement' if selected_algo != 'None' else 'tumor region',
                displacement=displacement if selected_algo != 'None' else None,
                vertex_labels=vertex_labels,
                show_wireframe=show_wireframe
            )
            st.plotly_chart(fig_proc, use_container_width=True, key="dashboard_proc")
    
    # --- TAB 2: 3D ANALYSIS ---
    with tab_3d:
        st.markdown("#### 🎨 Detailed 3D Mesh Analysis")
        
        if selected_algo != 'None':
            fig_comparison = create_side_by_side_3d(
                original_verts, original_faces,
                processed_verts, original_faces,
                vertex_labels=vertex_labels,
                displacement=displacement
            )
            st.plotly_chart(fig_comparison, use_container_width=True, key="3d_comparison")
        else:
            fig_single = create_stunning_3d_mesh(
                original_verts, original_faces,
                color_mode='tumor_region',
                vertex_labels=vertex_labels,
                show_wireframe=show_wireframe
            )
            st.plotly_chart(fig_single, use_container_width=True, key="3d_single")
            
        st.info("💡 Tip: Use the mouse to rotate, zoom, and pan the 3D model. Double-click to reset view.")

    # --- TAB 3: MRI VIEWER ---
    with tab_mri:
        st.markdown("#### 📷 Multi-Modal MRI with Tumor Overlay")
        
        if patient_data:
            sample_vol = next(iter([v for k, v in patient_data.items() if k != 'mask'])).get('volume')
            if sample_vol is not None:
                plane = st.radio("Plane", ["axial", "coronal", "sagittal"], horizontal=True)
                # Plane-specific max index to avoid blank slices
                if plane == 'axial':
                    max_idx = sample_vol.shape[2] - 1
                elif plane == 'coronal':
                    max_idx = sample_vol.shape[1] - 1
                else:
                    max_idx = sample_vol.shape[0] - 1
                default_idx = max_idx // 2 if max_idx > 0 else 0
                slice_idx = st.slider("Slice Index", 0, max_idx, default_idx, key=f"slice_{plane}_tab")
                
                fig_mri = create_multimodal_view(patient_data, slice_idx, plane=plane, show_tumor=show_tumor_overlay)
                if fig_mri:
                    st.plotly_chart(fig_mri, use_container_width=True, key="mri_viewer")

    # --- TAB 4: CROSS-SECTION ---
    with tab_cross:
        st.markdown("#### 🔪 Cross-section Viewer")
        st.markdown("Visualizing the intersection of the 3D mesh with the MRI volume.")
        if patient_data and 't1c' in patient_data:
            # Render off-screen and display as an image to avoid relying on st.pyvista_chart
            p = pv.Plotter(window_size=[800, 600], off_screen=True)

            pv_mesh = pv.PolyData(processed_verts, faces_padded)
            p.add_mesh(pv_mesh, color='red', opacity=0.4)

            vol = pv.wrap(patient_data['t1c']['volume'])
            p.add_mesh_slice(vol, cmap='gray')

            screenshot = p.show(screenshot=True, auto_close=True)
            if screenshot is not None:
                st.image(screenshot, caption="PyVista cross-section", use_column_width=True)
            else:
                st.info("Unable to render PyVista cross-section screenshot.")
                
    # --- TAB 5: METHODOLOGY ---
    with tab_methodology:
        st.markdown("### 📚 Smoothing Algorithms Explained")
        
        st.markdown("#### 1. Laplacian Smoothing")
        st.markdown("""
        The most basic smoothing algorithm. It moves each vertex to the average position of its neighbors.
        $$ v_i \leftarrow v_i + \lambda \sum_{j \in N(i)} (v_j - v_i) $$
        * **Pros:** Simple, fast.
        * **Cons:** Causes shrinkage (volume loss).
        """)
        
        st.markdown("#### 2. Taubin Smoothing")
        st.markdown("""
        Also known as "signal processing on meshes". It alternates between shrinking (Laplacian) and expanding steps to preserve volume.
        * **Pros:** Preserves volume better than Laplacian.
        * **Cons:** Can still distort shapes if parameters aren't tuned.
        """)
        
        st.markdown("#### 3. Geodesic Heat Smoothing")
        st.markdown("""
        Uses the heat diffusion equation on the mesh surface.
        * **Pros:** Mathematically robust, preserves geometric features.
        * **Cons:** Computationally expensive.
        """)
        
        st.markdown("#### 4. Anisotropic Tensor Smoothing")
        st.markdown("""
        Smooths differently in different directions (e.g., along curvature but not across edges).
        * **Pros:** Preserves sharp features (edges).
        * **Cons:** Complex implementation.
        """)
        
        st.markdown("#### 5. Information-Theoretic Smoothing")
        st.markdown("""
        Minimizes an entropy-based energy function.
        * **Pros:** Novel approach, good for noisy data.
        * **Cons:** Experimental.
        """)
    
    # ========================================================================
    # EXPORT
    # ========================================================================
    
    with st.expander("📥 Export Options"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
                processed_mesh.save(tmp.name)
                with open(tmp.name, "rb") as f:
                    st.download_button(
                        "📥 Download STL",
                        data=f.read(),
                        file_name=f"{selected_patient}_mesh.stl",
                        mime="application/octet-stream"
                    )
        
        with col2:
            with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tmp:
                processed_mesh.save(tmp.name)
                with open(tmp.name, "rb") as f:
                    st.download_button(
                        "📥 Download OBJ",
                        data=f.read(),
                        file_name=f"{selected_patient}_mesh.obj",
                        mime="application/octet-stream"
                    )
        
        with col3:
            # Export metrics as JSON
            metrics_data = {
                'patient_id': selected_patient,
                'algorithm': selected_algo,
                'iterations': iterations,
                'triangles': len(original_faces),
                'vertices': len(original_verts),
                'volume_change_percent': float(volume_change),
                'avg_displacement_mm': float(displacement.mean()),
                'processing_time_sec': float(processing_time)
            }
            st.download_button(
                "📥 Download Metrics (JSON)",
                data=json.dumps(metrics_data, indent=2),
                file_name=f"{selected_patient}_metrics.json",
                mime="application/json"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: rgba(255,255,255,0.3); padding: 1rem;">
        <div>CSCE 645 Geometric Modeling • Fall 2025</div>
        <div style="font-size: 0.85rem; margin-top: 0.3rem;">Shubham Vikas Mhaske • Professor John Keyser</div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
