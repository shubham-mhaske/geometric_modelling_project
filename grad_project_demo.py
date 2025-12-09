import streamlit as st
import os
import numpy as np
import nibabel as nib
from skimage import measure
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import time
from scipy import sparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import algorithms
try:
    from src.algorithms.novel_algorithms import (
        geodesic_heat_smoothing,
        information_theoretic_smoothing,
        anisotropic_tensor_smoothing
    )
    from src.algorithms.smoothing import taubin_smoothing, laplacian_smoothing
except ImportError:
    st.error("Could not import algorithms. Make sure you are running this from the project root.")
    st.stop()

# Page config
st.set_page_config(
    page_title="Geometric Modelling - Novel Smoothing Algorithms",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for sleek design
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .sidebar .sidebar-content {
        background: #262730;
    }
    h1 {
        color: #ffffff;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
    h2, h3 {
        color: #fafafa;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #ff4b4b;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #ff6b6b;
    }
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stMetric {
        background-color: #1f2937;
        padding: 10px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Title and Intro
st.title("🧠 Geometric Modelling: Novel Mesh Smoothing Algorithms")

# Dynamic description based on data type
if 'data_type_selector' in st.session_state and st.session_state['data_type_selector'] == "CT (Hemorrhage)":
    st.markdown("""
**Graduate Course Project: CSCE 645 Geometric Modeling**  
*Author: Shubham Vikas Mhaske*

This application demonstrates **novel mesh smoothing algorithms** on **CT intracranial hemorrhage segmentations**.

**What you're visualizing:**
- 🩸 **Hemorrhage regions** extracted from expert radiologist annotations
- 📐 **3D mesh surfaces** generated from binary segmentation masks
- 🔬 **Smoothing effects** of different geometric algorithms on hemorrhage boundaries

**Novel algorithms** (Geodesic Heat, Anisotropic Tensor, Information-Theoretic) are compared against baseline methods to show how they preserve critical anatomical features while reducing mesh noise.
""")
else:
    st.markdown("""
**Graduate Course Project: CSCE 645 Geometric Modeling**  
*Author: Shubham Vikas Mhaske*

This application demonstrates **novel mesh smoothing algorithms** on **MRI brain tumor segmentations**.

**What you're visualizing:**
- 🧬 **Tumor regions** from BraTS (Brain Tumor Segmentation Challenge)
- 📐 **3D mesh surfaces** generated from segmentation masks
- 🔬 **Smoothing effects** of different geometric algorithms on tumor boundaries

These methods explore unexplored directions in geometric processing, specifically designed to preserve anatomical features while reducing noise.
""")

# Sidebar
st.sidebar.header("Configuration")
# Data Selection
st.sidebar.subheader("1. Data Selection")

# Data type selection
data_type = st.sidebar.radio(
    "Data Type",
    ["MRI (BraTS)", "CT (Hemorrhage)"],
    key='data_type_selector'
)

if data_type == "MRI (BraTS)":
    data_root = "data/data"
else:
    data_root = "data/ct_data/computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.3.1/masks"
    ct_scan_root = "data/ct_data/computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.3.1/ct_scans"

# Get list of subjects
subjects = []
if os.path.exists(data_root):
    if data_type == "MRI (BraTS)":
        subjects = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        subjects.sort()
        # Add Synthetic Data option if available
        if "Synthetic-Sphere" in subjects:
            subjects.remove("Synthetic-Sphere")
            subjects.append("Synthetic-Sphere") # Put at bottom
        if "Synthetic-Cube" in subjects:
            subjects.remove("Synthetic-Cube")
            subjects.append("Synthetic-Cube")
    else:
        subjects = [f.replace('.nii', '') for f in os.listdir(data_root) if f.endswith('.nii')]
        subjects.sort()

def on_subject_change():
    # Clear results when subject changes
    keys_to_clear = ['smoothed_verts', 'info', 'elapsed', 'brain_mesh', 'tumor_mesh', 'ct_mesh']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

selected_subject = st.sidebar.selectbox(
    "Select Subject", 
    subjects, 
    index=0 if subjects else None,
    on_change=on_subject_change,
    key='subject_selector'
)

# Tumor Region Selection / CT Hemorrhage Options
if data_type == "MRI (BraTS)":
    st.sidebar.subheader("2. Tumor Region")
    tumor_label = st.sidebar.selectbox(
        "Select Tumor Region",
        [1, 2, 4],
        format_func=lambda x: f"Label {x} (Tumor Sub-region)"
    )
    show_ct_context = False
    ct_window_level = None
    ct_window_width = None
else:
    st.sidebar.subheader("2. Hemorrhage Visualization")
    st.sidebar.info("📊 Visualizing intracranial hemorrhage segmentation from expert radiologist annotations.")
    show_ct_context = st.sidebar.checkbox("Show Brain Context (CT Scan)", value=False)
    if show_ct_context:
        ct_window_level = st.sidebar.slider("CT Window Level (HU)", -100, 100, 40, key='ct_level')
        ct_window_width = st.sidebar.slider("CT Window Width (HU)", 50, 300, 120, key='ct_width')
    else:
        ct_window_level = None
        ct_window_width = None
    tumor_label = None

mask_path = None
brain_path = None
ct_path = None

if selected_subject:
    try:
        if data_type == "MRI (BraTS)":
            subject_path = os.path.join(data_root, selected_subject)
            # Find mask file
            mask_files = [f for f in os.listdir(subject_path) if f.endswith('mask.nii.gz')]
            # Find brain file (t1n)
            brain_files = [f for f in os.listdir(subject_path) if f.endswith('t1n.nii.gz')]
            
            if mask_files:
                mask_path = os.path.join(subject_path, mask_files[0])
            else:
                st.sidebar.error("No mask file found for this subject.")
                
            if brain_files:
                brain_path = os.path.join(subject_path, brain_files[0])
            else:
                st.sidebar.warning("No T1n brain file found. Context visualization disabled.")
        else:
            # CT data - load hemorrhage mask
            ct_path = os.path.join(data_root, selected_subject + '.nii')
            if not os.path.exists(ct_path):
                st.sidebar.error(f"Hemorrhage mask not found: {ct_path}")
                ct_path = None
            
            # Optionally load CT scan for context
            if show_ct_context:
                ct_scan_path = os.path.join(ct_scan_root, selected_subject + '.nii')
                if os.path.exists(ct_scan_path):
                    brain_path = ct_scan_path
                else:
                    st.sidebar.warning("CT scan not found for context visualization.")
                    brain_path = None
            
    except Exception as e:
        st.sidebar.error(f"Error finding data: {e}")

# Algorithm Selection
st.sidebar.subheader("3. Algorithm Selection")
algorithm = st.sidebar.selectbox(
    "Select Smoothing Algorithm",
    [
        "Geodesic Heat Diffusion",
        "Anisotropic Tensor",
        "Information-Theoretic",
        "Taubin (Baseline)",
        "Laplacian (Baseline)"
    ]
)

# Parameters
st.sidebar.subheader("4. Parameters")
params = {}

with st.sidebar.expander("Algorithm Parameters", expanded=True):
    # Adjust defaults based on data type (CT hemorrhages are smaller, need gentler smoothing)
    is_ct = data_type == "CT (Hemorrhage)"
    
    if algorithm == "Geodesic Heat Diffusion":
        default_iters = 3 if is_ct else 5
        default_time = 0.5 if is_ct else 1.0
        params['iterations'] = st.slider("Iterations", 1, 20, default_iters, key='geo_iter')
        params['time_scale'] = st.slider("Time Scale", 0.1, 5.0, default_time, key='geo_time')
        params['feature_threshold'] = st.slider("Feature Threshold", 0.0, 1.0, 0.3, key='geo_feat')
        st.info("🔬 Uses heat diffusion to approximate geodesic distances for neighbor weighting.")
        if is_ct:
            st.caption("⚠️ CT meshes are smaller - using gentler defaults")

    elif algorithm == "Information-Theoretic":
        default_iters = 5 if is_ct else 10
        params['iterations'] = st.slider("Iterations", 1, 20, default_iters, key='info_iter')
        params['entropy_weight'] = st.slider("Entropy Weight", 0.0, 1.0, 0.5, key='info_weight')
        st.info("🔬 Minimizes entropy of curvature distribution to preserve features.")
        if is_ct:
            st.caption("⚠️ CT meshes are smaller - using gentler defaults")

    elif algorithm == "Anisotropic Tensor":
        default_iters = 5 if is_ct else 10
        default_time = 0.05 if is_ct else 0.1
        params['iterations'] = st.slider("Iterations", 1, 20, default_iters, key='aniso_iter')
        params['diffusion_time'] = st.slider("Diffusion Time", 0.01, 0.5, default_time, key='aniso_time')
        st.info("🔬 Uses diffusion tensors aligned with surface geometry.")
        if is_ct:
            st.caption("⚠️ CT meshes are smaller - using gentler defaults")

    elif algorithm == "Taubin (Baseline)":
        default_iters = 10 if is_ct else 20
        params['iterations'] = st.slider("Iterations", 1, 50, default_iters, key='taub_iter')
        params['mu'] = st.slider("Mu (Shrink)", 0.0, 1.0, 0.5, key='taub_mu')
        params['lambda_'] = st.slider("Lambda (Inflate)", 0.0, 1.0, 0.53, key='taub_lam')

    elif algorithm == "Laplacian (Baseline)":
        default_iters = 3 if is_ct else 5
        params['iterations'] = st.slider("Iterations", 1, 20, default_iters, key='lap_iter')
        params['weight'] = st.slider("Weight", 0.0, 1.0, 0.5, key='lap_weight')

# Visualization Options
st.sidebar.subheader("5. Visualization")
with st.sidebar.expander("Display Settings", expanded=True):
    viz_mode = st.radio(
        "Color Mode",
        ["Solid Color", "Mean Curvature", "Displacement (Smoothed Only)"]
    )

    view_mode = st.radio(
        "View Mode",
        ["Side-by-Side (Independent)", "Side-by-Side (Synced)", "Overlay Comparison"]
    )

    show_brain_context = st.checkbox("Show Brain Context", value=True)
    brain_opacity = st.slider("Brain Opacity", 0.0, 1.0, 0.1)


# Helper Functions
@st.cache_data
def load_mesh_from_nifti(filepath, label_idx=None, step_size=1, ct_window=None):
    """Load mesh from NIfTI file using Marching Cubes."""
    img = nib.load(filepath)
    data = img.get_fdata()
    
    if ct_window is not None:
        # CT scan with windowing for context visualization
        level, width = ct_window
        lower = level - width / 2
        upper = level + width / 2
        mask = (data >= lower) & (data <= upper)
    elif label_idx is not None:
        # MRI with specific label
        mask = data == label_idx
    else:
        # Binary mask (hemorrhage segmentation or general mask)
        mask = data > 0
        
    if not np.any(mask):
        return None, None
        
    verts, faces, _, _ = measure.marching_cubes(mask, level=0.5, step_size=step_size)
    affine = img.affine
    verts_homo = np.hstack([verts, np.ones((verts.shape[0], 1))])
    verts = (affine @ verts_homo.T).T[:, :3]
    return verts.astype(np.float32), faces.astype(np.int64)

def compute_mean_curvature(verts, faces):
    """Compute mean curvature for visualization."""
    N = len(verts)
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2],
                          faces[:, 1], faces[:, 2], faces[:, 0]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0],
                          faces[:, 0], faces[:, 1], faces[:, 2]])
    data = np.ones(len(rows))
    A = sparse.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    degrees = np.array(A.sum(axis=1)).flatten()
    D_inv = sparse.diags(1.0 / (degrees + 1e-10))
    W = D_inv @ A
    neighbor_avg = W @ verts
    laplacian = verts - neighbor_avg
    # Mean curvature is proportional to magnitude of Laplacian
    curvature = np.linalg.norm(laplacian, axis=1)
    return curvature

def compute_displacement(orig_verts, smoothed_verts):
    """Compute displacement magnitude."""
    return np.linalg.norm(smoothed_verts - orig_verts, axis=1)

def get_mesh_trace(verts, faces, color='gray', opacity=1.0, name='Mesh', 
                   intensity=None, colorscale='Viridis', cmin=None, cmax=None, 
                   show_scale=False, wireframe=False):
    """Helper to create a mesh trace."""
    if wireframe:
        # Create wireframe using Scatter3d lines
        # This is expensive for large meshes, so we might just use Mesh3d with specific settings
        # or just a subset of edges. For now, let's use Mesh3d with wireframe look if possible
        # Plotly Mesh3d doesn't support true wireframe easily. 
        # We'll use a trick: opacity 0.1 + contour lines if possible, or just points.
        # Better: Use Scatter3d for vertices if small, or just semi-transparent mesh.
        # Actually, for overlay, we want to see the difference.
        # Let's return a semi-transparent mesh for "wireframe" equivalent
        return go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=color, opacity=0.3, name=name
        )
    
    common_args = dict(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        opacity=opacity,
        name=name,
        flatshading=False,
        lighting=dict(ambient=0.4, diffuse=0.9, specular=0.5, roughness=0.1)
    )
    
    if intensity is not None:
        return go.Mesh3d(
            intensity=intensity,
            colorscale=colorscale,
            cmin=cmin, cmax=cmax,
            showscale=show_scale,
            **common_args
        )
    else:
        return go.Mesh3d(
            color=color,
            **common_args
        )

def plot_scene(tumor_verts, tumor_faces, brain_verts=None, brain_faces=None, 
               title="Scene", color_data=None, color_scale='Viridis', cmin=None, cmax=None,
               brain_opacity=0.1):
    """Create a high-quality Plotly 3D scene with brain context."""
    
    meshes = []
    
    # 1. Brain Context (if available)
    if brain_verts is not None and brain_faces is not None:
        meshes.append(get_mesh_trace(brain_verts, brain_faces, color='gray', opacity=brain_opacity, name='Brain Context'))
    
    # 2. Tumor Mesh
    if color_data is not None:
        meshes.append(get_mesh_trace(tumor_verts, tumor_faces, intensity=color_data, 
                                     colorscale=color_scale, cmin=cmin, cmax=cmax, 
                                     show_scale=True, name='Tumor'))
    else:
        meshes.append(get_mesh_trace(tumor_verts, tumor_faces, color='#ff4b4b', name='Tumor'))

    fig = go.Figure(data=meshes)
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20, color='white')),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor="rgba(0,0,0,0)",
        uirevision='constant' # Preserve camera state
    )
    return fig

def plot_synced_comparison(orig_verts, smoothed_verts, faces, brain_verts=None, brain_faces=None,
                          color_data_orig=None, color_data_smooth=None, 
                          cmin=None, cmax=None, brain_opacity=0.1, title_suffix=""):
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=(f"Original{title_suffix}", f"Smoothed{title_suffix}"),
        horizontal_spacing=0.05
    )
    
    # Brain Context (Shared)
    if brain_verts is not None and brain_faces is not None:
        brain_trace = get_mesh_trace(brain_verts, brain_faces, color='gray', opacity=brain_opacity, name='Brain Context')
        fig.add_trace(brain_trace, row=1, col=1)
        fig.add_trace(brain_trace, row=1, col=2)
        
    # Original Tumor
    if color_data_orig is not None:
        fig.add_trace(get_mesh_trace(orig_verts, faces, intensity=color_data_orig, cmin=cmin, cmax=cmax, show_scale=False, name='Original'), row=1, col=1)
    else:
        fig.add_trace(get_mesh_trace(orig_verts, faces, color='#ff4b4b', name='Original'), row=1, col=1)
        
    # Smoothed Tumor
    if color_data_smooth is not None:
        fig.add_trace(get_mesh_trace(smoothed_verts, faces, intensity=color_data_smooth, cmin=cmin, cmax=cmax, show_scale=True, name='Smoothed'), row=1, col=2)
    else:
        fig.add_trace(get_mesh_trace(smoothed_verts, faces, color='#ff4b4b', name='Smoothed'), row=1, col=2)
        
    # Sync Cameras
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode='data', camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        scene2=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode='data', camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor="rgba(0,0,0,0)",
        uirevision='constant' # Preserve camera state
    )
    
    # JavaScript for syncing cameras is not directly supported in Streamlit/Plotly Python easily without custom components,
    # but sharing the layout scene properties helps. 
    # Actually, Plotly subplots don't automatically sync cameras in Python.
    # However, putting them in the same figure is better than separate figures.
    
    return fig

def plot_overlay(orig_verts, smoothed_verts, faces, brain_verts=None, brain_faces=None, brain_opacity=0.1):
    meshes = []
    
    # Brain
    if brain_verts is not None and brain_faces is not None:
        meshes.append(get_mesh_trace(brain_verts, brain_faces, color='gray', opacity=brain_opacity, name='Brain Context'))
        
    # Original (Red Wireframe-ish - using low opacity)
    meshes.append(go.Mesh3d(
        x=orig_verts[:, 0], y=orig_verts[:, 1], z=orig_verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color='red', opacity=0.3, name='Original (Red)'
    ))
    
    # Smoothed (Blue Solid)
    meshes.append(go.Mesh3d(
        x=smoothed_verts[:, 0], y=smoothed_verts[:, 1], z=smoothed_verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color='blue', opacity=1.0, name='Smoothed (Blue)'
    ))
    
    fig = go.Figure(data=meshes)
    fig.update_layout(
        title=dict(text="Overlay Comparison (Red=Original, Blue=Smoothed)", font=dict(color='white')),
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor="rgba(0,0,0,0)",
        uirevision='constant' # Preserve camera state
    )
    return fig

def compute_metrics(original_verts, smoothed_verts, faces):
    """Compute quality metrics."""
    # 1. Volume Change
    def volume(v):
        v0 = v[faces[:, 0]]
        v1 = v[faces[:, 1]]
        v2 = v[faces[:, 2]]
        return abs(np.sum(v0 * np.cross(v1, v2)) / 6.0)
    
    orig_vol = volume(original_verts)
    new_vol = volume(smoothed_verts)
    vol_change = 100 * (new_vol - orig_vol) / orig_vol
    
    # 2. Smoothness (Laplacian magnitude) - normalized by mesh scale
    def smoothness(v):
        N = len(v)
        rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2],
                              faces[:, 1], faces[:, 2], faces[:, 0]])
        cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0],
                              faces[:, 0], faces[:, 1], faces[:, 2]])
        data = np.ones(len(rows))
        A = sparse.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
        degrees = np.array(A.sum(axis=1)).flatten()
        D_inv = sparse.diags(1.0 / (degrees + 1e-10))
        W = D_inv @ A
        neighbor_avg = W @ v
        laplacian = v - neighbor_avg
        laplacian_mag = np.linalg.norm(laplacian, axis=1)
        
        # Normalize by mesh scale (average edge length)
        edge_lengths = []
        for face in faces[:min(1000, len(faces))]:  # Sample for efficiency
            for i in range(3):
                v1, v2 = v[face[i]], v[face[(i+1)%3]]
                edge_lengths.append(np.linalg.norm(v2 - v1))
        avg_edge = np.mean(edge_lengths) if edge_lengths else 1.0
        
        # Return relative roughness (dimensionless)
        return np.mean(laplacian_mag) / (avg_edge + 1e-10)
    
    orig_smooth = smoothness(original_verts)
    new_smooth = smoothness(smoothed_verts)
    smooth_improvement = 100 * (orig_smooth - new_smooth) / (orig_smooth + 1e-10)
    
    # Add mesh quality metric
    def mesh_quality(v):
        # Measure surface smoothness via normal variation
        normals = []
        for face in faces:
            v0, v1, v2 = v[face[0]], v[face[1]], v[face[2]]
            normal = np.cross(v1 - v0, v2 - v0)
            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-10:
                normals.append(normal / norm_len)
        if not normals:
            return 0.0
        normals = np.array(normals)
        # Variance of normal vectors (lower = smoother)
        return np.std(normals)
    
    orig_quality = mesh_quality(original_verts)
    new_quality = mesh_quality(smoothed_verts)
    quality_improvement = 100 * (orig_quality - new_quality) / (orig_quality + 1e-10)
    
    return {
        "Volume Change (%)": f"{vol_change:.2f}%",
        "Smoothness Improvement (%)": f"{smooth_improvement:.2f}%",
        "Quality Improvement (%)": f"{quality_improvement:.2f}%",
        "Original Roughness": f"{orig_smooth:.4f}",
        "Smoothed Roughness": f"{new_smooth:.4f}",
        "Original Vertices": len(original_verts),
        "Faces": len(faces)
    }

# Main Execution
if mask_path or ct_path:
    # Load Data
    if data_type == "MRI (BraTS)" and mask_path:
        if 'brain_mesh' not in st.session_state and brain_path:
            with st.spinner("Loading Brain Context..."):
                # Use step_size=2 for brain to keep it lightweight
                st.session_state['brain_mesh'] = load_mesh_from_nifti(brain_path, step_size=2)
                
        if 'tumor_mesh' not in st.session_state:
            with st.spinner(f"Loading Tumor Region {tumor_label}..."):
                st.session_state['tumor_mesh'] = load_mesh_from_nifti(mask_path, label_idx=tumor_label)
        
        # Get meshes from state
        brain_verts, brain_faces = st.session_state.get('brain_mesh', (None, None))
        orig_verts, faces = st.session_state.get('tumor_mesh', (None, None))
    
    elif data_type == "CT (Hemorrhage)" and ct_path:
        # Load CT context if enabled
        if show_ct_context and brain_path:
            if 'brain_mesh' not in st.session_state:
                with st.spinner("Loading CT scan context..."):
                    st.session_state['brain_mesh'] = load_mesh_from_nifti(
                        brain_path, 
                        ct_window=(ct_window_level, ct_window_width),
                        step_size=3
                    )
            brain_verts, brain_faces = st.session_state.get('brain_mesh', (None, None))
        else:
            brain_verts, brain_faces = None, None
        
        # Load hemorrhage segmentation mask
        if 'ct_mesh' not in st.session_state:
            with st.spinner(f"Loading hemorrhage segmentation..."):
                st.session_state['ct_mesh'] = load_mesh_from_nifti(ct_path, step_size=1)
        
        # Get mesh from state
        orig_verts, faces = st.session_state.get('ct_mesh', (None, None))
    else:
        orig_verts, faces = None, None
        brain_verts, brain_faces = None, None
    
    if orig_verts is None:
        st.warning(f"No tumor region found for Label {tumor_label} in this subject.")
    else:
        # Run Algorithm
        if st.sidebar.button("Run Smoothing", type="primary"):
            with st.spinner(f"Running {algorithm}..."):
                start_time = time.time()
                
                if algorithm == "Geodesic Heat Diffusion":
                    smoothed_verts, info = geodesic_heat_smoothing(orig_verts, faces, **params)
                elif algorithm == "Information-Theoretic":
                    smoothed_verts, info = information_theoretic_smoothing(orig_verts, faces, **params)
                elif algorithm == "Anisotropic Tensor":
                    smoothed_verts, info = anisotropic_tensor_smoothing(orig_verts, faces, **params)
                elif algorithm == "Taubin (Baseline)":
                    smoothed_verts = taubin_smoothing(orig_verts, faces, iterations=params['iterations'], mu_val=params['mu'], lambda_val=params['lambda_'])
                    info = {"method": "Taubin"}
                elif algorithm == "Laplacian (Baseline)":
                    smoothed_verts = laplacian_smoothing(orig_verts, faces, iterations=params['iterations'], lambda_val=params['weight'])
                    info = {"method": "Laplacian"}
                
                elapsed = time.time() - start_time
                
                # Store results in session state
                st.session_state['smoothed_verts'] = smoothed_verts
                st.session_state['info'] = info
                st.session_state['elapsed'] = elapsed
                
        # Display Results if available
        if 'smoothed_verts' in st.session_state:
            smoothed_verts = st.session_state['smoothed_verts']
            
            # Metrics
            st.subheader("Evaluation Metrics")
            metrics = compute_metrics(orig_verts, smoothed_verts, faces)
            metrics['Processing Time'] = f"{st.session_state['elapsed']:.4f} s"
            
            # Display metrics in columns
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Volume Change", metrics["Volume Change (%)"], delta_color="inverse", help="Negative = shrinkage, Positive = expansion")
                st.metric("Smoothness Improvement", metrics["Smoothness Improvement (%)"], help="Higher = more smoothing applied")
                st.metric("Quality Improvement", metrics["Quality Improvement (%)"], help="Based on normal vector consistency")
            with col2:
                st.metric("Processing Time", metrics["Processing Time"])
                st.metric("Mesh Complexity", f"{metrics['Original Vertices']:,} verts, {metrics['Faces']:,} faces")
                with st.expander("📊 Roughness Details"):
                    st.write(f"**Original Roughness:** {metrics['Original Roughness']}")
                    st.write(f"**Smoothed Roughness:** {metrics['Smoothed Roughness']}")
                    st.caption("Lower roughness = smoother surface. Normalized by mesh scale.")
            
            # Performance interpretation
            smooth_improv = float(metrics["Smoothness Improvement (%)"].replace('%', ''))
            quality_improv = float(metrics["Quality Improvement (%)"].replace('%', ''))
            
            if smooth_improv < 5 and quality_improv < 5:
                st.warning("⚠️ **Low smoothing effect detected.** This can happen with small/simple meshes or conservative parameters. Try increasing iterations or adjusting algorithm-specific parameters.")
            elif smooth_improv > 50:
                st.info("ℹ️ **High smoothing effect.** The mesh has been significantly smoothed. Check that important features are preserved in the visualization.")
            
            st.markdown("---")

            # Visualization
            # Prepare visualization data
            color_data_orig = None
            color_data_smooth = None
            cmin, cmax = None, None
            title_suffix = ""

            if viz_mode == "Mean Curvature":
                color_data_orig = compute_mean_curvature(orig_verts, faces)
                color_data_smooth = compute_mean_curvature(smoothed_verts, faces)
                # Normalize scales
                cmin = min(color_data_orig.min(), color_data_smooth.min())
                cmax = max(color_data_orig.max(), color_data_smooth.max())
                cmax = np.percentile(color_data_orig, 95)
                title_suffix = " (Curvature)"
                
            elif viz_mode == "Displacement (Smoothed Only)":
                color_data_smooth = compute_displacement(orig_verts, smoothed_verts)
                title_suffix = " (Displacement)"
            
            # Context for plotting
            b_verts = brain_verts if show_brain_context else None
            b_faces = brain_faces if show_brain_context else None
            
            if view_mode == "Side-by-Side (Independent)":
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Mesh")
                    st.plotly_chart(
                        plot_scene(orig_verts, faces, b_verts, b_faces, 
                                   title=f"Original{title_suffix}", 
                                   color_data=color_data_orig, 
                                   color_scale='Jet', 
                                   cmin=cmin, cmax=cmax,
                                   brain_opacity=brain_opacity), 
                        use_container_width=True
                    )
                    
                with col2:
                    st.subheader("Smoothed Mesh")
                    st.plotly_chart(
                        plot_scene(smoothed_verts, faces, b_verts, b_faces, 
                                   title=f"Smoothed ({algorithm}){title_suffix}", 
                                   color_data=color_data_smooth, 
                                   color_scale='Jet', 
                                   cmin=cmin, cmax=cmax,
                                   brain_opacity=brain_opacity), 
                        use_container_width=True
                    )
            
            elif view_mode == "Side-by-Side (Synced)":
                st.plotly_chart(
                    plot_synced_comparison(orig_verts, smoothed_verts, faces, b_verts, b_faces,
                                          color_data_orig, color_data_smooth,
                                          cmin, cmax, brain_opacity, title_suffix),
                    use_container_width=True
                )
                
            elif view_mode == "Overlay Comparison":
                st.plotly_chart(
                    plot_overlay(orig_verts, smoothed_verts, faces, b_verts, b_faces, brain_opacity),
                    use_container_width=True
                )
            
            # Algorithm Info
            with st.expander("Algorithm Details"):
                st.json(st.session_state['info'])
        else:
            # Show just the original mesh if no smoothing run yet
            st.info("Click 'Run Smoothing' to see the results.")
            
            b_verts = brain_verts if show_brain_context else None
            b_faces = brain_faces if show_brain_context else None
            
            st.plotly_chart(
                plot_scene(orig_verts, faces, b_verts, b_faces, 
                           title="Original Mesh (Preview)", 
                           brain_opacity=brain_opacity), 
                use_container_width=True
            )

else:
    st.info("Please select a subject from the sidebar to begin.")
