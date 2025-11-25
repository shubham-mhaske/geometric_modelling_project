import os
import io
import json
import glob
import tempfile

import streamlit as st
import nibabel as nib
import numpy as np
from skimage import measure
import pyvista as pv
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
import matplotlib.pyplot as plt
import pandas as pd
from src.algorithms import smoothing, simplification, metrics
from src.algorithms.processing import map_labels_to_vertices, coarsen_label_volume
from src.ml import get_ml_optimizer

# Ensure PyVista does not try to create native windows
pv.OFF_SCREEN = True

# --- Helper utilities ---
def compute_aspect_ratios(verts, faces):
    """Compute triangle aspect ratio for each triangle as (max_edge / min_edge)."""
    if faces is None or faces.size == 0:
        return np.array([])
    
    # faces is Nx3 indices into verts
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)
    max_e = np.maximum(np.maximum(e0, e1), e2)
    min_e = np.minimum(np.minimum(e0, e1), e2)
    # avoid division by zero
    min_e = np.where(min_e == 0, 1e-12, min_e)
    return max_e / min_e

def find_local_nifti_files(search_dir='data'):
    """Return sorted list of .nii/.nii.gz files under search_dir."""
    patterns = [os.path.join(search_dir, '**', '*.nii'), os.path.join(search_dir, '**', '*.nii.gz')]
    matches = []
    for pat in patterns:
        matches.extend(glob.glob(pat, recursive=True))
    # unique and sorted
    return sorted(list(dict.fromkeys(matches)))


@st.cache_data
def process_nifti_to_mesh(file_path_or_bytes):
    """
    Accept either a filesystem path or a bytes-like object (e.g. uploaded file).
    Returns (pyvista_mesh, aspect_ratios_array, verts, faces, label_volume, affine)
    """
    # Load nibabel image from path or bytes
    try:
        if isinstance(file_path_or_bytes, (bytes, bytearray)):
            img = nib.Nifti1Image.from_bytes(file_path_or_bytes) if hasattr(nib.Nifti1Image, 'from_bytes') else None
            if img is None:
                # fallback: write to temp file
                tmp = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
                tmp.write(file_path_or_bytes)
                tmp.flush()
                tmp.close()
                nii_img = nib.load(tmp.name)
                os.unlink(tmp.name)
                    # Try an interactive Plotly Mesh3d viewer in the browser; fallback to matplotlib if not available
            else:
                nii_img = img
        elif hasattr(file_path_or_bytes, 'read'):
            # file-like object (Streamlit upload)
            data = file_path_or_bytes.read()
            tmp = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
            tmp.write(data)
            tmp.flush()
            tmp.close()
            nii_img = nib.load(tmp.name)
            os.unlink(tmp.name)
        else:
            # assume path
            nii_img = nib.load(file_path_or_bytes)

        data = nii_img.get_fdata()
        label_volume = np.asarray(data, dtype=np.int16)
        # combine labels 1,2,4 into a single binary mask
        binary_mask = np.isin(label_volume, [1, 2, 4]).astype(np.uint8)
        # If those exact labels are not present, fall back to any non-zero voxel as mask
        if binary_mask.sum() == 0:
            try:
                # show a lightweight note in the UI (if available)
                st.warning("Labels 1/2/4 not found — falling back to non-zero mask for this file.")
            except Exception:
                pass
            binary_mask = (label_volume != 0).astype(np.uint8)

        # get voxel spacing (z,y,x) or (x,y,z) depending on header; marching_cubes expects spacing=(x,y,z)
        # nibabel header.get_zooms() returns spacing in the image axes order; we pass spacing=(dz, dy, dx)
        zooms = nii_img.header.get_zooms()[:3]
        # scikit-image marching_cubes expects spacing ordered like the array axes: (z_spacing, y_spacing, x_spacing)
        spacing = tuple(zooms)

        # run marching cubes on binary mask
        verts, faces, normals, values = measure.marching_cubes(binary_mask, level=0.5, spacing=spacing)

        # faces from marching_cubes are Nx3 indices; pyvista expects faces as [3, i0, i1, i2, 3, ...]
        faces_padded = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).astype(np.int64)

        mesh = pv.PolyData(verts, faces_padded)

        aspect_ratios = compute_aspect_ratios(verts, faces)

        # return verts and faces as well to enable non-VTK rendering paths
        return mesh, aspect_ratios, verts, faces, label_volume, nii_img.affine
    except Exception as e:
        # Provide a clear error back to Streamlit/UI
        raise RuntimeError(f"Error processing NIfTI to mesh: {e}")


def calculate_metrics(mesh, aspect_ratios):
    """Return surface_area (float), volume (float), and aspect_ratios array."""
    try:
        surface_area = float(mesh.area)
    except Exception:
        surface_area = float('nan')

    try:
        volume = float(mesh.volume)
    except Exception:
        # fallback: compute mesh volume via tetrahedralization (approx) or set nan
        volume = float('nan')

    return surface_area, volume, aspect_ratios


# --- Streamlit UI ---
st.set_page_config(page_title="CSCE 645: Brain Tumor 3D Analysis — Update 2", layout='wide')

st.title("🧠 Brain Tumor 3D Analysis — Complete Pipeline")
st.markdown(
    """
    High-fidelity mesh improvement for MRI-derived anatomical models.  
    **Features:** Laplacian & Taubin smoothing • QEM simplification • Hausdorff distance • Volume tracking
    """
)

with st.sidebar:
    st.header("Controls")

    # File input: uploader or pick from local data/
    uploaded = st.file_uploader("Upload a BraTS segmentation (.nii or .nii.gz)", type=['nii', 'gz', 'nii.gz'])

    local_files = find_local_nifti_files('data')
    local_choice = None
    if local_files:
        options = ['-- none --'] + local_files
        # If the user has not uploaded a file, auto-select the first local sample for a demo
        default_index = 1 if uploaded is None else 0
        default_index = min(default_index, len(options) - 1)
        local_choice = st.selectbox("Or choose a local sample file (from data/)", options, index=default_index)
        if uploaded is None and local_choice != '-- none --':
            st.info(f"Auto-selected sample: {os.path.basename(local_choice)}")

    st.markdown("---")
    st.subheader("Processing Options")
    
    # ML-based parameter optimization
    use_ml_optimizer = st.checkbox("🤖 ML-Optimized Parameters", value=False, 
                                   help="Use AI to predict optimal smoothing settings")
    
    if use_ml_optimizer:
        st.info("AI will analyze mesh and recommend parameters automatically")
        processing_algo = None  # Will be set by ML
        iterations = None
    else:
        processing_algo = st.selectbox("Smoothing Algorithm", ['None', 'Laplacian', 'Taubin'])
        iterations = st.slider("Smoothing Iterations", 0, 50, 10, help="Number of smoothing iterations")

    semantic_smoothing_requested = st.checkbox(
        "Semantic Smoothing",
        value=False,
        help="Respect tissue boundaries by reducing smoothing across different labels."
    )
    
    apply_simplification = st.checkbox("Apply QEM Simplification", value=False)
    if apply_simplification:
        target_reduction = st.slider("Reduction %", 0, 95, 50, help="Percentage of triangles to remove") / 100.0
    else:
        target_reduction = 0.0
    
    st.markdown("---")
    st.subheader("Analysis Options")
    show_comparison = st.checkbox("Side-by-Side Comparison", value=True)
    track_volume = st.checkbox("Track Volume Over Iterations", value=False)
    compute_hausdorff = st.checkbox("Compute Hausdorff Distance", value=True)
    
    st.markdown("---")
    batch_mode = st.checkbox("Batch Processing Mode", value=False, help="Process all local files")
    if batch_mode:
        st.info("Batch mode: Will process all files in data/ folder")

# Determine which file to use
selected_path = None
uploaded_file_obj = None
if uploaded is not None:
    uploaded_file_obj = uploaded
elif local_choice and local_choice != '-- none --':
    selected_path = local_choice

if selected_path is None and uploaded_file_obj is None:
    st.info("Please upload a BraTS segmentation (.nii.gz) or choose a sample file from the sidebar to begin.")
    st.stop()

# Batch processing mode
if batch_mode and len(local_files) > 1:
    st.header("📦 Batch Processing Results")
    
    batch_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, file_path in enumerate(local_files):
        status_text.text(f"Processing {idx+1}/{len(local_files)}: {os.path.basename(file_path)}")
        progress_bar.progress((idx + 1) / len(local_files))
        
        try:
            # Process mesh
            mesh, aspect_ratios, verts, faces, label_volume, affine = process_nifti_to_mesh(file_path)
            orig_vol = float(mesh.volume) if hasattr(mesh, 'volume') else 0
            orig_tris = mesh.n_faces_strict if hasattr(mesh, 'n_faces_strict') else mesh.n_cells

            vertex_labels = None
            if semantic_smoothing_requested:
                semantic_volume = coarsen_label_volume(label_volume)
                vertex_labels = map_labels_to_vertices(semantic_volume, affine, verts)
            
            # Apply smoothing
            if processing_algo == 'Laplacian':
                verts = smoothing.laplacian_smoothing(verts, faces, iterations, vertex_labels=vertex_labels)
            elif processing_algo == 'Taubin':
                verts = smoothing.taubin_smoothing(verts, faces, iterations, vertex_labels=vertex_labels)
            
            # Apply simplification
            if apply_simplification and target_reduction > 0:
                verts, faces = simplification.qem_simplification(verts, faces, target_reduction)
            
            # Reconstruct
            faces_padded = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).astype(np.int64)
            new_mesh = pv.PolyData(verts, faces_padded)
            new_vol = float(new_mesh.volume) if hasattr(new_mesh, 'volume') else 0
            new_tris = new_mesh.n_faces_strict if hasattr(new_mesh, 'n_faces_strict') else new_mesh.n_cells
            
            vol_change = ((new_vol - orig_vol) / orig_vol) * 100 if orig_vol > 0 else 0
            
            batch_results.append({
                'File': os.path.basename(file_path),
                'Original Triangles': orig_tris,
                'Processed Triangles': new_tris,
                'Reduction %': ((orig_tris - new_tris) / orig_tris * 100) if orig_tris > 0 else 0,
                'Original Volume (mm³)': orig_vol,
                'Processed Volume (mm³)': new_vol,
                'Volume Change %': vol_change
            })
        except Exception as e:
            batch_results.append({
                'File': os.path.basename(file_path),
                'Original Triangles': 'Error',
                'Processed Triangles': 'Error',
                'Reduction %': 'Error',
                'Original Volume (mm³)': 'Error',
                'Processed Volume (mm³)': 'Error',
                'Volume Change %': str(e)
            })
    
    status_text.text("✅ Batch processing complete!")
    
    # Display results table
    df_results = pd.DataFrame(batch_results)
    st.dataframe(df_results, use_container_width=True)
    
    # Download CSV
    csv = df_results.to_csv(index=False)
    st.download_button(
        label="📥 Download Results (CSV)",
        data=csv,
        file_name="batch_processing_results.csv",
        mime="text/csv"
    )
    
    st.stop()

# Single file processing mode
st.header("🔬 Single File Analysis")

# Process file
with st.spinner("Processing NIfTI and generating mesh..."):
    try:
        if uploaded_file_obj is not None:
            mesh, aspect_ratios, verts, faces, segmentation_volume, nifti_affine = process_nifti_to_mesh(uploaded_file_obj)
        else:
            mesh, aspect_ratios, verts, faces, segmentation_volume, nifti_affine = process_nifti_to_mesh(selected_path)
        
        # Store original for all comparisons
        original_verts = verts.copy()
        original_faces = faces.copy()
        original_mesh = mesh

        semantic_volume = coarsen_label_volume(segmentation_volume)
        vertex_labels = map_labels_to_vertices(semantic_volume, nifti_affine, original_verts)
        
    except Exception as e:
        st.error(f"Failed to process the file: {e}")
        st.stop()

# Get original stats
try:
    original_triangles = mesh.n_faces_strict
except Exception:
    original_triangles = getattr(mesh, 'n_cells', 0)

if mesh is None or original_triangles == 0:
    st.warning("Marching cubes produced no mesh. Ensure the segmentation contains labels 1, 2, or 4.")
    st.stop()

original_surface_area, original_volume, _ = calculate_metrics(mesh, aspect_ratios)

nonzero_vertex_labels = vertex_labels[vertex_labels > 0]
has_multiple_tissues = np.unique(nonzero_vertex_labels).size > 1
semantic_auto_forced = bool(use_ml_optimizer and has_multiple_tissues)
semantic_smoothing_active = semantic_smoothing_requested or semantic_auto_forced
active_vertex_labels = vertex_labels if semantic_smoothing_active else None

# ML Optimization: Predict parameters if enabled
ml_prediction = None
if use_ml_optimizer:
    with st.spinner("🤖 AI analyzing mesh characteristics..."):
        try:
            ml_opt = get_ml_optimizer('models/smoothing_optimizer.pth')
            ml_prediction = ml_opt.predict(original_verts, original_faces)
            
            processing_algo = ml_prediction['algorithm']
            iterations = ml_prediction['iterations']
            
            # Show ML predictions in an expander
            with st.expander("🔍 AI Recommendations", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Algorithm", ml_prediction['algorithm'])
                with col2:
                    st.metric("Iterations", ml_prediction['iterations'])
                with col3:
                    st.metric("Confidence", f"{ml_prediction['confidence']:.0%}")
                
                st.caption(f"Lambda: {ml_prediction['lambda']:.3f} | "
                          f"Based on {original_verts.shape[0]:,} vertices, {original_faces.shape[0]:,} faces")

                if semantic_auto_forced:
                    st.info("Semantic smoothing enforced (multiple tissue labels detected).")
        except Exception as e:
            st.warning(f"ML optimizer not available: {e}. Using heuristics.")
            # Fallback to simple heuristic
            if original_triangles > 50000:
                processing_algo = 'Taubin'
                iterations = 20
            else:
                processing_algo = 'Laplacian'
                iterations = 15

# Volume tracking for iterative smoothing
volume_history = []
if track_volume and processing_algo != 'None' and iterations > 0:
    with st.spinner("Tracking volume over iterations..."):
        temp_verts = original_verts.copy()
        for i in range(iterations + 1):
            if i == 0:
                volume_history.append((0, original_volume))
            else:
                if processing_algo == 'Laplacian':
                    temp_verts = smoothing.laplacian_smoothing(
                        temp_verts, original_faces, 1, vertex_labels=active_vertex_labels
                    )
                elif processing_algo == 'Taubin':
                    temp_verts = smoothing.taubin_smoothing(
                        temp_verts, original_faces, 1, vertex_labels=active_vertex_labels
                    )
                
                # Compute volume
                faces_padded = np.hstack([np.full((original_faces.shape[0], 1), 3, dtype=np.int64), original_faces]).astype(np.int64)
                temp_mesh = pv.PolyData(temp_verts, faces_padded)
                try:
                    vol = float(temp_mesh.volume)
                except:
                    vol = float('nan')
                volume_history.append((i, vol))

# Apply processing
processed_verts = original_verts.copy()
processed_faces = original_faces.copy()

if processing_algo != 'None':
    if processing_algo == 'Laplacian':
        with st.spinner(f"Applying Laplacian Smoothing ({iterations} iterations)..."):
            processed_verts = smoothing.laplacian_smoothing(
                processed_verts, processed_faces, iterations, vertex_labels=active_vertex_labels
            )
            
    elif processing_algo == 'Taubin':
        with st.spinner(f"Applying Taubin Smoothing ({iterations} iterations)..."):
            processed_verts = smoothing.taubin_smoothing(
                processed_verts, processed_faces, iterations, vertex_labels=active_vertex_labels
            )

# Apply simplification
if apply_simplification and target_reduction > 0:
    with st.spinner(f"Applying QEM simplification ({int(target_reduction*100)}% reduction)..."):
        processed_verts, processed_faces = simplification.qem_simplification(
            processed_verts, processed_faces, target_reduction
        )

# Reconstruct final mesh
faces_padded = np.hstack([np.full((processed_faces.shape[0], 1), 3, dtype=np.int64), processed_faces]).astype(np.int64)
mesh = pv.PolyData(processed_verts, faces_padded)
aspect_ratios = compute_aspect_ratios(processed_verts, processed_faces)

# Get processed stats
try:
    processed_triangles = mesh.n_faces_strict
except Exception:
    processed_triangles = getattr(mesh, 'n_cells', 0)

# Calculate Displacement (Heatmap Data) - need to match vertex counts
if processed_verts.shape[0] == original_verts.shape[0]:
    displacement = np.linalg.norm(processed_verts - original_verts, axis=1)
else:
    # If vertex count changed (simplification), displacement not applicable
    displacement = np.zeros(processed_verts.shape[0])

mesh["Displacement (mm)"] = displacement

# Compute metrics
surface_area, volume, aspect_ratios = calculate_metrics(mesh, aspect_ratios)

# Compute Hausdorff distance if requested
hausdorff_dist = None
if compute_hausdorff and processing_algo != 'None':
    with st.spinner("Computing Hausdorff distance..."):
        hausdorff_dist = metrics.hausdorff_distance(original_verts, processed_verts)

# Compute volume change
volume_change_pct = metrics.compute_volume_change_percent(original_volume, volume)

# Layout: left visualization, right metrics
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("3D Visualization")
    
    # Visualization mode
    if show_comparison and processing_algo != 'None':
        # Side-by-side comparison
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            subplot_titles=('Original Mesh', f'Processed Mesh ({processing_algo})')
        )
        
        # Original mesh
        fig.add_trace(
            go.Mesh3d(
                x=original_verts[:, 0], y=original_verts[:, 1], z=original_verts[:, 2],
                i=original_faces[:, 0], j=original_faces[:, 1], k=original_faces[:, 2],
                color='lightcoral',
                lighting=dict(ambient=0.5, diffuse=0.6, specular=0.2),
                flatshading=False,
                name='Original',
                showscale=False
            ),
            row=1, col=1
        )
        
        # Processed mesh
        fig.add_trace(
            go.Mesh3d(
                x=processed_verts[:, 0], y=processed_verts[:, 1], z=processed_verts[:, 2],
                i=processed_faces[:, 0], j=processed_faces[:, 1], k=processed_faces[:, 2],
                color='gold',
                lighting=dict(ambient=0.5, diffuse=0.6, specular=0.2),
                flatshading=False,
                name='Processed',
                showscale=False
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode='data'
            ),
            scene2=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Single mesh view with optional heatmap
        viz_mode = st.radio("Color Map", ["Solid Color", "Displacement Heatmap"], horizontal=True)
        
        if viz_mode == "Displacement Heatmap" and displacement.max() > 0:
            fig = go.Figure(data=[
                go.Mesh3d(
                    x=processed_verts[:, 0], y=processed_verts[:, 1], z=processed_verts[:, 2],
                    i=processed_faces[:, 0], j=processed_faces[:, 1], k=processed_faces[:, 2],
                    intensity=displacement,
                    colorscale='Jet',
                    showscale=True,
                    lighting=dict(ambient=0.5, diffuse=0.6, specular=0.2),
                    flatshading=False,
                    name='Mesh'
                )
            ])
        else:
            fig = go.Figure(data=[
                go.Mesh3d(
                    x=processed_verts[:, 0], y=processed_verts[:, 1], z=processed_verts[:, 2],
                    i=processed_faces[:, 0], j=processed_faces[:, 1], k=processed_faces[:, 2],
                    color='gold',
                    lighting=dict(ambient=0.5, diffuse=0.6, specular=0.2),
                    flatshading=False,
                    name='Mesh'
                )
            ])

        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Download Button
    st.markdown("### Export")
    
    # Save mesh to a temporary buffer
    # PyVista save is file-based, so we use a temp file
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp_mesh:
        mesh.save(tmp_mesh.name)
        with open(tmp_mesh.name, "rb") as f:
            st.download_button(
                label="📥 Download Processed Mesh (.stl)",
                data=f,
                file_name="processed_brain.stl",
                mime="application/octet-stream"
            )
    # Cleanup is handled by OS eventually or we can unlink, but inside streamlit flow it's tricky. 
    # For small temp files it's usually fine.

with col2:
    st.subheader("Quantitative Metrics")
    
    # Comparison metrics
    col_a, col_b = st.columns(2)
    with col_a:
        st.caption("Original")
        st.metric("Triangles", f"{original_triangles:,}")
        st.metric("Volume", f"{original_volume:,.0f} mm³")
    with col_b:
        st.caption("Processed")
        st.metric("Triangles", f"{processed_triangles:,}", delta=f"{processed_triangles - original_triangles:,}")
        st.metric("Volume", f"{volume:,.0f} mm³", delta=f"{volume_change_pct:.2f}%")
    
    if hausdorff_dist is not None:
        st.metric("Hausdorff Distance", f"{hausdorff_dist:.3f} mm", help="Max geometric deviation from original")
    
    st.metric("Surface Area", f"{surface_area:,.2f} mm²")
    
    st.markdown("---")
    st.subheader("Triangle Quality")
    if aspect_ratios.size == 0:
        st.write("No triangle aspect ratios computed.")
    else:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(aspect_ratios, bins=60, color='#00A0B0', edgecolor='black')
        ax.set_xlabel('Aspect Ratio (max_edge/min_edge)')
        ax.set_ylabel('Triangle Count')
        ax.grid(alpha=0.3)
        mean_ar = np.mean(aspect_ratios)
        ax.axvline(mean_ar, color='red', linestyle='--', label=f'Mean: {mean_ar:.2f}')
        ax.legend()
        st.pyplot(fig)

# Volume tracking chart
if track_volume and len(volume_history) > 0:
    st.markdown("---")
    st.subheader("📊 Volume Preservation Analysis")
    
    df = pd.DataFrame(volume_history, columns=['Iteration', 'Volume (mm³)'])
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df['Iteration'], df['Volume (mm³)'], marker='o', linewidth=2, markersize=4, color='#0ea5e9')
    ax.axhline(original_volume, color='red', linestyle='--', label='Original Volume', linewidth=1.5)
    ax.set_xlabel('Smoothing Iteration')
    ax.set_ylabel('Volume (mm³)')
    ax.set_title(f'Volume Change Over {processing_algo} Smoothing Iterations')
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    
    # Show statistics
    final_volume_change = ((df['Volume (mm³)'].iloc[-1] - original_volume) / original_volume) * 100
    st.info(f"**Volume Change:** {final_volume_change:+.2f}% after {iterations} iterations")

st.markdown("---")
st.caption("Complete pipeline: Marching Cubes → Smoothing (Laplacian/Taubin) → QEM Simplification → Quality Metrics")