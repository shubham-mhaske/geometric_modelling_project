
import streamlit as st
import nibabel as nib
import numpy as np
from skimage import measure
import pyvista as pv
from stpyvista import st_pyvista
import matplotlib.pyplot as plt
import tempfile
import os
import json
import synapseclient
import synapseutils

# --- Synapse Integration ---
def synapse_login_and_download():
    """
    Logs into Synapse using token from config.json and downloads the dataset.
    """
    try:
        with open('config.json') as f:
            config = json.load(f)
        token = config.get('dataset_synapse_token')
        if not token:
            st.error("Synapse token not found in config.json.")
            return None

        syn = synapseclient.Synapse()
        syn.login(authToken=token)
        
        st.sidebar.info("Downloading dataset from Synapse...")
        files = synapseutils.syncFromSynapse(syn, 'syn64952532')
        st.sidebar.success("Dataset downloaded.")
        
        # Return a list of file paths
        return [file.path for file in files]

    except Exception as e:
        st.sidebar.error(f"Synapse connection failed: {e}")
        return None

# --- Page Configuration ---
st.set_page_config(
    page_title="CSCE 645 Project: Brain Tumor Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Styling ---
# Use a dark, sleek theme for matplotlib plots
plt.style.use('seaborn-v0_8-darkgrid')

# --- Helper Functions (Backend Logic) ---

def process_nifti_file(file_path):
    """
    Loads a .nii.gz file from a path, processes the segmentation mask, and generates a 3D mesh.

    Args:
        file_path: The path to the .nii.gz file.

    Returns:
        pv.PolyData: A PyVista mesh object, or None if processing fails.
    """
    try:
        # 1. Load the NIfTI file
        nii_img = nib.load(file_path)

        # 2. Get the 3D data array
        data = nii_img.get_fdata()

        # 3. Combine tumor labels (1, 2, 4) into a single binary mask
        # Label 1: Necrotic and Non-Enhancing Tumor Core (NCR/NET)
        # Label 2: Peritumoral Edema (ED)
        # Label 4: GD-enhancing Tumor (ET)
        binary_mask = np.isin(data, [1, 2, 4]).astype(np.uint8)

        # 4. Get voxel spacing from the NIfTI header (affine matrix)
        # This is crucial for generating a geometrically accurate mesh.
        voxel_spacing = nii_img.header.get_zooms()[:3]

        # 5. Run the Marching Cubes algorithm
        # The `spacing` parameter ensures the mesh is scaled correctly.
        verts, faces, _, _ = measure.marching_cubes(
            binary_mask,
            level=0.5,  # Isovalue for the binary mask
            spacing=voxel_spacing
        )

        # 6. Create a PyVista mesh
        # The faces array needs to be padded for PyVista's format.
        # Marching cubes returns triangles, so we pad with '3'.
        faces_padded = np.hstack((np.full((faces.shape[0], 1), 3), faces))
        mesh = pv.PolyData(verts, faces_padded)

        return mesh

    except Exception as e:
        st.error(f"An error occurred during NIfTI processing: {e}")
        return None

def calculate_quantitative_metrics(mesh):
    """
    Computes surface area, volume, and triangle aspect ratios for a given mesh.

    Args:
        mesh (pv.PolyData): The input PyVista mesh.

    Returns:
        tuple: A tuple containing:
            - float: Surface area.
            - float: Volume.
            - np.ndarray: Array of aspect ratio values for each triangle.
    """
    surface_area = mesh.area
    volume = mesh.volume
    
    # Calculate triangle quality (aspect ratio)
    # PyVista's aspect ratio is the ratio of the longest edge length to the
    # shortest altitude for each triangle. Perfect equilateral triangles have a low aspect ratio.
    aspect_ratios = mesh.compute_cell_quality('aspect_ratio')

    return surface_area, volume, aspect_ratios

def create_quality_histogram(aspect_ratios):
    """
    Generates a matplotlib histogram of triangle aspect ratios.

    Args:
        aspect_ratios (np.ndarray): Array of aspect ratio values.

    Returns:
        matplotlib.figure.Figure: The generated figure object for plotting.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Use a specific number of bins for better visualization
    ax.hist(aspect_ratios, bins=80, color='#00A0B0', edgecolor='black')
    
    ax.set_title('Triangle Quality Distribution (Aspect Ratio)', fontsize=16)
    ax.set_xlabel('Aspect Ratio', fontsize=12)
    ax.set_ylabel('Number of Triangles', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Add a vertical line for the mean aspect ratio
    mean_ratio = np.mean(aspect_ratios)
    ax.axvline(mean_ratio, color='#CC333F', linestyle='dashed', linewidth=2)
    ax.text(mean_ratio * 1.1, ax.get_ylim()[1] * 0.8, f'Mean: {mean_ratio:.2f}', color='#CC333F')

    fig.tight_layout()
    return fig


# --- Streamlit UI (Frontend) ---

st.title("🧠 Brain Tumor 3D Analysis & Visualization")
st.markdown("""
Welcome to the First Update for the **CSCE 645 Geometric Modeling** project.
This application loads a BraTS 2023 segmentation file, reconstructs the tumor as a 3D mesh,
and provides a suite of quantitative analyses.
""")

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("⚙️ Controls")
    
    # Part 1 Goal: File Uploader
    uploaded_file = st.file_uploader(
        "Upload BraTS .nii.gz File",
        type=['nii', 'gz'],
        help="Upload the segmentation file (e.g., 'sub-0001_seg.nii.gz') from the BraTS 2023 dataset."
    )
    
    st.markdown("---")
    
    # Part 2 Goal: Final Project Wireframe (Placeholders)
    st.subheader("Final Project Wireframe")
    
    processing_algo = st.selectbox(
        "Processing Algorithm",
        ('None (Original Mesh)', 'Laplacian Smoothing', 'Taubin Smoothing'),
        help="Future work: Apply mesh smoothing algorithms."
    )
    
    iterations = st.slider(
        "Number of Iterations", 0, 100, 10,
        help="Future work: Controls the intensity of the smoothing algorithm."
    )
    
    target_reduction = st.slider(
        "Target Reduction (QEM)", 0.0, 1.0, 0.5,
        help="Future work: Set the target percentage for mesh simplification using Quadric Edge Collapse."
    )

# --- Main Content Area ---
if uploaded_file is None:
    st.info("Please upload a `.nii.gz` file using the sidebar to begin analysis.")

else:
    # Check if the selected algorithm is the one implemented for this update
    if processing_algo != 'None (Original Mesh)':
        st.warning(
            f"The '{processing_algo}' algorithm is a placeholder for the final project. "
            "Please select 'None (Original Mesh)' to see the functionality for Update 1.",
            icon="⚠️"
        )
    else:
        # --- Run the full pipeline ---
        with st.spinner('Processing NIfTI file and generating 3D mesh...'):
            mesh = process_nifti_file(uploaded_file)

        if mesh and mesh.n_points > 0:
            st.success("Mesh generated successfully!")

            # --- Quantitative Analysis ---
            surface_area, volume, aspect_ratios = calculate_quantitative_metrics(mesh)

            # --- Display Metrics and Plots ---
            st.header("📊 Quantitative Analysis (Update 1 Deliverable)")
            
            # Layout for metrics and histogram
            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("Mesh Metrics")
                st.metric(label="Surface Area (mm²)", value=f"{surface_area:.2f}")
                st.metric(label="Volume (mm³)", value=f"{volume:.2f}")
                st.metric(label="Total Triangles", value=f"{mesh.n_faces}")

            with col2:
                st.subheader("Triangle Quality Histogram")
                quality_hist_fig = create_quality_histogram(aspect_ratios)
                st.pyplot(quality_hist_fig)

            st.markdown("---")

            # --- 3D Visualization ---
            st.header("🖥️ 3D Mesh Visualization")
            
            # Initialize the PyVista plotter
            plotter = pv.Plotter(window_size=[800, 600])
            plotter.background_color = 'black' # Sleek dark background
            
            # Add the mesh with a visually appealing color and style
            plotter.add_mesh(
                mesh,
                color='#FFD700',  # Gold color for the tumor
                style='surface',
                smooth_shading=True,
                specular=0.5,      # Add some shininess
                specular_power=10
            )
            
            # Set up camera for a good initial view
            plotter.view_isometric()
            plotter.enable_zoom_scaling()

            # Pass the plotter to the streamlit component
            st_pyvista(plotter, key="pv_viewer")

        elif mesh and mesh.n_points == 0:
            st.warning(
                "The Marching Cubes algorithm did not produce any vertices. "
                "This can happen if the selected labels (1, 2, 4) are not present in the uploaded file, "
                "or if the segmentation is empty."
            )
        else:
            st.error("Failed to generate or display the mesh.")
