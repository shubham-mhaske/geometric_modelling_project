import numpy as np
import nibabel as nib
import os
from skimage import measure

def create_noisy_sphere(shape=(100, 100, 100), radius=30, noise_level=2.0):
    """Create a binary mask of a sphere with added boundary noise."""
    center = np.array(shape) // 2
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    
    # Add noise to the distance field
    noise = np.random.normal(0, noise_level, shape)
    dist_noisy = dist + noise
    
    # Create binary mask
    mask = dist_noisy <= radius
    return mask.astype(np.uint8)

def save_synthetic_data(output_dir):
    """Generate and save synthetic datasets."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Noisy Sphere
    print("Generating Noisy Sphere...")
    sphere_mask = create_noisy_sphere(noise_level=1.5)
    
    # Create a dummy T1 image (just the sphere with some intensity)
    t1_data = sphere_mask * 100.0 + np.random.normal(0, 10, sphere_mask.shape)
    t1_data[t1_data < 0] = 0
    
    # Save
    sphere_dir = os.path.join(output_dir, "Synthetic-Sphere")
    os.makedirs(sphere_dir, exist_ok=True)
    
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(sphere_mask, affine), os.path.join(sphere_dir, "mask.nii.gz"))
    nib.save(nib.Nifti1Image(t1_data, affine), os.path.join(sphere_dir, "t1n.nii.gz"))
    
    # 2. Noisy Cube
    print("Generating Noisy Cube...")
    cube_mask = np.zeros((100, 100, 100), dtype=np.uint8)
    cube_mask[30:70, 30:70, 30:70] = 1
    # Add noise by flipping random pixels near boundary
    boundary = measure.find_contours(cube_mask[50], 0.5) # Just to check
    # Simple noise: add random blocks
    noise = np.random.rand(100, 100, 100)
    cube_mask[(noise > 0.95) & (cube_mask == 0) & (np.abs(50-np.indices((100,100,100))[0]) < 25)] = 1
    
    cube_dir = os.path.join(output_dir, "Synthetic-Cube")
    os.makedirs(cube_dir, exist_ok=True)
    nib.save(nib.Nifti1Image(cube_mask, affine), os.path.join(cube_dir, "mask.nii.gz"))
    nib.save(nib.Nifti1Image(t1_data, affine), os.path.join(cube_dir, "t1n.nii.gz")) # Reuse t1 for simplicity

    print(f"Synthetic data saved to {output_dir}")

if __name__ == "__main__":
    save_synthetic_data("data/data")
