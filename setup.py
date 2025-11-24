"""
Setup script for High-Fidelity Mesh Improvement Pipeline
"""

from setuptools import setup, find_packages

with open("docs/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mesh-improvement-pipeline",
    version="1.0.0",
    author="Shubham Vikas Mhaske",
    author_email="shubham.mhaske@tamu.edu",
    description="High-fidelity mesh improvement for MRI-derived anatomical models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shubham-mhaske/geometric_modelling_project",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Visualization",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "streamlit>=1.28.0",
        "nibabel>=5.1.0",
        "numpy>=1.24.0",
        "pyvista>=0.46.4",
        "plotly>=5.19.0",
        "scipy>=1.11.0",
        "scikit-image>=0.21.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "torch>=2.0.0",
        "synapseclient>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mesh-train=scripts.train_ml_model:main",
            "mesh-download=scripts.download_data:main",
            "mesh-test=tests.test_pipeline:main",
        ],
    },
)
