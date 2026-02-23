# Tutorials

Interactive Jupyter notebooks demonstrating tetra3rs workflows.

!!! note
    These notebooks use pre-saved outputs and do not require running. The data files (TESS FITS images, Hipparcos catalog) are not included in the repository — see [Installation](../getting-started/installation.md) for how to obtain them if you want to run the notebooks yourself.

## Available Tutorials

### [Basic Plate Solving](basic-solve.ipynb)

Demonstrates the full plate-solving pipeline on a single TESS Full Frame Image:

- Loading a FITS image and extracting centroids
- Solving with iterative distortion calibration (multiple passes)
- Comparing solved coordinates against FITS WCS headers
- Visualizing matched stars overlaid on the image

### [Multi-Image Camera Calibration](multi-image-calibration.ipynb)

Demonstrates multi-image camera calibration using 10 TESS images from the same CCD:

- Solving multiple images with shared camera parameters
- Iterative calibration passes with progressively tighter match radius and higher polynomial order
- Achieving sub-10″ RMSE and sub-3″ agreement with FITS WCS
- Visualization of matched centroids across all images
