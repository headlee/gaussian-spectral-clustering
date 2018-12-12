# gaussian-spectral-clustering
Implementation of Automated Gaussian spectral clustering of hyperspectral data (https://www.spiedigitallibrary.org/conference-proceedings-of-spie/4725/1/Automated-Gaussian-spectral-clustering-of-hyperspectral-data/10.1117/12.478758.short)

## Implementation files
- data_handling.py: Functions to load in and pre-process hyperspectral data from AVIRIS and ARCHER sources
- utils.py: Various machine learning and helper functions used that aren't exactly part of Gaussian spectral clustering
- gaussian_spectral_clustering.py: Contains the code that implements the functional blocks from the paper

## Jupyter notebooks - "main" runscripts
- gaussian-spectral-clustering-[ARCHER/AVIRIS].ipynb: Python notebooks implementing the "main" and demonstrating how to use the codebase/generate results for ARCHER and AVIRIS data

## Dependencies
The main dependencies are NumPy, SciPy, and SpectralPython. An environment.yml file is included in this repo that shows my Conda development environment packages (use at your own risk - I've had issues with creating Conda environments from YAML before).
