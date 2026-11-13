"""CCD star-field preprocessing, star extraction, moving-target detection
and astrometry pipeline for the 2.5 m telescope data-flow demonstration.

Modules
-------
config      typed configuration
simulate    synthetic CCD sequence + calibration frames
preprocess  bias/dark/flat, background, denoising
detect      star extraction/centroid, frame differencing, morphology, Hough
astrometry  WCS, distortion, catalogue matching
evaluate    detection rate / false alarm / astrometric accuracy
visualize   diagnostic figures
run         command-line entry point
"""

__version__ = "1.0.0"
