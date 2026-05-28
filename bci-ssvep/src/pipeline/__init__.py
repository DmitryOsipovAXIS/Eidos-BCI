"""Offline SSVEP signal processing and CCA classification pipeline.

Modules
-------
signal_filtering  — notch (50 Hz) + Butterworth bandpass (3-30 Hz)
preprocessing     — common average rereferencing, optional downsampling
cca_classifier    — training-free CCA frequency detector
pipeline          — end-to-end offline pipeline (preprocess + classify + evaluate)
"""

from pipeline.pipeline import SSVEPPipeline
from pipeline.cca_classifier import SSVEPClassifierCCA
from pipeline.signal_filtering import apply_filters, notch_filter, bandpass_filter
from pipeline.preprocessing import common_average_reference, downsample

__all__ = [
    "SSVEPPipeline",
    "SSVEPClassifierCCA",
    "apply_filters",
    "notch_filter",
    "bandpass_filter",
    "common_average_reference",
    "downsample",
]
