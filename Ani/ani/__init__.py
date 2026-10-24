"""Ani – a free, CPU-friendly annotator for correcting cell segmentations and building training data."""
from .safety import prepare_cpu

prepare_cpu()

__version__ = "0.4.2"
