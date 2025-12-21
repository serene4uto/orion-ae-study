"""Data transforms for preprocessing and feature extraction."""

from src.data.transforms.base import BaseTransform

# Import pipelines from base module
from src.data.transforms.base import (
    FilterPipeline,
    NormPipeline,
    MiscPipeline,
    PreprocessPipeline,
    FeaturePipeline,
)

# Module aliases for backward compatibility (used in notebooks, scripts)
from src.data.transforms import preprocess as preprocessing
from src.data.transforms import features

__all__ = [
    "BaseTransform",
    # Pipeline Transforms
    "FilterPipeline",
    "NormPipeline",
    "MiscPipeline",
    "PreprocessPipeline",
    "FeaturePipeline",
    # Module aliases
    "preprocessing",
    "features",
]