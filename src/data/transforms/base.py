from typing import Callable, Optional
import numpy as np


class BaseTransform:
    """Base class for all transforms."""
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class FilterPipeline(BaseTransform):
    """
    Pipeline that applies multiple filter transforms serially.
    """
    
    def __init__(self, filters: list[Callable]):
        self.filters = filters if filters else []
    
    def __call__(self, data: np.ndarray, series: Optional[str] = None) -> np.ndarray:
        """Apply all filters serially."""
        result = data.copy()
        for filter_fn in self.filters:
            # Pass series if transform accepts it, otherwise ignore
            try:
                result = filter_fn(result, series=series)
            except TypeError:
                result = filter_fn(result)
        return result

class NormPipeline(BaseTransform):
    """
    Pipeline that applies multiple normalization transforms serially.
    """
    
    def __init__(self, norms: list[Callable]):
        self.norms = norms if norms else []
    
    def __call__(self, data: np.ndarray, series: Optional[str] = None) -> np.ndarray:
        """Apply all norms serially."""
        result = data.copy()
        for norm_fn in self.norms:
            # Pass series if transform accepts it, otherwise ignore
            try:
                result = norm_fn(result, series=series)
            except TypeError:
                result = norm_fn(result)
        return result

class MiscPipeline(BaseTransform):
    """
    Pipeline that applies multiple misc transforms serially.
    """
    
    def __init__(self, miscs: list[Callable]):
        self.miscs = miscs if miscs else []
    
    def __call__(self, data: np.ndarray, series: Optional[str] = None) -> np.ndarray:
        """Apply all miscs serially."""
        result = data.copy()
        for misc_fn in self.miscs:
            # Pass series if transform accepts it, otherwise ignore
            try:
                result = misc_fn(result, series=series)
            except TypeError:
                result = misc_fn(result)
        return result

class PreprocessPipeline(BaseTransform):
    """
    Top-level preprocessing pipeline with fixed order:
    1. FilterPipeline (all filters)
    2. NormPipeline (all normalizations)
    3. MiscPipeline (all miscs)
    """
    
    def __init__(
        self, 
        filters: Optional[list[Callable]] = None,
        norms: Optional[list[Callable]] = None,
        miscs: Optional[list[Callable]] = None,
    ):
        self.filter_pipeline = FilterPipeline(filters or [])
        self.norm_pipeline = NormPipeline(norms or [])
        self.misc_pipeline = MiscPipeline(miscs or [])
    
    def __call__(self, data: np.ndarray, series: Optional[str] = None) -> np.ndarray:
        """
        Apply transforms serially: filters first, then norms, then miscs.
        
        Args:
            data: Data to preprocess
            series: Optional series name for series-aware transforms
        """
        result = self.filter_pipeline(data, series=series)
        result = self.norm_pipeline(result, series=series)
        result = self.misc_pipeline(result, series=series)
        return result

class FeaturePipeline(BaseTransform):
    """
    Pipeline that applies multiple feature extraction transforms.
    Each transform extracts features from the input signal.
    """
    
    def __init__(self, features: list[Callable]):
        """
        Args:
            features: List of feature extraction transforms
        """
        self.features = features if features else []
    
    def __call__(self, data: np.ndarray) -> dict:
        """
        Apply all feature extraction transforms.
        
        Args:
            data: Input data (can be 1D or 2D array)
            
        Returns:
            Dictionary with feature names as keys and feature arrays as values
        """
        result = {}
        for feature_fn in self.features:
            # Apply feature transform
            feature_result = feature_fn(data)
            
            if isinstance(feature_result, dict):
                # Prefix keys with transform name to avoid collisions
                transform_name = feature_fn.__class__.__name__
                prefixed_dict = {f"{transform_name}_{k}": v for k, v in feature_result.items()}
                result.update(prefixed_dict)
            else:
                # If transform doesn't return dict, use transform class name as key
                feature_name = feature_fn.__class__.__name__
                result[feature_name] = feature_result
        return result