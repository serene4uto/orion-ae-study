# we will have preprocessing pipeline with fixed order that go from filters group then norms group
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
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Apply all filters serially."""
        result = data.copy()
        for filter_fn in self.filters:
            result = filter_fn(result)
        return result


class NormPipeline(BaseTransform):
    """
    Pipeline that applies multiple normalization transforms serially.
    """
    
    def __init__(self, norms: list[Callable]):
        self.norms = norms if norms else []
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Apply all norms serially."""
        result = data.copy()
        for norm_fn in self.norms:
            result = norm_fn(result)
        return result


class PreprocessingPipeline(BaseTransform):
    """
    Top-level preprocessing pipeline with fixed order:
    1. FilterPipeline (all filters)
    2. NormPipeline (all normalizations)
    """
    
    def __init__(
        self, 
        filters: Optional[list[Callable]] = None,
        norms: Optional[list[Callable]] = None
    ):
        self.filter_pipeline = FilterPipeline(filters or [])
        self.norm_pipeline = NormPipeline(norms or [])
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Apply transforms serially: filters first, then norms.
        """
        result = self.filter_pipeline(data)
        result = self.norm_pipeline(result)
        return result


# # Example: Individual filter transform
# class HighPassFilter(BaseTransform):
#     def __init__(self, cutoff: float):
#         self.cutoff = cutoff
    
#     def __call__(self, data: np.ndarray) -> np.ndarray:
#         # Your filter implementation
#         return filtered_data

# # Example: Individual norm transform
# class MinMaxNorm(BaseTransform):
#     def __call__(self, data: np.ndarray) -> np.ndarray:
#         # Your normalization implementation
#         return normalized_data
