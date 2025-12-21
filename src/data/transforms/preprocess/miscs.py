from src.data.transforms.base import BaseTransform
import numpy as np


class HanningWindow(BaseTransform):
    """Hanning window transform to reduce edge effects."""
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Apply Hanning window to reduce edge effects."""
        if data.ndim == 1:
            window = np.hanning(len(data))
            return data * window
        else:
            # Apply along time dimension (last axis)
            window = np.hanning(data.shape[-1])
            return data * window