from src.data.transforms.base import BaseTransform
import numpy as np


class HighPassFilter(BaseTransform):
    """High-pass filter transform."""
    
    def __init__(self, cutoff: float, fs: float, order: int = 4):
        """
        Args:
            cutoff: Cutoff frequency in Hz
            fs: Sampling frequency in Hz
            order: Filter order
        """
        try:
            from scipy import signal
        except ImportError:
            raise ImportError("scipy is required for HighPassFilter. Install with: pip install scipy")
        
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        self.sos = signal.butter(order, cutoff, btype='high', fs=fs, output='sos')
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Apply high-pass filter to data."""
        from scipy import signal
        
        if data.ndim == 1:
            return signal.sosfilt(self.sos, data)
        else:
            # Apply filter along the last axis (time dimension)
            # For data shape (channels, time_steps), filters along axis=-1 (time_steps)
            return signal.sosfilt(self.sos, data, axis=-1)