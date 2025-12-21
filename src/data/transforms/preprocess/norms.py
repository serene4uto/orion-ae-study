from src.data.transforms.base import BaseTransform
import numpy as np


class MinMaxNorm(BaseTransform):
    """Min-Max normalization transform (scales to [0, 1] by default)."""
    
    def __init__(self, feature_range: tuple[float, float] = (0, 1)):
        """
        Args:
            feature_range: Desired range of transformed data (min, max)
        """
        self.feature_range = feature_range
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Apply min-max normalization per channel.
        For data shape (channels, time_steps), normalizes each channel independently
        by computing min/max along the time dimension (axis=1).
        """
        # Compute min/max along time axis (axis=1) for each channel
        data_min = np.min(data, axis=1, keepdims=True)  # Shape: (channels, 1)
        data_max = np.max(data, axis=1, keepdims=True)  # Shape: (channels, 1)
        
        # Avoid division by zero
        data_range = data_max - data_min
        data_range[data_range == 0] = 1
        
        # Normalize each channel independently to [0, 1]
        normalized = (data - data_min) / data_range
        
        # Scale to feature_range if needed
        if self.feature_range != (0, 1):
            min_val, max_val = self.feature_range
            normalized = normalized * (max_val - min_val) + min_val
        
        return normalized

class ZScoreNorm(BaseTransform):
    """Z-score normalization (standardization) using mean and std from reference data."""
    
    def __init__(self, mean: [float], std: [float]):
        """
        Args:
            mean: List of means for each channel
            std: List of standard deviations for each channel
        """
        self.mean = np.asarray(np.array(mean))
        self.std = np.asarray(np.array(std))
        
        # Ensure mean and std are 1D or 2D with compatible shapes
        if self.mean.ndim > 2 or self.std.ndim > 2:
            raise ValueError(f"mean and std must be 1D or 2D arrays, got shapes {self.mean.shape} and {self.std.shape}")
        
        # Ensure they have the same shape
        if self.mean.shape != self.std.shape:
            raise ValueError(f"mean and std must have the same shape, got {self.mean.shape} and {self.std.shape}")
        
        # Handle zero std: set to 1 (consistent with MinMaxNorm behavior)
        self.std = np.where(self.std == 0, 1.0, self.std)
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data using reference mean and std per channel.
        
        Args:
            data: Array of shape (channels, time_steps) to normalize.
                  Each channel is normalized using its corresponding mean/std value.
        
        Returns:
            Normalized data with same shape as input.
        """
        data = np.asarray(data)
        
        # Validate data shape matches expected number of channels
        if data.ndim != 2:
            raise ValueError(
                f"data must be 2D array (channels, time_steps), got shape {data.shape}"
            )
        
        if data.shape[0] != len(self.mean):
            raise ValueError(
                f"Number of channels in data ({data.shape[0]}) must match "
                f"number of channels in mean/std ({len(self.mean)})"
            )
        
        # Broadcasting: (channels, time_steps) - (channels,) → (channels, time_steps)
        # NumPy automatically broadcasts (channels,) to match (channels, time_steps)
        # Each channel row is normalized independently using its mean/std
        return (data - self.mean[:, np.newaxis]) / self.std[:, np.newaxis]


class SeriesZScoreNorm(BaseTransform):
    """Z-score normalization with series-specific mean and std parameters."""
    
    def __init__(self, series_params: dict[str, dict[str, list[float]]]):
        """
        Args:
            series_params: Dictionary mapping series names to normalization parameters.
                          Format: {
                              'series_name': {
                                  'mean': [mean_ch1, mean_ch2, ...],
                                  'std': [std_ch1, std_ch2, ...]
                              },
                              ...
                          }
        """
        self.series_params = {}
        for series_name, params in series_params.items():
            mean = np.asarray(params['mean'])
            std = np.asarray(params['std'])
            
            # Validate shapes
            if mean.ndim > 2 or std.ndim > 2:
                raise ValueError(f"mean and std must be 1D or 2D arrays for series {series_name}")
            if mean.shape != std.shape:
                raise ValueError(f"mean and std must have same shape for series {series_name}")
            
            # Handle zero std
            std = np.where(std == 0, 1.0, std)
            
            self.series_params[series_name] = {
                'mean': mean,
                'std': std
            }
    
    def __call__(self, data: np.ndarray, series: str) -> np.ndarray:
        """
        Normalize data using series-specific mean and std.
        
        Args:
            data: Array of shape (channels, time_steps) to normalize.
            series: Series name to use for normalization.
        
        Returns:
            Normalized data with same shape as input.
        """
        if series not in self.series_params:
            raise ValueError(
                f"Series '{series}' not found in normalization parameters. "
                f"Available: {list(self.series_params.keys())}"
            )
        
        data = np.asarray(data)
        
        # Validate data shape
        if data.ndim != 2:
            raise ValueError(
                f"data must be 2D array (channels, time_steps), got shape {data.shape}"
            )
        
        params = self.series_params[series]
        mean = params['mean']
        std = params['std']
        
        if data.shape[0] != len(mean):
            raise ValueError(
                f"Number of channels in data ({data.shape[0]}) must match "
                f"number of channels in mean/std ({len(mean)}) for series {series}"
            )
        
        return (data - mean[:, np.newaxis]) / std[:, np.newaxis]