# src/data/transforms/features/cwt.py

import numpy as np
from PIL import Image
from typing import Optional, Tuple, Dict, Any, Union
from matplotlib import colormaps

from src.data.transforms.base import BaseTransform

# Try importing both libraries
try:
    from ssqueezepy import cwt
    SSQUEEZEPY_AVAILABLE = True
except ImportError:
    SSQUEEZEPY_AVAILABLE = False
    cwt = None

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    pywt = None


class CWTScalogramTransform(BaseTransform):
    """
    CWT Scalogram feature extraction: CWT → Magnitude → Resize → Colormap → 224×224 RGB
    
    Supports wavelets from two libraries:
    
    1. ssqueezepy wavelets (library='ssqueezepy'):
       - 'gmw' or 'morse': Generalized Morse Wavelet
       - 'morlet': Morlet wavelet
       - 'bump': Bump wavelet
       - 'cmhat': Complex Mexican Hat
    
    2. PyWavelets wavelets (library='pywt'):
       - 'morl': Morlet wavelet
       - 'mexh': Mexican Hat wavelet
       - 'gaus1' to 'gaus8': Gaussian wavelets (1-8)
       - 'cgau1' to 'cgau8': Complex Gaussian wavelets (1-8)
       - 'cmor': Complex Morlet wavelet
       - 'fbsp': Frequency B-Spline wavelet
       - 'shan': Shannon wavelet
    """
    
    # Available PyWavelets wavelets
    PYWT_WAVELETS = [
        'morl', 'mexh',  # Real wavelets
        'gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8',  # Gaussian
        'cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5', 'cgau6', 'cgau7', 'cgau8',  # Complex Gaussian
        'cmor',  # Complex Morlet
        'fbsp',  # Frequency B-Spline
        'shan',  # Shannon
    ]
    
    def __init__(
        self,
        wavelet_type: Optional[str] = None,
        wavelet: Optional[str] = None,  # Alias for wavelet_type
        library: Optional[str] = None,  # 'ssqueezepy' or 'pywt' (auto-detected if None)
        wavelet_params: Optional[Dict[str, Any]] = None,
        fs: Optional[float] = None,
        sampling_rate: Optional[float] = None,  # Alias for fs
        n_scales: int = 12,
        freq_min: Optional[float] = None,
        freq_max: Optional[float] = None,
        min_scale: Optional[float] = None,  # Alternative to freq_min (scale-based)
        max_scale: Optional[float] = None,  # Alternative to freq_max (scale-based)
        target_size: Tuple[int, int] = (224, 224),
        colormap: str = 'viridis',
        l1_norm: bool = True,
    ):
        """
        Parameters:
        -----------
        wavelet_type : str, optional
            Wavelet type name (depends on library). Can also use 'wavelet' parameter.
        wavelet : str, optional
            Alias for wavelet_type
        library : str, optional
            Library to use: 'ssqueezepy' or 'pywt'. If None, auto-detects based on wavelet name.
        wavelet_params : dict, optional
            Wavelet-specific parameters (for ssqueezepy wavelets)
        fs : float, optional
            Sampling frequency. Can also use 'sampling_rate' parameter. (default: 5e6 Hz)
        sampling_rate : float, optional
            Alias for fs
        n_scales : int
            Number of wavelet scales (default: 12)
        freq_min, freq_max : float, optional
            Frequency range to cover in Hz (default: 50 kHz to 2.5 MHz)
        min_scale, max_scale : float, optional
            Scale range (alternative to freq_min/freq_max). If provided, will be converted to frequencies.
        target_size : tuple
            Output image size (default: (224, 224))
        colormap : str
            Matplotlib colormap name (default: 'viridis')
        l1_norm : bool
            Use L1 normalization (only for ssqueezepy)
        """
        # Handle aliases
        wavelet_type_input = (wavelet_type or wavelet or 'gmw').lower()
        self.fs = fs or sampling_rate or 5e6
        
        # Auto-detect library if not specified and wavelet is known
        if library is None:
            # Auto-detect based on wavelet name
            # Check if it's a known PyWavelets wavelet
            if wavelet_type_input in self.PYWT_WAVELETS:
                self.library = 'pywt'
            elif wavelet_type_input in ['gmw', 'morse', 'morlet', 'bump', 'cmhat']:
                self.library = 'ssqueezepy'
            else:
                # Default to ssqueezepy
                self.library = 'ssqueezepy'
        else:
            library_lower = library.lower()
            if library_lower == 'ssqueezepy':
                self.library = 'ssqueezepy'
            elif library_lower == 'pywt':
                self.library = 'pywt'
            else:
                raise ValueError(f"Unknown library: {library}. Must be 'ssqueezepy' or 'pywt'")
        
        # Validate library availability
        if self.library == 'ssqueezepy' and not SSQUEEZEPY_AVAILABLE:
            raise ImportError("ssqueezepy is required. Install with: pip install ssqueezepy")
        if self.library == 'pywt' and not PYWT_AVAILABLE:
            raise ImportError("PyWavelets is required. Install with: pip install PyWavelets")
        
        # Normalize and validate wavelet type
        if self.library == 'ssqueezepy':
            self.wavelet_type = self._normalize_ssqueezepy_wavelet(wavelet_type_input)
            if not wavelet_params:
                self.wavelet_params = self._get_default_ssqueezepy_params(self.wavelet_type)
            else:
                self.wavelet_params = wavelet_params
        elif self.library == 'pywt':
            self.wavelet_type = self._normalize_pywt_wavelet(wavelet_type_input)
            self.wavelet_params = wavelet_params or {}
        else:
            raise ValueError(f"Unknown library: {library}. Must be 'ssqueezepy' or 'pywt'")
        
        # Handle scale-based vs frequency-based parameters
        if min_scale is not None or max_scale is not None:
            # Convert scales to frequencies
            if min_scale is None or max_scale is None:
                raise ValueError("Both min_scale and max_scale must be provided if using scale-based parameters")
            
            # Get wavelet center frequency for conversion
            if self.library == 'pywt':
                center_freq = self._get_pywt_center_frequency()
                # frequency = fs / (scale * center_freq) => scale = fs / (frequency * center_freq)
                # So: frequency = fs / (scale * center_freq)
                self.freq_min = self.fs / (max_scale * center_freq)  # Higher scale = lower frequency
                self.freq_max = self.fs / (min_scale * center_freq)  # Lower scale = higher frequency
            else:
                # For ssqueezepy, use similar conversion
                center_freq = 1.0  # Approximate
                self.freq_min = self.fs / (max_scale * center_freq)
                self.freq_max = self.fs / (min_scale * center_freq)
        else:
            # Use frequency-based parameters
            self.freq_min = freq_min or 50e3
            self.freq_max = freq_max or 2.5e6
        
        self.n_scales = n_scales
        self.target_size = target_size
        self.colormap = colormap
        self.l1_norm = l1_norm
    
    def _normalize_ssqueezepy_wavelet(self, wavelet_type: str) -> str:
        """Normalize ssqueezepy wavelet type name"""
        if wavelet_type in ['morse', 'gmw']:
            return 'gmw'
        return wavelet_type
    
    def _normalize_pywt_wavelet(self, wavelet_type: str) -> str:
        """Normalize and validate PyWavelets wavelet type name"""
        # Check if it's a valid PyWavelets wavelet
        if wavelet_type in self.PYWT_WAVELETS:
            return wavelet_type
        
        # Try to find similar names
        if wavelet_type == 'morlet' or wavelet_type == 'morl':
            return 'morl'
        if wavelet_type == 'mexican' or wavelet_type == 'mexh':
            return 'mexh'
        if wavelet_type.startswith('gaus'):
            return wavelet_type  # Already correct format
        if wavelet_type.startswith('cgau'):
            return wavelet_type  # Already correct format
        
        raise ValueError(
            f"Unknown PyWavelets wavelet: {wavelet_type}. "
            f"Available: {', '.join(self.PYWT_WAVELETS)}"
        )
    
    def _get_default_ssqueezepy_params(self, wavelet_type: str) -> Dict[str, Any]:
        """Get default parameters for ssqueezepy wavelets"""
        defaults = {
            'gmw': {'gamma': 3, 'beta': 60},
            'morlet': {'mu': 5.0},
            'bump': {'mu': 5.0, 'sigma': 1.0},
            'cmhat': {},
        }
        return defaults.get(wavelet_type, {})
    
    def _compute_scales_ssqueezepy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute scales for ssqueezepy wavelets"""
        if self.wavelet_type == 'gmw':
            gamma = self.wavelet_params.get('gamma', 3)
            beta = self.wavelet_params.get('beta', 60)
            peak_freq_factor = (beta / (gamma + 1)) ** (1.0 / gamma)
            
            scale_max = peak_freq_factor * self.fs / (2 * np.pi * self.freq_min)
            scale_min = peak_freq_factor * self.fs / (2 * np.pi * self.freq_max)
        else:
            scale_max = self.fs / (2 * np.pi * self.freq_min)
            scale_min = self.fs / (2 * np.pi * self.freq_max)
        
        scales = np.logspace(np.log10(scale_min), np.log10(scale_max), self.n_scales)
        
        # Compute frequencies
        if self.wavelet_type == 'gmw':
            gamma = self.wavelet_params.get('gamma', 3)
            beta = self.wavelet_params.get('beta', 60)
            peak_freq_factor = (beta / (gamma + 1)) ** (1.0 / gamma)
            frequencies = peak_freq_factor * self.fs / (2 * np.pi * scales)
        else:
            frequencies = self.fs / (2 * np.pi * scales)
        
        return scales, frequencies
    
    def _compute_scales_pywt(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute scales for PyWavelets CWT.
        PyWavelets uses scales directly (not frequencies), so we compute scales
        that correspond to the desired frequency range.
        """
        # For PyWavelets, scales are typically 1, 2, 3, ... or log-spaced
        # The relationship between scale and frequency depends on the wavelet
        # For most wavelets: frequency ≈ fs / (scale * wavelet_center_frequency)
        
        # Get wavelet center frequency (approximate)
        # For Morlet: ~1.0, for Mexican Hat: ~0.25, etc.
        center_freq = self._get_pywt_center_frequency()
        
        # Convert frequency range to scale range
        # frequency = fs / (scale * center_freq) => scale = fs / (frequency * center_freq)
        scale_min = self.fs / (self.freq_max * center_freq)
        scale_max = self.fs / (self.freq_min * center_freq)
        
        # Log-spaced scales
        scales = np.logspace(np.log10(scale_min), np.log10(scale_max), self.n_scales)
        
        # Compute corresponding frequencies
        frequencies = self.fs / (scales * center_freq)
        
        return scales, frequencies
    
    def _get_pywt_center_frequency(self) -> float:
        """Get approximate center frequency for PyWavelets wavelets"""
        # These are approximate center frequencies
        center_freqs = {
            'morl': 1.0,      # Morlet
            'mexh': 0.25,     # Mexican Hat
            'cmor': 1.0,      # Complex Morlet
            'shan': 1.0,      # Shannon
            'fbsp': 1.0,      # Frequency B-Spline
        }
        
        # For Gaussian wavelets, use order-dependent frequency
        if self.wavelet_type.startswith('gaus') or self.wavelet_type.startswith('cgau'):
            order = int(self.wavelet_type[-1]) if len(self.wavelet_type) > 4 else 1
            return 1.0 / (2 * np.pi * order)  # Approximate
        
        return center_freqs.get(self.wavelet_type, 1.0)  # Default to 1.0
    
    def _compute_cwt_ssqueezepy(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute CWT using ssqueezepy"""
        scales, frequencies = self._compute_scales_ssqueezepy()
        
        # Build wavelet specification
        if self.wavelet_type == 'gmw':
            wavelet = ('gmw', self.wavelet_params)
        else:
            wavelet = (self.wavelet_type, self.wavelet_params)
        
        # Compute CWT
        Wx, scales_computed, *_ = cwt(
            signal,
            wavelet=wavelet,
            scales=scales,
            fs=self.fs,
            l1_norm=self.l1_norm,
            vectorized=True
        )
        
        # Recompute frequencies from actual scales
        if self.wavelet_type == 'gmw':
            gamma = self.wavelet_params.get('gamma', 3)
            beta = self.wavelet_params.get('beta', 60)
            peak_freq_factor = (beta / (gamma + 1)) ** (1.0 / gamma)
            frequencies = peak_freq_factor * self.fs / (2 * np.pi * scales_computed)
        else:
            frequencies = self.fs / (2 * np.pi * scales_computed)
        
        return Wx, frequencies
    
    def _compute_cwt_pywt(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute CWT using PyWavelets"""
        scales, frequencies = self._compute_scales_pywt()
        
        # PyWavelets CWT: pywt.cwt(signal, scales, wavelet_name)
        # Returns: (coefficients, frequencies)
        # Note: PyWavelets returns frequencies, but we'll use our computed ones
        coefficients, _ = pywt.cwt(signal, scales, self.wavelet_type)
        
        return coefficients, frequencies
    
    def _compute_cwt(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute CWT using the selected library"""
        if self.library == 'ssqueezepy':
            return self._compute_cwt_ssqueezepy(signal)
        elif self.library == 'pywt':
            return self._compute_cwt_pywt(signal)
        else:
            raise ValueError(f"Unknown library: {self.library}")
    
    def _magnitude_to_rgb(self, magnitude: np.ndarray) -> np.ndarray:
        """
        Convert magnitude array to RGB image:
        Magnitude → Normalize → Resize → Colormap → RGB
        """
        # Normalize to [0, 255]
        magnitude_norm = magnitude / (np.max(magnitude) + 1e-10)
        magnitude_8bit = (magnitude_norm * 255).astype(np.uint8)
        
        # Resize to target size using PIL
        img = Image.fromarray(magnitude_8bit, mode='L')
        try:
            img_resized = img.resize(
                (self.target_size[1], self.target_size[0]), 
                Image.Resampling.BILINEAR
            )
        except AttributeError:
            img_resized = img.resize(
                (self.target_size[1], self.target_size[0]), 
                Image.BILINEAR
            )
        
        magnitude_resized = np.array(img_resized)
        
        # Apply colormap
        norm = magnitude_resized / 255.0
        colormap = colormaps[self.colormap]
        rgb_image = (colormap(norm)[:, :, :3] * 255).astype(np.uint8)
        
        return rgb_image
    
    def _process_single_channel(self, signal: np.ndarray) -> np.ndarray:
        """
        Process a single channel signal to scalogram.
        
        Parameters:
        -----------
        signal : array (time_steps,)
            Input signal (1D array)
            
        Returns:
        --------
        scalogram_rgb : array (height × width × 3)
            RGB scalogram image (uint8)
        """
        # Step 1: Compute CWT
        cwt_coeffs, frequencies = self._compute_cwt(signal)
        
        # Step 2: Take magnitude
        magnitude = np.abs(cwt_coeffs)
        
        # Step 3: Convert to RGB scalogram
        scalogram_rgb = self._magnitude_to_rgb(magnitude)
        
        return scalogram_rgb
    
    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply CWT scalogram transform.
        
        Parameters:
        -----------
        signal : array (time_steps,) or (channels, time_steps)
            Input signal (1D or 2D array)
            
        Returns:
        --------
        scalogram_rgb : array (height × width × 3) or (channels, height × width × 3)
            RGB scalogram image(s) (uint8)
        """
        if signal.ndim == 1:
            # Single channel
            return self._process_single_channel(signal)
        elif signal.ndim == 2:
            # Multi-channel: (channels, time_steps)
            n_channels = signal.shape[0]
            scalograms = []
            for i in range(n_channels):
                scalogram = self._process_single_channel(signal[i])
                scalograms.append(scalogram)
            return np.stack(scalograms, axis=0)  # (channels, height, width, 3)
        else:
            raise ValueError(f"Expected 1D or 2D signal, got shape {signal.shape}")