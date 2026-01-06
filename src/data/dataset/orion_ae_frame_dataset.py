from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.transforms import PreprocessPipeline, FeaturePipeline
from src.data.dataset import BaseDataset, register_dataset


@register_dataset("OrionAEFrameDataset")
class OrionAEFrameDataset(BaseDataset):
    """
    Dataset for loading raw frame data from segmented cycles.
    
    Handles data from: data/raw/segmented_cycles_positive_c1_l42373_c_A_B_C_D_20251220_154951
    """

    CHANNELS = ['A', 'B', 'C', 'D']

    def __init__(
        self,
        data_path: str,
        config: dict,
        type: str = 'train',  # 'train', 'val', 'test', 'all'
        preprocess_pipeline: Optional[PreprocessPipeline] = None,
        feature_pipeline: Optional[FeaturePipeline] = None,
    ):
        # Initialize base class (handles config, metadata filtering, label building)
        super().__init__(data_path, config, type)
        
        self.num_frames = 0

        # Get selected channels from config and validate
        selected_channels = self.config.get('channels', self.CHANNELS)
        if not isinstance(selected_channels, list):
            raise ValueError(f"channels in config must be a list, got {type(selected_channels)}")
        
        # Validate that all selected channels exist
        invalid_channels = [ch for ch in selected_channels if ch not in self.CHANNELS]
        if invalid_channels:
            raise ValueError(f"Invalid channels in config: {invalid_channels}. Valid channels are: {self.CHANNELS}")
        
        self.selected_channels = selected_channels
        
        # Create channel indices for fast indexing
        # Maps selected channel names to their indices in the full CHANNELS list
        self.channel_indices = [self.CHANNELS.index(ch) for ch in self.selected_channels]

        # Initialize preprocessing pipeline
        # If not provided, create an empty pipeline (no-op)
        self.preprocess_pipeline = preprocess_pipeline or PreprocessPipeline()
        
        # Initialize feature extraction pipeline
        # If not provided, create an empty pipeline (returns empty dict)
        self.feature_pipeline = feature_pipeline or FeaturePipeline([])

        # Calculate frame offsets and counts (frame-based dataset logic)
        self.num_frames = sum(self.metadata['num_frames'])
        self.file_frame_offsets = []
        self.file_num_frames = []  # Store num_frames for each file for bounds checking
        
        # Calculate cumulative offsets: offset[i] = sum of frames in files 0 to i-1
        # This means file i contains frames from offset[i] to offset[i] + num_frames[i] - 1
        cumulative_offset = 0
        for file_path in self.file_paths:
            self.file_frame_offsets.append(cumulative_offset)
            # Get num_frames for this file from metadata
            file_num_frames = self.metadata.loc[self.metadata['file_path'] == file_path, 'num_frames'].values[0]
            self.file_num_frames.append(file_num_frames)
            cumulative_offset += file_num_frames

    def _select_channels(self, data: np.ndarray) -> np.ndarray:
        """
        Selects only the channels specified in config.
        
        Args:
            data: Raw data array with shape (time_steps, num_channels)
                  where num_channels matches len(CHANNELS)
        
        Returns:
            Data array with only selected channels, shape (time_steps, len(selected_channels))
            Note: This format is maintained throughout - (time_steps, channels)
        """
        # Data shape: (time_steps, num_channels)
        # Select channels using indices
        return data[:, self.channel_indices]

    def _preprocess_data(self, data: np.ndarray, series: Optional[str] = None) -> np.ndarray:
        """
        Preprocesses the data using the configured preprocessing pipeline.
        The pipeline applies filters first, then normalizations, serially.

        Args:
            data: Data to preprocess
            series: Optional series name for series-aware transforms
        """
        return self.preprocess_pipeline(data, series=series)
    
    def _extract_features(self, data: np.ndarray) -> dict:
        """
        Extracts the features from the data using the feature pipeline.
        
        Args:
            data: Preprocessed data with shape (channels, 1, time_steps)
            
        Returns:
            Dictionary with feature names as keys and feature arrays as values
        """
        # Handle shape: (channels, 1, time_steps) -> (channels, time_steps)
        # Feature transforms expect (channels, time_steps) or (time_steps,)
        if data.ndim == 3:
            # Remove the middle dimension: (channels, 1, time_steps) -> (channels, time_steps)
            data = data.squeeze(axis=1)
        elif data.ndim == 2 and data.shape[0] == 1:
            # Single channel: (1, time_steps) -> (time_steps,)
            data = data[0]
        
        # Apply feature pipeline
        return self.feature_pipeline(data)

    def _load_data(self, file_path: str) -> np.ndarray:
        """
        Loads the data from a file.
        
        Args:
            file_path: Relative path to the data file from self.data_path
            
        Returns:
            Loaded numpy array
        """
        # Resolve relative file_path to absolute path relative to data_path
        full_path = self.data_path / file_path
        return np.load(full_path)
        
    def _locate_file_and_frame_index(self, index: int) -> tuple[int, int]:
        """
        Locates the file index and local frame index for a given global frame index.

        Args:
            index: Global frame index in the dataset

        Returns:
            tuple: (file_index, local_frame_index) where:
                - file_index: Index of the file containing this frame
                - local_frame_index: Index of the frame within that file (0-based)
        """
        # Validate index
        if index < 0 or index >= self.num_frames:
            raise IndexError(f"Index {index} is out of range [0, {self.num_frames})")
        
        # Iterate backwards to find the largest offset <= index
        # This finds the file that contains this global index
        for file_index in range(len(self.file_frame_offsets) - 1, -1, -1):
            if index >= self.file_frame_offsets[file_index]:
                local_frame_index = index - self.file_frame_offsets[file_index]
                
                # Verify bounds (safety check using stored num_frames)
                if local_frame_index >= self.file_num_frames[file_index]:
                    raise IndexError(
                        f"Local frame index {local_frame_index} is out of bounds for file {self.file_paths[file_index]} "
                        f"(file has {self.file_num_frames[file_index]} frames)"
                    )
                
                return file_index, local_frame_index

        # This should never happen if index is valid, but return 0 as fallback
        return 0, index

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        sample_item = {}

        # Locate file index where the frame is located 
        file_index, local_frame_index = self._locate_file_and_frame_index(index)
        file_path = self.file_paths[file_index]
        file_label = self.file_labels[file_index]
        file_serie = self.file_series[file_index]

        # Load raw data for the frame
        raw_data = self._load_data(file_path)[local_frame_index]

        # Select only configured channels
        selected_data = self._select_channels(raw_data)

        # Store raw data
        sample_item['raw'] = selected_data.T  # (channels, time_steps) from _select_channels

        # Preprocess the selected channels (input is (channels, time_steps))
        preprocessed = self._preprocess_data(sample_item['raw'], series=file_serie)
        
        # Reshape to (channels, 1, time_steps) for Conv2d compatibility
        # This will become (batch, channels, 1, time_steps) when batched
        # Models using Conv1d can reshape/transpose as needed
        if preprocessed.ndim == 2:
            # (channels, time_steps) -> (channels, 1, time_steps)
            preprocessed = preprocessed[:, np.newaxis, :]
        
        sample_item['final'] = preprocessed  # (channels, 1, time_steps) - final processed data ready for model
        sample_item['label'] = file_label
        sample_item['serie'] = file_serie

        return sample_item