from pathlib import Path
from typing import Optional

import numpy as np

from src.data.dataset import BaseDataset


class CWTScalogramDataset(BaseDataset):
    """
    Dataset for loading pre-computed CWT scalogram features.
    
    Handles data from: data/processed/example_1_features_20251221_182611
    The data files already contain extracted CWT scalogram features (RGB images).
    
    Expected feature format:
    - Each file contains features for multiple frames
    - Shape: (num_frames, channels, height, width, 3) or (num_frames, channels, height, width)
    - Features are pre-computed CWT scalograms (RGB images typically 224x224x3)
    """

    def __init__(
        self,
        data_path: str,
        config_path: str,
        type: str = 'train',  # 'train', 'val', 'test', 'all'
    ):
        """
        Initialize CWT Scalogram dataset.
        
        Args:
            data_path: Path to processed feature dataset directory containing metadata.csv
            config_path: Path to dataset configuration YAML file
            type: Dataset split type ('train', 'val', 'test', 'all')
        """
        # Initialize base class (handles config, metadata filtering, label building)
        super().__init__(data_path, config_path, type)
        
        # Calculate frame offsets and counts (frame-based dataset logic, similar to OrionAEFrameDataset)
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

    def _load_data(self, file_path: str) -> np.ndarray:
        """
        Loads pre-computed feature data from a file.
        
        Args:
            file_path: Relative path to the feature file from self.data_path
                      (e.g., "data\\B_05_001.npy" or "data/B_05_001.npy")
            
        Returns:
            Loaded feature array (typically shape: (num_frames, channels, height, width, 3))
        """
        # Normalize path separators (handle both \ and /)
        normalized_path = file_path.replace('\\', '/')
        
        # Convert to Path object and resolve
        file_path_obj = Path(normalized_path)
        
        # Extract filename and check if it needs _features suffix
        # Metadata might have "B_05_001.npy" but actual file is "B_05_001_features.npy"
        filename = file_path_obj.name
        parent_dir = file_path_obj.parent
        
        # Try with original filename first
        full_path = self.data_path / parent_dir / filename
        
        # If file doesn't exist, try adding _features suffix before extension
        if not full_path.exists():
            # Insert _features before the extension
            stem = file_path_obj.stem
            suffix = file_path_obj.suffix
            features_filename = f"{stem}_features{suffix}"
            full_path = self.data_path / parent_dir / features_filename
        
        # Final check
        if not full_path.exists():
            raise FileNotFoundError(
                f"Feature file not found at {full_path}. "
                f"Tried: {self.data_path / parent_dir / filename} and {full_path}"
            )
        
        # Load with allow_pickle=True in case features are stored as dictionaries or objects
        return np.load(full_path, allow_pickle=True)

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
        """
        Get a feature sample from the dataset.
        
        Args:
            index: Global frame index in the dataset
            
        Returns:
            Dictionary with 'features', 'label', and 'serie' keys
        """
        # Locate file index where the frame is located
        file_index, local_frame_index = self._locate_file_and_frame_index(index)
        file_path = self.file_paths[file_index]
        file_label = self.file_labels[file_index]
        file_serie = self.file_series[file_index]
        
        # Load pre-computed features for the entire file
        # Features are saved as a list of dictionaries (one dict per frame) via np.save with allow_pickle=True
        # This becomes a numpy array with dtype=object when loaded
        features_data = self._load_data(file_path)
        
        # Extract the dictionary for this specific frame
        # features_data is an array of shape (num_frames,) containing dictionaries
        if isinstance(features_data, np.ndarray) and features_data.dtype == object:
            # Standard case: array of dictionaries
            frame_features = features_data[local_frame_index]
            # frame_features is now a dict, but if numpy wrapped it, unwrap it
            if isinstance(frame_features, np.ndarray) and frame_features.dtype == object:
                frame_features = frame_features.item()
        elif isinstance(features_data, list):
            # Fallback: if it's a list instead of numpy array
            frame_features = features_data[local_frame_index]
        else:
            # Unexpected format - raise informative error
            raise ValueError(
                f"Unexpected feature data format. Expected numpy object array or list of dicts, "
                f"got {type(features_data)}. File: {file_path}"
            )
        
        # Ensure frame_features is a dictionary
        if not isinstance(frame_features, dict):
            raise ValueError(
                f"Expected dictionary for frame features, got {type(frame_features)}. "
                f"File: {file_path}, Frame: {local_frame_index}"
            )
        
        # Build sample item
        # If frame_features is a dict, merge it with other fields
        # Otherwise, use 'features' key
        if isinstance(frame_features, dict):
            sample_item = frame_features.copy()
            sample_item['label'] = file_label
            sample_item['serie'] = file_serie
        else:
            # Single feature array - use 'features' key
            sample_item = {
                'features': frame_features,  # Pre-computed CWT scalogram features
                'label': file_label,
                'serie': file_serie,
            }
        
        return sample_item
