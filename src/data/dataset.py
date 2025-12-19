from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional

import pandas as pd
import numpy as np
import yaml

from src.data.transforms.preprocessing import PreprocessingPipeline


class OrionAEFrameDataset(Dataset):

    CHANNELS = ['A', 'B', 'C', 'D']

    def __init__(
        self,
        data_path: str,
        config_path: str,
        type: str = 'train', # 'train', 'val', 'test'
        preprocessing_pipeline: Optional[PreprocessingPipeline] = None,
    ):
        # Convert to Path objects if strings are passed
        self.data_path = Path(data_path)
        config_path = Path(config_path)
        self.type = type
        self.num_frames = 0

        # check if metadata.csv exists
        if not (self.data_path / 'metadata.csv').exists():
            raise FileNotFoundError(f"Metadata file not found at {self.data_path / 'metadata.csv'}")

        # check if config file exists
        if not config_path.exists():
            raise FileNotFoundError(f'Config file not found at {config_path}')

        # load config
        with open(config_path, 'r') as f:
            config_raw = yaml.safe_load(f)
        
        # Extract dataset config if YAML has 'dataset' as root key
        self.config = config_raw.get('dataset', config_raw)

        self.load_val_label_map = self.config['labels']
        
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
        self.preprocessing_pipeline = preprocessing_pipeline or PreprocessingPipeline()

        # load metadata
        metadata = self._filter_metadata(
            pd.read_csv(self.data_path / 'metadata.csv'), 
            self.type, 
            self.config
        )

        self.num_frames = sum(metadata['num_frames'])
        self.file_series = metadata['series'].tolist()
        self.file_paths = metadata['file_path'].tolist()
        self.file_frame_offsets = []
        self.file_num_frames = []  # Store num_frames for each file for bounds checking
        
        # Calculate cumulative offsets: offset[i] = sum of frames in files 0 to i-1
        # This means file i contains frames from offset[i] to offset[i] + num_frames[i] - 1
        cumulative_offset = 0
        for file_path in self.file_paths:
            self.file_frame_offsets.append(cumulative_offset)
            # Get num_frames for this file from metadata
            file_num_frames = metadata.loc[metadata['file_path'] == file_path, 'num_frames'].values[0]
            self.file_num_frames.append(file_num_frames)
            cumulative_offset += file_num_frames
        
        self.label_names, self.file_labels = self._build_file_labels(metadata, self.load_val_label_map)
        

    def _filter_metadata(self, metadata: pd.DataFrame, type: str, config: dict):
        """
        Filters the metadata DataFrame according to the dataset split configuration.
        This method selects the appropriate subset of the data for 'train', 'val', or 'test' sets based on the configuration file.
        - For type 'test', it performs a serie-based split using the series listed in config['splits']['test'].
        - For 'train' or 'val', it excludes test series and splits based on either chunk index or series index, 
          depending on config['splits']['train_val']['type'] ('chunk-based' or 'serie-based').
        Returns the filtered metadata DataFrame.
        """
        # Handle test split (always serie-based)
        if type == 'test':
            return metadata[metadata['series'].isin(config['splits']['test'])]
        
        # Exclude test series for train/val splits
        metadata = metadata[~metadata['series'].isin(config['splits']['test'])]
        
        # Get train_val config
        train_val_config = config['splits']['train_val']
        split_type = train_val_config['type']
        
        # Determine column and values based on split type and dataset type
        if split_type == 'chunk-based':
            column = 'chunk'
        elif split_type == 'serie-based':
            column = 'series'
        else:
            raise ValueError(f"Invalid split type: {split_type}")
        
        # Get the values to filter by
        if type == 'train':
            values = train_val_config['train']
        elif type == 'val':
            values = train_val_config['val']
        else:
            raise ValueError(f"Invalid dataset type: {type}")
        
        return metadata[metadata[column].isin(values)]

    def _build_file_labels(self, metadata: pd.DataFrame, load_val_label_map: dict) -> [list, list]:
        """
        Constructs label indices for each file in the dataset based on 'load_val' and the provided label mapping.
        Since every frame in a file corresponds to the same load, all frames in a file will share the same label.
        Returns both the list of label names and a list mapping each file to its label index.
        """
        label_names = list(load_val_label_map.keys())
        file_labels = []

        for row in metadata.itertuples():
            label_key = None
            load_val = float(row.load_val)
            
            for key, values in load_val_label_map.items():
                # Ensure values is a list and convert to floats for comparison
                if not isinstance(values, list):
                    values = [values]  # Convert single value to list
                
                # Convert all values in the list to floats for comparison
                values_float = [float(v) for v in values]
                
                if load_val in values_float:
                    label_key = key
                    break
            
            if label_key is None:
                raise ValueError(f"load_val {row.load_val} not found in label map")
            
            file_labels.append(label_names.index(label_key)) # Convert label name to index

        return label_names, file_labels

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
        return self.preprocessing_pipeline(data, series=series)
    
    def _extract_features(self, data: np.ndarray) -> dict:
        """
        Extracts the features from the data.
        """
        return {}

    def _load_data(self, file_path: str) -> np.ndarray:
        """
        Loads the data from a file.
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

        # locate file index where the frame is located 
        file_index, local_frame_index = self._locate_file_and_frame_index(index)
        file_path = self.file_paths[file_index]
        file_label = self.file_labels[file_index]
        file_serie = self.file_series[file_index]  # Add this line

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
        
        sample_item['preprocessed'] = preprocessed  # (channels, 1, time_steps)
        sample_item['features'] = self._extract_features(sample_item['preprocessed'])
        sample_item['label'] = file_label
        sample_item['serie'] = file_serie

        return sample_item