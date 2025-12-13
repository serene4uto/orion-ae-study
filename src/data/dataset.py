from pathlib import Path
from torch.utils.data import Dataset

import pandas as pd
import numpy as np
import yaml


class OrionAEFrameDataset(Dataset):

    def __init__(
        self,
        data_path: str,
        config_path: str,
        type: str = 'train', # 'train', 'val', 'test'

    ):
        # Convert to Path objects if strings are passed
        self.type = type
        self.num_frames = 0

        # check if metadata.csv exists
        if not (Path(data_path) / 'metadata.csv').exists():
            raise FileNotFoundError(f"Metadata file not found at {data_path / 'metadata.csv'}")

        # check if config file exists
        if not Path(config_path).exists():
            raise FileNotFoundError(f'Config file not found at {config_path}')

        # load config
        with open(Path(config_path), 'r') as f:
            self.config = yaml.safe_load(f)

        self.load_val_label_map = self.config['labels']

        # load metadata
        metadata = self._filter_metadata(
            pd.read_csv(Path(data_path) / 'metadata.csv'), 
            self.type, 
            self.config
        )

        self.num_frames = sum(metadata['num_frames'])
        self.file_paths = metadata['file_path'].tolist()
        self.file_frame_offsets = []

        for file_path, file_path_idx in zip(self.file_paths, range(len(self.file_paths))):
            if file_path_idx == 0:
                self.file_frame_offsets.append(0)
            else:
                self.file_frame_offsets.append(
                    self.file_frame_offsets[-1] + metadata.loc[metadata['file_path'] == file_path, 'num_frames'].values[0]
                )
        
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


    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        sample_item = {}

        # locate file index where the frame is located 

        return sample_item
        
    def _locate_file_index(self, index: int) -> int:
        """
        Locates the file index where the frame is located.
        """
        for file_index, file_frame_offset in enumerate(self.file_frame_offsets):
            if index < file_frame_offset:
                return file_index
        return len(self.file_frame_offsets) - 1
