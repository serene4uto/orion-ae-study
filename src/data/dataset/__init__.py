from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional
from abc import ABC, abstractmethod

import pandas as pd
import yaml

__all__ = [
    "BaseDataset",
    "OrionAEFrameDataset",
    "CWTScalogramDataset",
    "register_dataset",
    "get_dataset",
    "list_datasets",
]

# Dataset registry
DATASET_REGISTRY = {}


class BaseDataset(Dataset, ABC):
    """
    Base class for all Orion AE datasets.
    
    This class provides common functionality for:
    - Config loading and validation
    - Metadata filtering by split type (train/val/test/all)
    - Label building from load values
    - Common initialization structure
    
    Subclasses should implement:
    - _load_data(): Load data from file
    - __getitem__(): Return sample item dictionary
    - __len__(): Return dataset length
    """
    
    def __init__(
        self,
        data_path: str,
        config: dict,
        type: str = 'train',  # 'train', 'val', 'test', 'all'
    ):
        """
        Initialize base dataset.
        
        Args:
            data_path: Path to dataset directory containing metadata.csv
            config: Dataset configuration dictionary
            type: Dataset split type ('train', 'val', 'test', 'all')
        """
        # Convert to Path objects if strings are passed
        self.data_path = Path(data_path)
        self.config = config
        self.type = type

        # Check if metadata.csv exists
        if not (self.data_path / 'metadata.csv').exists():
            raise FileNotFoundError(f"Metadata file not found at {self.data_path / 'metadata.csv'}")
        
        # Get label mapping from config
        self.load_val_label_map = self.config['labels']
        
        # Load metadata
        metadata = self._filter_metadata(
            pd.read_csv(self.data_path / 'metadata.csv'), 
            self.type, 
            self.config
        )

        # Build labels
        self.label_names, self.file_labels = self._build_file_labels(
            metadata, 
            self.load_val_label_map
        )
        
        # Store metadata-derived attributes (subclasses may use these)
        self.metadata = metadata
        self.file_series = metadata['series'].tolist()
        self.file_paths = metadata['file_path'].tolist()

    def _filter_metadata(self, metadata: pd.DataFrame, type: str, config: dict) -> pd.DataFrame:
        """
        Filters the metadata DataFrame according to the dataset split configuration.
        
        Supports two formats:
        
        1. New format (per-series, per-split chunks):
           splits:
             train:
               'B': [0,1,2,3,4,5,6,7,8,9]
               'C': [0,1,2,3,4,5,6,7,8,9]
             val:
               'B': [8,9]
             test:
               'F': [0,1,2,3,4,5,6,7,8,9]
        
        2. Old format (backward compatibility):
           splits:
             train_val:
               type: chunk-based  # or serie-based
               train: [0,1,2,3,4,5,6,7]
               val: [8,9]
             test: ['F']
        
        Args:
            metadata: Full metadata DataFrame
            type: Dataset type ('train', 'val', 'test', 'all')
            config: Dataset configuration dictionary
            
        Returns:
            Filtered metadata DataFrame
        """
        # Handle 'all' type - return all metadata without filtering
        if type == 'all':
            return metadata
        
        splits_config = config.get('splits', {})
        
        # Check if using new format (has 'train', 'val', 'test' as top-level keys with series dicts)
        # New format: splits['train'] is a dict with series as keys
        # Old format: splits['train_val'] exists or splits['test'] is a list
        is_new_format = (
            type in splits_config and 
            isinstance(splits_config.get(type), dict)
        )
        
        if is_new_format:
            # New format: per-series, per-split chunk selection
            split_config = splits_config[type]
            
            if not isinstance(split_config, dict):
                raise ValueError(
                    f"Invalid split config for type '{type}'. "
                    f"Expected dict with series as keys, got {type(split_config)}"
                )
            
            # Build filter conditions: (series='B' AND chunk in [0,1,2,...]) OR (series='C' AND chunk in [0,1,2,...]) ...
            conditions = []
            for series, chunks in split_config.items():
                if not isinstance(chunks, list):
                    raise ValueError(
                        f"Invalid chunks for series '{series}' in '{type}' split. "
                        f"Expected list, got {type(chunks)}"
                    )
                # Filter: series matches AND chunk is in the list
                series_condition = (metadata['series'] == series) & (metadata['chunk'].isin(chunks))
                conditions.append(series_condition)
            
            if not conditions:
                # No conditions means no data for this split
                return metadata.iloc[0:0]  # Return empty DataFrame with same structure
            
            # Combine all conditions with OR
            combined_condition = conditions[0]
            for condition in conditions[1:]:
                combined_condition = combined_condition | condition
            
            return metadata[combined_condition]
        
        else:
            # Old format: backward compatibility
            # Handle test split (always serie-based)
            if type == 'test':
                test_config = splits_config.get('test', [])
                if isinstance(test_config, list):
                    return metadata[metadata['series'].isin(test_config)]
                else:
                    raise ValueError(
                        f"Invalid test split config. Expected list of series, got {type(test_config)}"
                    )
            
            # Exclude test series for train/val splits
            test_config = splits_config.get('test', [])
            if isinstance(test_config, list):
                metadata = metadata[~metadata['series'].isin(test_config)]
            
            # Get train_val config
            train_val_config = splits_config.get('train_val', {})
            if not train_val_config:
                raise ValueError(
                    "Old format requires 'train_val' config in splits. "
                    "If using new format, ensure splits have 'train', 'val', 'test' as dict keys."
                )
            
            split_type = train_val_config.get('type')
            if not split_type:
                raise ValueError("train_val config must have 'type' field ('chunk-based' or 'serie-based')")
            
            # Determine column and values based on split type and dataset type
            if split_type == 'chunk-based':
                column = 'chunk'
            elif split_type == 'serie-based':
                column = 'series'
            else:
                raise ValueError(f"Invalid split type: {split_type}. Must be 'chunk-based' or 'serie-based'")
            
            # Get the values to filter by
            if type == 'train':
                values = train_val_config.get('train')
            elif type == 'val':
                values = train_val_config.get('val')
            else:
                raise ValueError(f"Invalid dataset type: {type}")
            
            if values is None:
                raise ValueError(f"train_val config must have '{type}' field")
            
            return metadata[metadata[column].isin(values)]

    def _build_file_labels(self, metadata: pd.DataFrame, load_val_label_map: dict) -> tuple[list, list]:
        """
        Constructs label indices for each file in the dataset based on 'load_val' and the provided label mapping.
        
        Since every frame in a file corresponds to the same load, all frames in a file will share the same label.
        
        Args:
            metadata: Filtered metadata DataFrame
            load_val_label_map: Dictionary mapping label names to load value lists
            
        Returns:
            Tuple of (label_names, file_labels) where:
                - label_names: List of label names in order
                - file_labels: List of label indices, one per file
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
            
            file_labels.append(label_names.index(label_key))  # Convert label name to index

        return label_names, file_labels

    @abstractmethod
    def _load_data(self, file_path: str):
        """
        Load data from a file. Must be implemented by subclasses.
        
        Args:
            file_path: Relative path to the data file from self.data_path
            
        Returns:
            Loaded data (format depends on subclass)
        """
        pass

    @abstractmethod
    def __getitem__(self, index):
        """
        Get a sample from the dataset. Must be implemented by subclasses.
        
        Args:
            index: Sample index
            
        Returns:
            Dictionary with at least 'label' and 'serie' keys
        """
        pass

    @abstractmethod
    def __len__(self):
        """
        Return the number of samples in the dataset. Must be implemented by subclasses.
        """
        pass


def register_dataset(name: str):
    """
    Decorator to automatically register a dataset class.
    
    Usage:
        @register_dataset("OrionAEFrameDataset")
        class OrionAEFrameDataset(BaseDataset):
            ...
    """
    def decorator(dataset_class):
        if name in DATASET_REGISTRY:
            raise ValueError(f"Dataset name '{name}' is already registered")
        if not issubclass(dataset_class, BaseDataset):
            raise ValueError(f"Dataset {dataset_class.__name__} must inherit from BaseDataset")
        
        DATASET_REGISTRY[name] = dataset_class
        return dataset_class
    
    return decorator


def get_dataset(dataset_name: str, **params):
    """
    Get a dataset by name and return an instance of the dataset.
    
    Args:
        dataset_name: Name of the dataset type (e.g., "OrionAEFrameDataset")
        **params: Parameters to pass to dataset constructor
    
    Returns:
        Dataset instance
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Dataset '{dataset_name}' not found in registry. "
            f"Available datasets: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[dataset_name](**params)


def list_datasets():
    """List all available dataset types."""
    return list(DATASET_REGISTRY.keys())


# Import subclasses here to avoid circular imports
# They will be registered via decorators
from src.data.dataset.orion_ae_frame_dataset import OrionAEFrameDataset
from src.data.dataset.cwt_scalogram_dataset import CWTScalogramDataset
