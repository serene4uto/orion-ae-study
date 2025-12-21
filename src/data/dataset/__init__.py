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
        config_path: str,
        type: str = 'train',  # 'train', 'val', 'test', 'all'
    ):
        """
        Initialize base dataset.
        
        Args:
            data_path: Path to dataset directory containing metadata.csv
            config_path: Path to dataset configuration YAML file
            type: Dataset split type ('train', 'val', 'test', 'all')
        """
        # Convert to Path objects if strings are passed
        self.data_path = Path(data_path)
        config_path = Path(config_path)
        self.type = type

        # Check if metadata.csv exists
        if not (self.data_path / 'metadata.csv').exists():
            raise FileNotFoundError(f"Metadata file not found at {self.data_path / 'metadata.csv'}")

        # Check if config file exists
        if not config_path.exists():
            raise FileNotFoundError(f'Config file not found at {config_path}')

        # Load config
        with open(config_path, 'r') as f:
            config_raw = yaml.safe_load(f)
        
        # Extract dataset config if YAML has 'dataset' as root key
        self.config = config_raw.get('dataset', config_raw)

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
        
        This method selects the appropriate subset of the data for 'train', 'val', 'test', or 'all' sets 
        based on the configuration file.
        - For type 'all', it returns all metadata without any filtering.
        - For type 'test', it performs a serie-based split using the series listed in config['splits']['test'].
        - For 'train' or 'val', it excludes test series and splits based on either chunk index or series index, 
          depending on config['splits']['train_val']['type'] ('chunk-based' or 'serie-based').
        
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
