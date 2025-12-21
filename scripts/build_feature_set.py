
from pathlib import Path
import sys

# Get the project root from the script's location
# Script is in scripts/, so go up one level to get project root
ROOT_PATH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_PATH))

import argparse
import yaml
import numpy as np
import json
import shutil
from datetime import datetime
from tqdm import tqdm

from src.data.dataset import OrionAEFrameDataset
from src.data.transforms.preprocess import (filters, norms, miscs)
from src.data.transforms import features
from src.data.transforms import (
    PreprocessPipeline, 
    NormPipeline, 
    FilterPipeline, 
    MiscPipeline,
    FeaturePipeline,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract features from frame dataset and save to processed directory"
    )

    parser.add_argument(
        '--frame-path',
        type=Path,
        required=True,
        help='Path to directory containing frame dataset (built by build_frame_cycle)'
    )
    parser.add_argument(
        '--dataset-config-path',
        type=Path,
        required=True,
        help='Path to .yaml dataset config file (e.g., configs/dataset/example_1.yaml)'
    )
    parser.add_argument(
        '--preprocess-config-path',
        type=Path,
        default=None,
        help='Path to .yaml preprocess config file (optional, can be empty)'
    )
    parser.add_argument(
        '--feature-config-path',
        type=Path,
        required=True,
        help='Path to .yaml feature config file'
    )
    parser.add_argument(
        '--save-path',
        type=Path,
        default=None,
        help='Path to save processed dataset (default: data/processed)'
    )

    return parser


def process_preprocess_cfg(preprocess_cfg: dict):

    filter_list = []
    norm_list = []
    misc_list = []

    for f in preprocess_cfg.get('filters', []):
        filter_name = f.get('name')
        if filter_name is None:
            raise ValueError("Filter configuration missing 'name' field")
        filter_params = f.get('params', {})
        filter_list.append(getattr(filters, filter_name)(**filter_params))

    for n in preprocess_cfg.get('norms', []):   
        norm_name = n.get('name')
        if norm_name is None:
            raise ValueError("Norm configuration missing 'name' field")
        norm_params = n.get('params', {})
        norm_list.append(getattr(norms, norm_name)(**norm_params))

    for m in preprocess_cfg.get('miscs', []):
        misc_name = m.get('name')
        if misc_name is None:
            raise ValueError("Misc configuration missing 'name' field")
        misc_params = m.get('params', {})
        misc_list.append(getattr(miscs, misc_name)(**misc_params))

    return PreprocessPipeline(
        filters=filter_list,
        norms=norm_list,
        miscs=misc_list
    )

def process_feature_cfg(feature_cfg: dict):
    """
    Process feature configuration and create FeaturePipeline.
    
    Args:
        feature_cfg: Dictionary with 'features' key containing list of feature configs
                    Each feature config has 'name' and 'params'
    
    Returns:
        FeaturePipeline instance
    """
    feature_list = []
    
    # Extract features list from config
    features_config = feature_cfg.get('features', [])
    
    for feat in features_config:
        feature_name = feat.get('name')
        if feature_name is None:
            raise ValueError("Feature configuration missing 'name' field")
        feature_params = feat.get('params', {})
        
        # Get feature transform class from features module
        feature_class = getattr(features, feature_name, None)
        if feature_class is None:
            available = [x for x in dir(features) if not x.startswith('_') and x != 'FeatureExtractionPipeline']
            raise ValueError(
                f"Unknown feature type: {feature_name}. "
                f"Available: {available}"
            )
        
        feature_list.append(feature_class(**feature_params))
    
    return FeaturePipeline(feature_list)


def save_features_dataset(
    dataset: OrionAEFrameDataset,
    save_path: Path,
):
    """
    Extract features from dataset and save to processed directory.
    
    Structure mirrors input:
        data/processed/{dataset_name}_features_{timestamp}/
            data/
                B_05_001_features.npy  # Features for all frames in file
                B_05_002_features.npy
                ...
            metadata.csv  # Copy from raw dataset
            dataset_info.json  # Copy from raw dataset
            feature_info.json  # Info about extracted features
    """
    # Create save directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = dataset.config.get('name', 'dataset')
    output_dir = save_path / f"{dataset_name}_features_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    print(f"Extracting features for {len(dataset)} frames across {len(dataset.file_paths)} files...")
    
    # Process each file
    for file_idx, file_path in enumerate(tqdm(dataset.file_paths, desc="Processing files")):
        # Load all frames from this file
        raw_data = dataset._load_data(file_path)  # Shape: (num_frames, time_steps, channels)
        
        # Process each frame
        file_features = []
        for frame_idx in range(len(raw_data)):
            # Get frame data
            frame_data = raw_data[frame_idx]  # (time_steps, channels)
            
            # Select channels
            selected_data = dataset._select_channels(frame_data)  # (time_steps, selected_channels)
            
            # Preprocess
            preprocessed = dataset._preprocess_data(
                selected_data.T,  # (channels, time_steps)
                series=dataset.file_series[file_idx]
            )
            
            # Reshape to (channels, 1, time_steps) for consistency with dataset.__getitem__
            if preprocessed.ndim == 2:
                preprocessed = preprocessed[:, np.newaxis, :]
            
            # Extract features
            features = dataset._extract_features(preprocessed)  # dict of features
            
            # Store features for this frame
            file_features.append(features)
        
        # Save all frames' features for this file
        # Use original filename with _features suffix
        original_filename = Path(file_path).stem
        save_file_path = data_dir / f"{original_filename}_features.npy"
        np.save(save_file_path, file_features, allow_pickle=True)
    
    # Copy metadata and dataset_info
    shutil.copy(dataset.data_path / "metadata.csv", output_dir / "metadata.csv")
    if (dataset.data_path / "dataset_info.json").exists():
        shutil.copy(dataset.data_path / "dataset_info.json", output_dir / "dataset_info.json")
    
    # Save feature info
    feature_info = {
        "feature_types": [feat.__class__.__name__ for feat in dataset.feature_pipeline.features],
        "num_files": len(dataset.file_paths),
        "total_frames": len(dataset),
        "extraction_timestamp": timestamp,
    }
    with open(output_dir / "feature_info.json", 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print(f"\n✓ Features saved to: {output_dir}")
    return output_dir


def main():
    parser = parse_args()
    args = parser.parse_args()
    
    # Validate inputs
    if not args.frame_path.exists():
        raise ValueError(f"Frame path does not exist: {args.frame_path}")
    if not args.dataset_config_path.exists():
        raise ValueError(f"Dataset config not found: {args.dataset_config_path}")
    if not args.feature_config_path.exists():
        raise ValueError(f"Feature config not found: {args.feature_config_path}")
    
    # Set default save path
    if args.save_path is None:
        args.save_path = ROOT_PATH / "data" / "processed"
    args.save_path.mkdir(parents=True, exist_ok=True)
    
    # Load configs
    # Preprocess config is optional - use empty dict if not provided
    if args.preprocess_config_path is None or not args.preprocess_config_path.exists():
        preprocess_cfg = {}
        print("No preprocess config provided, using empty preprocessing pipeline")
    else:
        with open(args.preprocess_config_path, 'r') as f:
            preprocess_cfg = yaml.safe_load(f) or {}
    
    with open(args.feature_config_path, 'r') as f:
        feature_cfg = yaml.safe_load(f)
    
    # Load and validate dataset config
    with open(args.dataset_config_path, 'r') as f:
        dataset_config_raw = yaml.safe_load(f)
    
    dataset_config = dataset_config_raw.get('dataset', dataset_config_raw)
    
    # Validate dataset type
    dataset_type = dataset_config.get('type')
    if dataset_type != 'OrionAEFrameDataset':
        raise ValueError(
            f"Dataset type must be 'OrionAEFrameDataset', got '{dataset_type}'. "
            f"Please add 'type: \"OrionAEFrameDataset\"' to your dataset config."
        )
    
    # Build pipelines
    # Handle empty preprocess config
    if not preprocess_cfg:
        preprocess_pipeline = PreprocessPipeline()
    else:
        preprocess_pipeline = process_preprocess_cfg(preprocess_cfg.get('preprocess', preprocess_cfg))
    
    feature_pipeline = process_feature_cfg(feature_cfg)
    
    # Create dataset with type='all' to process all data
    dataset = OrionAEFrameDataset(
        data_path=str(args.frame_path),
        config_path=str(args.dataset_config_path),
        type='all',  # Process all data regardless of splits
        preprocess_pipeline=preprocess_pipeline,
        feature_pipeline=feature_pipeline,
    )
    
    # Extract and save features
    output_dir = save_features_dataset(
        dataset=dataset,
        save_path=args.save_path,
    )
    
    print(f"\n✓ Feature extraction complete!")
    print(f"  Output directory: {output_dir}")
    return output_dir
if __name__ == "__main__":
    main()


