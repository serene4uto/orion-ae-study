#!/usr/bin/env python3
"""
build_feature_set_optimized.py - Extract features from frame dataset (Optimized)

This is an optimized version of build_feature_set.py with:
- Multiprocessing for parallel file processing
- Batch frame processing where possible
- Reduced memory overhead
- Better progress tracking

Usage:
------
    python scripts/build_feature_set_optimized.py \\
        --frame-path <path_to_frame_dataset> \\
        --dataset-config-path <path_to_dataset_config.yaml> \\
        --feature-config-path <path_to_feature_config.yaml> \\
        [--preprocess-config-path <path_to_preprocess_config.yaml>] \\
        [--save-path <output_directory>] \\
        [--workers <num_workers>]

Arguments:
----------
    --frame-path            Path to segmented frame dataset directory
                            (created by build_frame_cycle.py)
    --dataset-config-path   Path to dataset config YAML (defines channels, labels, splits)
    --feature-config-path   Path to feature extraction config YAML (defines feature transforms)
    --preprocess-config-path (Optional) Path to preprocessing config YAML (filters, norms)
    --save-path             (Optional) Output directory (default: data/processed)
    --workers               (Optional) Number of parallel workers (default: CPU count)

Example:
--------
    # Basic usage with CWT scalogram features (uses all CPU cores)
    python scripts/build_feature_set_optimized.py \\
        --frame-path data/raw/segmented_cycles_positive_c1_l42373_c_A_B_C_D_20260107_205752 \\
        --dataset-config-path configs/dataset/example_1.yaml \\
        --feature-config-path configs/feature/feature_cwt_scalogram_gmw.yaml

    # With 4 workers and preprocessing
    python scripts/build_feature_set_optimized.py \\
        --frame-path data/raw/segmented_cycles_positive_c1_l42373_c_A_B_C_D_20260107_205752 \\
        --dataset-config-path configs/dataset/example_1.yaml \\
        --feature-config-path configs/feature/feature_cwt_scalogram_gmw.yaml \\
        --preprocess-config-path configs/preprocess/preprocess_hanning_only.yaml \\
        --workers 4

Output:
-------
    Creates a directory structure:
        data/processed/{dataset_name}_features_{timestamp}/
            data/
                {filename}_features.npy   # Feature dict for each file
            metadata.csv                  # Copy from source dataset
            dataset_info.json             # Updated with selected channels
            feature_info.json             # Feature extraction metadata
"""

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
from multiprocessing import Pool, cpu_count
from functools import partial
import traceback

from src.data.transforms.preprocess import (filters, norms, miscs)
from src.data.transforms import features
from src.data.transforms import (
    PreprocessPipeline, 
    FeaturePipeline,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract features from frame dataset (Optimized with multiprocessing)"
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
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: CPU count)'
    )

    return parser


def process_preprocess_cfg(preprocess_cfg: dict):
    """Build preprocessing pipeline from config."""
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
    """Process feature configuration and create FeaturePipeline."""
    feature_list = []
    features_config = feature_cfg.get('features', [])
    
    for feat in features_config:
        feature_name = feat.get('name')
        if feature_name is None:
            raise ValueError("Feature configuration missing 'name' field")
        feature_params = feat.get('params', {})
        
        feature_class = getattr(features, feature_name, None)
        if feature_class is None:
            available = [x for x in dir(features) if not x.startswith('_') and x != 'FeatureExtractionPipeline']
            raise ValueError(
                f"Unknown feature type: {feature_name}. "
                f"Available: {available}"
            )
        
        feature_list.append(feature_class(**feature_params))
    
    return FeaturePipeline(feature_list)


def process_single_file(args):
    """
    Process a single file - designed to be called in parallel.
    
    Args:
        args: Tuple of (file_idx, file_path, config_dict)
        
    Returns:
        Tuple of (file_path, num_frames, success, error_msg)
    """
    file_idx, file_path, config = args
    
    try:
        # Reconstruct pipelines in worker process
        # (Can't pickle complex objects, so we rebuild from config)
        preprocess_pipeline = process_preprocess_cfg(config['preprocess_cfg'])
        feature_pipeline = process_feature_cfg(config['feature_cfg'])
        
        data_path = Path(config['data_path'])
        data_dir = Path(config['data_dir'])
        channel_indices = config['channel_indices']
        file_series = config['file_series']
        
        # Load raw data
        full_path = data_path / file_path
        raw_data = np.load(full_path)
        
        # Process all frames in this file
        file_features = []
        num_frames = len(raw_data)
        
        for frame_idx in range(num_frames):
            # Get frame data: (time_steps, channels)
            frame_data = raw_data[frame_idx]
            
            # Select channels: (time_steps, selected_channels)
            selected_data = frame_data[:, channel_indices]
            
            # Transpose to (channels, time_steps) for preprocessing
            data_ct = selected_data.T
            
            # Preprocess
            preprocessed = preprocess_pipeline(data_ct, series=file_series)
            
            # Extract features (pipeline handles shape internally)
            # No need for intermediate reshape - feature pipeline handles (C, T)
            extracted_features = feature_pipeline(preprocessed)
            
            file_features.append(extracted_features)
        
        # Save features for this file
        original_filename = Path(file_path).stem
        save_file_path = data_dir / f"{original_filename}_features.npy"
        np.save(save_file_path, file_features, allow_pickle=True)
        
        return (file_path, num_frames, True, None)
        
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        return (file_path, 0, False, error_msg)


def save_features_dataset_parallel(
    data_path: Path,
    file_paths: list,
    file_series: list,
    channel_indices: list,
    selected_channels: list,
    dataset_config: dict,
    preprocess_cfg: dict,
    feature_cfg: dict,
    save_path: Path,
    num_workers: int,
):
    """
    Extract features from dataset using parallel processing.
    """
    # Create save directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = dataset_config.get('name', 'dataset')
    output_dir = save_path / f"{dataset_name}_features_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    print(f"Extracting features from {len(file_paths)} files using {num_workers} workers...")
    
    # Prepare config dict for workers (serializable)
    config = {
        'data_path': str(data_path),
        'data_dir': str(data_dir),
        'channel_indices': channel_indices,
        'preprocess_cfg': preprocess_cfg,
        'feature_cfg': feature_cfg,
    }
    
    # Prepare arguments for each file
    args_list = [
        (idx, fp, {**config, 'file_series': file_series[idx]})
        for idx, fp in enumerate(file_paths)
    ]
    
    # Process files in parallel
    total_frames = 0
    failed_files = []
    
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_file, args_list),
            total=len(args_list),
            desc="Processing files"
        ))
    
    # Collect results
    for file_path, num_frames, success, error_msg in results:
        if success:
            total_frames += num_frames
        else:
            failed_files.append((file_path, error_msg))
    
    # Report failures
    if failed_files:
        print(f"\n⚠ {len(failed_files)} files failed:")
        for fp, err in failed_files[:5]:  # Show first 5
            print(f"  - {fp}: {err.split(chr(10))[0]}")
        if len(failed_files) > 5:
            print(f"  ... and {len(failed_files) - 5} more")
    
    # Copy metadata
    shutil.copy(data_path / "metadata.csv", output_dir / "metadata.csv")
    
    # Save configs for reproducibility
    config_dir = output_dir / "config"
    config_dir.mkdir(exist_ok=True)
    
    with open(config_dir / "dataset_config.yaml", 'w') as f:
        yaml.dump({'dataset': dataset_config}, f, default_flow_style=False, sort_keys=False)
    
    with open(config_dir / "preprocess_config.yaml", 'w') as f:
        yaml.dump({'preprocess': preprocess_cfg}, f, default_flow_style=False, sort_keys=False)
    
    with open(config_dir / "feature_config.yaml", 'w') as f:
        yaml.dump(feature_cfg, f, default_flow_style=False, sort_keys=False)
    
    print(f"  Configs saved to: {config_dir}")
    
    # Load and update dataset_info.json
    if (data_path / "dataset_info.json").exists():
        with open(data_path / "dataset_info.json", 'r') as f:
            dataset_info = json.load(f)
        
        dataset_info['channel_names'] = selected_channels
        dataset_info['num_channels'] = len(selected_channels)
        
        with open(output_dir / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
    else:
        print(f"Warning: dataset_info.json not found at {data_path / 'dataset_info.json'}")
    
    # Save feature info
    feature_types = [feat.get('name') for feat in feature_cfg.get('features', [])]
    feature_info = {
        "feature_types": feature_types,
        "num_files": len(file_paths),
        "num_files_processed": len(file_paths) - len(failed_files),
        "total_frames": total_frames,
        "extraction_timestamp": timestamp,
        "num_workers": num_workers,
    }
    with open(output_dir / "feature_info.json", 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print(f"\n✓ Features saved to: {output_dir}")
    print(f"  Processed: {len(file_paths) - len(failed_files)}/{len(file_paths)} files, {total_frames} frames")
    
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
    
    # Set defaults
    if args.save_path is None:
        args.save_path = ROOT_PATH / "data" / "processed"
    args.save_path.mkdir(parents=True, exist_ok=True)
    
    num_workers = args.workers if args.workers else cpu_count()
    
    # Load configs
    if args.preprocess_config_path is None or not args.preprocess_config_path.exists():
        preprocess_cfg = {'filters': [], 'norms': [], 'miscs': []}
        print("No preprocess config provided, using empty preprocessing pipeline")
    else:
        with open(args.preprocess_config_path, 'r') as f:
            preprocess_cfg_raw = yaml.safe_load(f) or {}
        preprocess_cfg = preprocess_cfg_raw.get('preprocess', preprocess_cfg_raw)
    
    with open(args.feature_config_path, 'r') as f:
        feature_cfg = yaml.safe_load(f)
    
    with open(args.dataset_config_path, 'r') as f:
        dataset_config_raw = yaml.safe_load(f)
    
    dataset_config = dataset_config_raw.get('dataset', dataset_config_raw)
    
    # Validate dataset type
    dataset_type = dataset_config.get('type')
    if dataset_type != 'OrionAEFrameDataset':
        raise ValueError(
            f"Feature extraction currently only supports 'OrionAEFrameDataset', got '{dataset_type}'."
        )
    
    # Load metadata to get file paths and series
    import pandas as pd
    metadata = pd.read_csv(args.frame_path / 'metadata.csv')
    file_paths = metadata['file_path'].tolist()
    file_series = metadata['series'].tolist()
    
    # Get channel configuration
    CHANNELS = ['A', 'B', 'C', 'D']
    selected_channels = dataset_config.get('channels', CHANNELS)
    channel_indices = [CHANNELS.index(ch) for ch in selected_channels]
    
    print(f"Dataset: {args.frame_path.name}")
    print(f"Channels: {selected_channels}")
    print(f"Workers: {num_workers}")
    
    # Extract and save features with parallel processing
    output_dir = save_features_dataset_parallel(
        data_path=args.frame_path,
        file_paths=file_paths,
        file_series=file_series,
        channel_indices=channel_indices,
        selected_channels=selected_channels,
        dataset_config=dataset_config,
        preprocess_cfg=preprocess_cfg,
        feature_cfg=feature_cfg,
        save_path=args.save_path,
        num_workers=num_workers,
    )
    
    print(f"\n✓ Feature extraction complete!")
    print(f"  Output directory: {output_dir}")
    return output_dir


if __name__ == "__main__":
    main()

