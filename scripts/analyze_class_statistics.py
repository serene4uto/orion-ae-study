#!/usr/bin/env python3
"""
analyze_class_statistics.py - Statistical analysis of raw signals for user-specified class groups

This script calculates statistical information (mean, std, min, max) of raw signals
for user-specified groups of classes at each experimental session (series).
Useful for:
  - Computing baseline statistics for Z-normalization
  - Analyzing signal characteristics across different conditions
  - Comparing class distributions across sessions

Usage:
------
    python scripts/analyze_class_statistics.py \\
        --frame-path <path_to_frame_dataset> \\
        --classes <class_ids...> \\
        [--channels <channel_names...>] \\
        [--output <output_file.json>]

Arguments:
----------
    --frame-path    Path to segmented frame dataset directory
    --classes       Space-separated list of class IDs to analyze (e.g., 0 1 2)
    --channels      (Optional) Space-separated list of channels to analyze (default: all)
    --output        (Optional) Path to save results as JSON file

Example:
--------
    # Analyze classes 0, 1, 2 (baseline classes) for all channels
    python scripts/analyze_class_statistics.py \\
        --frame-path data/raw/segmented_cycles_positive_c1_l42373_c_A_B_C_D_20260107_205752 \\
        --classes 0 1 2

    # Analyze specific channels A and B for classes 5 and 6
    python scripts/analyze_class_statistics.py \\
        --frame-path data/raw/segmented_cycles_positive_c1_l42373_c_A_B_C_D_20260107_205752 \\
        --classes 5 6 \\
        --channels A B

    # Save results to JSON for later use
    python scripts/analyze_class_statistics.py \\
        --frame-path data/raw/segmented_cycles_positive_c1_l42373_c_A_B_C_D_20260107_205752 \\
        --classes 0 1 2 \\
        --output baseline_stats.json

Output:
-------
    Prints statistics per series and overall average:
    
    Series B (classes [0, 1, 2]):
      Channel A: mean=0.5801, std=4.7861, min=-45.23, max=52.18
      Channel B: mean=2.1396, std=6.4908, min=-62.51, max=71.34
      ...
    
    Series C (classes [0, 1, 2]):
      ...
    
    Overall Average (all series, classes [0, 1, 2]):
      Channel A: mean=0.5198, std=4.8150
      ...
"""

from pathlib import Path
import sys

# Get the project root from the script's location
ROOT_PATH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_PATH))

import argparse
import numpy as np
import pandas as pd
import json
from collections import defaultdict
from tqdm import tqdm


# Available channels in the dataset
AVAILABLE_CHANNELS = ['A', 'B', 'C']


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate statistical information for user-specified class groups at each session"
    )
    
    parser.add_argument(
        '--frame-path',
        type=Path,
        required=True,
        help='Path to directory containing frame dataset'
    )
    parser.add_argument(
        '--classes',
        type=int,
        nargs='+',
        required=True,
        help='Space-separated list of class IDs to analyze (e.g., 0 1 2)'
    )
    parser.add_argument(
        '--channels',
        type=str,
        nargs='+',
        default=None,
        help='Space-separated list of channels to analyze (default: all available)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='(Optional) Path to save results as JSON file'
    )
    parser.add_argument(
        '--per-channel',
        action='store_true',
        help='Calculate statistics per channel (default: aggregate all channels)'
    )
    
    return parser.parse_args()


def load_metadata(frame_path: Path) -> pd.DataFrame:
    """Load metadata from the frame dataset."""
    metadata_path = frame_path / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    return pd.read_csv(metadata_path)


def get_channel_indices(channels: list, available_channels: list = AVAILABLE_CHANNELS) -> list:
    """Get indices for selected channels."""
    indices = []
    for ch in channels:
        if ch not in available_channels:
            raise ValueError(f"Invalid channel '{ch}'. Available channels: {available_channels}")
        indices.append(available_channels.index(ch))
    return indices


def calculate_statistics(frame_path: Path, metadata: pd.DataFrame, 
                         target_classes: list, channels: list, 
                         per_channel: bool = False) -> dict:
    """
    Calculate statistics for specified classes at each series.
    
    Returns:
        Dictionary with structure:
        {
            'series_stats': {
                'B': {'mean': [...], 'std': [...], 'min': [...], 'max': [...], 'count': int},
                'C': {...},
                ...
            },
            'overall_stats': {'mean': [...], 'std': [...], ...},
            'config': {'classes': [...], 'channels': [...], 'per_channel': bool}
        }
    """
    # Get channel indices
    channel_indices = get_channel_indices(channels)
    
    # Filter metadata for target classes
    filtered_metadata = metadata[metadata['load_class'].isin(target_classes)]
    
    if len(filtered_metadata) == 0:
        raise ValueError(f"No samples found for classes {target_classes}")
    
    # Get unique series
    series_list = sorted(filtered_metadata['series'].unique())
    
    # Initialize storage for running statistics per series
    # Using Welford's online algorithm for numerical stability
    series_stats = {}
    
    print(f"\nAnalyzing classes {target_classes} across {len(series_list)} series...")
    print(f"Channels: {channels}")
    print(f"Per-channel statistics: {per_channel}\n")
    
    for series in series_list:
        series_data = filtered_metadata[filtered_metadata['series'] == series]
        
        # Collect all samples for this series
        all_samples = []
        
        print(f"Processing Series {series}...")
        for _, row in tqdm(series_data.iterrows(), total=len(series_data), desc=f"  Series {series}"):
            file_path = frame_path / row['file_path']
            
            if not file_path.exists():
                print(f"  Warning: File not found: {file_path}")
                continue
            
            # Load frames: shape (num_frames, time_steps, num_channels)
            frames = np.load(file_path)
            
            # Select channels: shape (num_frames, time_steps, len(channels))
            frames = frames[:, :, channel_indices]
            
            # Reshape to (num_frames * time_steps, len(channels)) for statistics
            # or (num_frames * time_steps * len(channels),) if not per_channel
            if per_channel:
                # Keep channels separate: (samples, channels)
                samples = frames.reshape(-1, len(channels))
            else:
                # Flatten all: (samples,)
                samples = frames.flatten()
            
            all_samples.append(samples)
        
        if len(all_samples) == 0:
            print(f"  Warning: No valid samples for series {series}")
            continue
        
        # Concatenate all samples
        all_samples = np.concatenate(all_samples, axis=0)
        
        # Calculate statistics
        if per_channel:
            series_stats[series] = {
                'mean': all_samples.mean(axis=0).tolist(),
                'std': all_samples.std(axis=0).tolist(),
                'min': all_samples.min(axis=0).tolist(),
                'max': all_samples.max(axis=0).tolist(),
                'count': len(all_samples)
            }
        else:
            series_stats[series] = {
                'mean': float(all_samples.mean()),
                'std': float(all_samples.std()),
                'min': float(all_samples.min()),
                'max': float(all_samples.max()),
                'count': len(all_samples)
            }
    
    # Calculate overall statistics (weighted average across series)
    if per_channel:
        total_count = sum(s['count'] for s in series_stats.values())
        overall_mean = np.zeros(len(channels))
        overall_var = np.zeros(len(channels))
        overall_min = np.full(len(channels), np.inf)
        overall_max = np.full(len(channels), -np.inf)
        
        # Calculate weighted mean
        for series, stats in series_stats.items():
            weight = stats['count'] / total_count
            overall_mean += np.array(stats['mean']) * weight
            overall_min = np.minimum(overall_min, stats['min'])
            overall_max = np.maximum(overall_max, stats['max'])
        
        # Calculate pooled variance (approximation)
        for series, stats in series_stats.items():
            weight = stats['count'] / total_count
            series_mean = np.array(stats['mean'])
            series_var = np.array(stats['std']) ** 2
            # Pooled variance = weighted avg of (variance + (mean - overall_mean)^2)
            overall_var += weight * (series_var + (series_mean - overall_mean) ** 2)
        
        overall_stats = {
            'mean': overall_mean.tolist(),
            'std': np.sqrt(overall_var).tolist(),
            'min': overall_min.tolist(),
            'max': overall_max.tolist(),
            'count': total_count
        }
    else:
        total_count = sum(s['count'] for s in series_stats.values())
        overall_mean = sum(s['mean'] * s['count'] for s in series_stats.values()) / total_count
        
        # Pooled standard deviation
        overall_var = 0
        for stats in series_stats.values():
            weight = stats['count'] / total_count
            overall_var += weight * (stats['std'] ** 2 + (stats['mean'] - overall_mean) ** 2)
        
        overall_stats = {
            'mean': overall_mean,
            'std': np.sqrt(overall_var),
            'min': min(s['min'] for s in series_stats.values()),
            'max': max(s['max'] for s in series_stats.values()),
            'count': total_count
        }
    
    return {
        'series_stats': series_stats,
        'overall_stats': overall_stats,
        'config': {
            'classes': target_classes,
            'channels': channels,
            'per_channel': per_channel
        }
    }


def print_statistics(results: dict):
    """Print statistics in a formatted table."""
    config = results['config']
    classes = config['classes']
    channels = config['channels']
    per_channel = config['per_channel']
    
    print("\n" + "=" * 70)
    print(f"STATISTICAL ANALYSIS - Classes: {classes}")
    print("=" * 70)
    
    # Print per-series statistics
    for series, stats in sorted(results['series_stats'].items()):
        print(f"\nSeries {series} (n={stats['count']:,} samples):")
        print("-" * 50)
        
        if per_channel:
            for i, ch in enumerate(channels):
                print(f"  Channel {ch}:")
                print(f"    mean = {stats['mean'][i]:12.6f}")
                print(f"    std  = {stats['std'][i]:12.6f}")
                print(f"    min  = {stats['min'][i]:12.6f}")
                print(f"    max  = {stats['max'][i]:12.6f}")
        else:
            print(f"  mean = {stats['mean']:12.6f}")
            print(f"  std  = {stats['std']:12.6f}")
            print(f"  min  = {stats['min']:12.6f}")
            print(f"  max  = {stats['max']:12.6f}")
    
    # Print overall statistics
    overall = results['overall_stats']
    print("\n" + "=" * 70)
    print(f"OVERALL AVERAGE (all series combined, n={overall['count']:,} samples)")
    print("=" * 70)
    
    if per_channel:
        for i, ch in enumerate(channels):
            print(f"\n  Channel {ch}:")
            print(f"    mean = {overall['mean'][i]:12.6f}")
            print(f"    std  = {overall['std'][i]:12.6f}")
            print(f"    min  = {overall['min'][i]:12.6f}")
            print(f"    max  = {overall['max'][i]:12.6f}")
    else:
        print(f"\n  mean = {overall['mean']:12.6f}")
        print(f"  std  = {overall['std']:12.6f}")
        print(f"  min  = {overall['min']:12.6f}")
        print(f"  max  = {overall['max']:12.6f}")
    
    print("\n")


def print_znorm_config(results: dict):
    """Print statistics in a format ready for Z-normalization config."""
    config = results['config']
    per_channel = config['per_channel']
    channels = config['channels']
    
    print("\n" + "=" * 70)
    print("Z-NORMALIZATION CONFIG (copy to your dataset config)")
    print("=" * 70)
    
    if per_channel:
        # Per-series Z-norm config
        print("\n# SeriesZScoreNorm parameters:")
        print("# series_params:")
        for series, stats in sorted(results['series_stats'].items()):
            print(f"#   '{series}':")
            print(f"#     mean: {stats['mean']}")
            print(f"#     std: {stats['std']}")
        
        # Overall Z-norm config
        overall = results['overall_stats']
        print("\n# ZScoreNorm parameters (overall):")
        print(f"#   mean: {overall['mean']}")
        print(f"#   std: {overall['std']}")
    else:
        print("\n# Note: Statistics are aggregated across all channels.")
        print("# For per-channel Z-norm, re-run with --per-channel flag.")
        overall = results['overall_stats']
        print(f"\n# Overall mean: {overall['mean']:.6f}")
        print(f"# Overall std:  {overall['std']:.6f}")


def main():
    args = parse_args()
    
    # Validate frame path
    if not args.frame_path.exists():
        print(f"Error: Frame path does not exist: {args.frame_path}")
        sys.exit(1)
    
    # Set default channels if not specified
    channels = args.channels if args.channels else AVAILABLE_CHANNELS.copy()
    
    # Load metadata
    metadata = load_metadata(args.frame_path)
    
    # Validate classes exist
    available_classes = sorted(metadata['load_class'].unique())
    invalid_classes = [c for c in args.classes if c not in available_classes]
    if invalid_classes:
        print(f"Error: Invalid class IDs: {invalid_classes}")
        print(f"Available classes: {available_classes}")
        sys.exit(1)
    
    # Calculate statistics
    results = calculate_statistics(
        frame_path=args.frame_path,
        metadata=metadata,
        target_classes=args.classes,
        channels=channels,
        per_channel=args.per_channel
    )
    
    # Print results
    print_statistics(results)
    print_znorm_config(results)
    
    # Save to JSON if output path specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()

