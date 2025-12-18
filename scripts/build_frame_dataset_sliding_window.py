# scripts/build_frame_dataset.py
"""
Build a dataset of AE frames from raw .mat files.

Each .mat file is a 1-second chunk from a continuous recorded stream at a specific load.
The dataset is built by:
1. Loading each 1-second chunk (.mat file)
2. Segmenting each chunk into frames of a fixed duration
3. Saving all frames from a chunk as a single .npy file

The metadata is saved as a .csv file and a .json file.

The dataset is saved in the following structure:
- data/raw/segmented/
  - data/
    - file_id.npy  (contains all frames from one 1-second chunk)
    - file_id.npy
    - ...
  - metadata.csv
  - dataset_info.json
"""

from pathlib import Path
import logging
import argparse
import json
from datetime import datetime

import scipy.io
import pandas as pd
import numpy as np
from tqdm import tqdm

SAMPLING_FREQUENCY_HZ = 5e6  # 5 MHz
MAX_FRAME_DURATION_MS = 1000.0  # 1000 ms

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_timestamp_for_sorting(filename: str) -> str:
    """
    Extract timestamp string from filename for sorting purposes.
    Format: salves_out_XXcNm_Y_Fs5MHz_Tf1s_YYYY-MM-DD-HH-MM-SS_1.mat
    Returns: YYYY-MM-DD-HH-MM-SS (as string for chronological sorting)
    """
    parts = filename.split('_')
    return parts[-2]  # YYYY-MM-DD-HH-MM-SS


def get_measurement_files(input_dir: Path) -> list[Path]:
    """
    Get all measurement files from the source directory, sorted chronologically.
    Each .mat file is a 1-second chunk from a continuous stream, so we sort by timestamp.
    """
    files = list(input_dir.glob("measurementSeries_*/*/*.mat"))
    # Sort by (series, load, timestamp) to ensure chronological order within each (series, load) group
    def sort_key(path: Path) -> tuple:
        series = path.parent.parent.name.split('_')[-1]
        load = path.parent.name
        timestamp_str = extract_timestamp_for_sorting(path.name)
        return (series, load, timestamp_str)
    
    return sorted(files, key=sort_key)


def ms_to_samples(ms: float, fs_hz: float) -> int:
    """Convert milliseconds to samples."""
    return int(ms * fs_hz / 1000)


def extract_timestamp(filename: str) -> str:
    """
    Extract timestamp from filename.
    Format: salves_out_XXcNm_Y_Fs5MHz_Tf1s_YYYY-MM-DD-HH-MM-SS_1.mat
    Output: YYYY-MM-DD HH:MM:SS
    """
    parts = filename.split('_')
    timestamp_str = parts[-2]  # YYYY-MM-DD-HH-MM-SS (second to last part before .mat)
    # Split by dashes: ['YYYY', 'MM', 'DD', 'HH', 'MM', 'SS']
    timestamp_parts = timestamp_str.split('-')
    date_part = '-'.join(timestamp_parts[:3])  # YYYY-MM-DD
    time_part = ':'.join(timestamp_parts[3:])  # HH:MM:SS
    return f"{date_part} {time_part}"


def segment_stream(stream_data: np.ndarray, frame_length: int, overlap: float) -> np.ndarray:
    """
    Divide continuous stream into fixed-length frames.
    
    Args:
        stream_data: (N, num_channels) array - multi-channel signal
        frame_length: length of one frame in samples
        overlap: [0.0, 1.0) - overlap ratio between frames
    
    Returns:
        np.ndarray: array of frames with shape (num_frames, frame_length, num_channels)
    """
    stride = int(frame_length * (1 - overlap))
    frames = []

    for start_idx in range(0, len(stream_data) - frame_length + 1, stride):
        frame = stream_data[start_idx:start_idx + frame_length]
        frames.append(frame)
    
    return np.array(frames)  # (num_frames, frame_length, num_channels)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert .mat files to segmented .npy files"
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=PROJECT_ROOT / 'data' / 'raw' / 'original',
        help='Path to directory containing raw .mat files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=PROJECT_ROOT / 'data' / 'raw' ,
        help='Path to output directory'
    )
    parser.add_argument(
        '--frame-ms',
        type=float,
        default=10.0,
        help='Frame duration in milliseconds (default: 10.0 ms) with maximum value of 1000 ms'
    )
    parser.add_argument(
        '--overlap',
        type=float,
        default=0.0,
        help='Overlap ratio between frames [0.0, 1.0) (default: 0.0 = no overlap)'
    )
    parser.add_argument(
        '--channels',
        type=str,
        nargs='+',
        default=['A', 'B', 'C', 'D'],
        help='Channels to extract from .mat files (default: A B C D)'
    )
    return parser.parse_args()


def convert_and_segment_dataset(
    source_root: Path,
    target_root: Path,
    frame_ms: float,
    overlap: float,
    channels: list[str]
):
    """
    Convert raw .mat files to segmented .npy files.
    
    Args:
        source_root: Path to source directory containing .mat files
        target_root: Path to output directory
        frame_ms: Frame duration in milliseconds
        overlap: Overlap ratio between frames
        channels: List of channel names to extract
    """
    
    # Mapping load string -> (class, value)
    load_map = {
        "05cNm": (0, 0.05), "10cNm": (1, 0.10), "20cNm": (2, 0.20),
        "30cNm": (3, 0.30), "40cNm": (4, 0.40), "50cNm": (5, 0.50), "60cNm": (6, 0.60)
    }
    
    # Convert ms to samples
    frame_length = ms_to_samples(frame_ms, SAMPLING_FREQUENCY_HZ)
    
    logger.info(f"Frame duration: {frame_ms} ms -> {frame_length:,} samples")
    logger.info(f"Channels: {channels}")
    logger.info(f"Overlap ratio: {overlap}")
    
    # Create output directory
    target_root.mkdir(parents=True, exist_ok=True)
    data_dir = target_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {target_root}")
    logger.info(f"Created data subdirectory: {data_dir}")
    
    # List to store metadata
    metadata = []
    
    # Counter for sequential chunks within each (series, load_class) combination
    # Each .mat file is a 1-second chunk from a continuous stream, numbered sequentially
    series_chunk_counter = {}  # (series, load_class) -> chunk_number
    
    # Get all measurement files
    data_files = get_measurement_files(source_root)
    logger.info(f"Found {len(data_files)} measurement files")
    
    # Convert and segment
    for mat_path in tqdm(data_files, desc="Converting & Segmenting"):
        # Parse from path: measurementSeries_B / 05cNm / filename.mat
        series_name = mat_path.parent.parent.name.split('_')[-1]
        load_str = mat_path.parent.name
        
        try:
            # Load .mat
            mat = scipy.io.loadmat(mat_path)
            
            # Stack selected channels: (N, num_channels)
            signal = np.column_stack([
                mat[ch].reshape(-1) for ch in channels
            ]).astype(np.float32)
            
            logger.debug(f"Loaded {mat_path.name}: shape={signal.shape}")
            
            # Validation: verify all channels exist and have same length
            for ch in channels:
                if ch not in mat:
                    raise KeyError(f"Channel '{ch}' not found in {mat_path.name}")
                if mat[ch].shape[0] != signal.shape[0]:
                    raise ValueError(f"Channel '{ch}' has mismatched length in {mat_path.name}")
            
            # Extract metadata
            load_class, load_val = load_map[load_str]
            timestamp = extract_timestamp(mat_path.name)
            
            # Segment stream into frames: (num_frames, frame_length, num_channels)
            segmented_data = segment_stream(signal, frame_length, overlap)
            num_frames = segmented_data.shape[0]
            
            # Create unique file ID: number chunks sequentially within this (series, load_class) combination
            # Files are already sorted chronologically by get_measurement_files()
            key = (series_name, load_class)
            series_chunk_counter[key] = series_chunk_counter.get(key, 0) + 1
            chunk_num = series_chunk_counter[key]
            file_id = f"{series_name}_{load_str.replace('cNm', '')}_{chunk_num:03d}"
            npy_path = target_root / "data" / f"{file_id}.npy"
            
            # Save all frames from this .mat file as single .npy
            np.save(npy_path, segmented_data)
            
            # Append metadata (only file-specific fields, dataset-level fields go to dataset_info.json)
            metadata.append({
                "file_id": file_id,
                "file_path": str(npy_path.relative_to(target_root)),
                "series": series_name,
                "chunk": chunk_num,
                "load_class": load_class,
                "load_val": load_val,
                "num_frames": num_frames,
                "timestamp": timestamp,
                "original_file": mat_path.name
            })
        
        except KeyError as e:
            logger.error(f"Missing channel in {mat_path}: {e}")
            continue
        except Exception as e:
            logger.error(f"Error processing {mat_path}: {e}")
            continue
    
    # Save metadata to CSV
    df = pd.DataFrame(metadata)
    csv_path = target_root / "metadata.csv"
    df.to_csv(csv_path, index=False)
    
    logger.info(f"✓ Conversion complete!")
    logger.info(f"    Total files: {len(df)}")
    logger.info(f"    Frame duration: {frame_ms} ms ({frame_length:,} samples)")
    logger.info(f"    Series: {sorted(df['series'].unique().tolist())}")
    logger.info(f"    Loads: {sorted(df['load_class'].unique().tolist())}")
    logger.info(f"    Metadata saved to: {csv_path}")
    
    # Save dataset info (dataset_info.json)
    dataset_info = {
        "frame_duration_ms": frame_ms,
        "frame_length_samples": frame_length,
        "sampling_frequency_hz": int(SAMPLING_FREQUENCY_HZ),
        "overlap_ratio": overlap,
        "num_channels": len(channels),
        "channel_names": channels,
        "num_files": len(df),
        "series": sorted(df['series'].unique().tolist()),
        "loads": {
            load_class: load_val for load_str, (load_class, load_val) in load_map.items()
        }
    }
    
    info_path = target_root / "dataset_info.json"
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    logger.info(f"    Dataset info saved to: {info_path}")


def main():
    args = parse_args()
    
    # Create target_root with format: output-dir / segmented_{frame_duration}_{datetime}
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    frame_ms_str = str(args.frame_ms).replace('.', '_')
    overlap_str = f"{args.overlap:.2f}".replace('.', '_')
    channels_str = '_'.join(args.channels)
    target_root = args.output_dir / f"segmented_ms_{frame_ms_str}_o_{overlap_str}_c_{channels_str}_{datetime_str}"
    
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Target root: {target_root}")
    logger.info(f"Frame duration: {args.frame_ms} ms")
    logger.info(f"Overlap ratio: {args.overlap}")
    logger.info(f"Channels: {args.channels}")

    if args.frame_ms > MAX_FRAME_DURATION_MS:
        logger.error(f"Frame duration is greater than maximum value of {MAX_FRAME_DURATION_MS} ms")
        raise ValueError(f"Frame duration is greater than maximum value of {MAX_FRAME_DURATION_MS} ms")
    
    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    
    convert_and_segment_dataset(
        source_root=args.input_dir,
        target_root=target_root,
        frame_ms=args.frame_ms,
        overlap=args.overlap,
        channels=args.channels
    )


if __name__ == "__main__":
    main()