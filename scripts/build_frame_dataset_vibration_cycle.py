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

import scipy
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

def find_zero_crossings(signal, direction='both'):
    """
    Find zero-crossing indices in a signal.
    
    Args:
        signal: 1D array of signal values
        direction: 'positive' (negative to positive), 'negative' (positive to negative), 
                   or 'both' (all crossings)
    
    Returns:
        Array of indices where zero-crossings occur
    """
    # Check for positive-going crossings: negative before, positive after
    positive_crossings = np.where((signal[:-1] < 0) & (signal[1:] > 0))[0]
    
    # Check for negative-going crossings: positive before, negative after
    negative_crossings = np.where((signal[:-1] > 0) & (signal[1:] < 0))[0]
    
    if direction == 'positive':
        return positive_crossings
    elif direction == 'negative':
        return negative_crossings
    else:  # 'both'
        # Combine and sort
        all_crossings = np.concatenate([positive_crossings, negative_crossings])
        return np.sort(all_crossings)

def detect_vibration_cycles_with_peaks(signal, start_with_positive=True, min_cycle_length=None, max_cycle_length=None):
    """
    Detect vibration cycles with peak identification.
    
    Args:
        signal: 1D array of signal values
        start_with_positive: 
            - True: Cycle is 0 (neg→pos) → pos peak → 0 (pos→neg) → neg peak → 0 (neg→pos)
            - False: Cycle is 0 (pos→neg) → neg peak → 0 (neg→pos) → pos peak → 0 (pos→neg)
        min_cycle_length: Minimum samples per cycle (optional filter)
        max_cycle_length: Maximum samples per cycle (optional filter)
    
    Returns:
        List of tuples: (start_idx, first_peak_idx, mid_zero_idx, second_peak_idx, end_idx, cycle_length)
        - If start_with_positive=True: (start, pos_peak, mid_zero, neg_peak, end, length)
        - If start_with_positive=False: (start, neg_peak, mid_zero, pos_peak, end, length)
    """
    # Find positive-going (neg→pos) and negative-going (pos→neg) zero-crossings
    positive_crossings = find_zero_crossings(signal, direction='positive')
    negative_crossings = find_zero_crossings(signal, direction='negative')
    
    if len(positive_crossings) == 0 or len(negative_crossings) == 0:
        return []
    
    cycles = []
    
    if start_with_positive:
        # Pattern: 0 (neg→pos) → pos peak → 0 (pos→neg) → neg peak → 0 (neg→pos)
        starting_crossings = positive_crossings
        mid_crossings = negative_crossings
        ending_crossings = positive_crossings
    else:
        # Pattern: 0 (pos→neg) → neg peak → 0 (neg→pos) → pos peak → 0 (pos→neg)
        starting_crossings = negative_crossings
        mid_crossings = positive_crossings
        ending_crossings = negative_crossings
    
    for i, start_idx in enumerate(starting_crossings):
        # Find the next mid zero-crossing after start
        next_mid = mid_crossings[mid_crossings > start_idx]
        if len(next_mid) == 0:
            continue
        mid_zero_idx = next_mid[0]
        
        # Find the next ending zero-crossing after mid
        next_end = ending_crossings[ending_crossings > mid_zero_idx]
        if len(next_end) == 0:
            continue
        end_idx = next_end[0]
        
        cycle_length = end_idx - start_idx
        
        # Optional filtering
        if min_cycle_length and cycle_length < min_cycle_length:
            continue
        if max_cycle_length and cycle_length > max_cycle_length:
            continue
        
        # Find peaks within cycle segments
        if start_with_positive:
            # First segment: start to mid_zero (should contain positive peak)
            first_segment = signal[start_idx:mid_zero_idx]
            first_peak_local = np.argmax(first_segment)  # Positive peak
            first_peak_idx = start_idx + first_peak_local
            
            # Second segment: mid_zero to end (should contain negative peak)
            second_segment = signal[mid_zero_idx:end_idx]
            second_peak_local = np.argmin(second_segment)  # Negative peak
            second_peak_idx = mid_zero_idx + second_peak_local
        else:
            # First segment: start to mid_zero (should contain negative peak)
            first_segment = signal[start_idx:mid_zero_idx]
            first_peak_local = np.argmin(first_segment)  # Negative peak
            first_peak_idx = start_idx + first_peak_local
            
            # Second segment: mid_zero to end (should contain positive peak)
            second_segment = signal[mid_zero_idx:end_idx]
            second_peak_local = np.argmax(second_segment)  # Positive peak
            second_peak_idx = mid_zero_idx + second_peak_local
        
        cycles.append((start_idx, first_peak_idx, mid_zero_idx, second_peak_idx, end_idx, cycle_length))
    
    return cycles

def preprocess_vibration_signal(
    raw_signal: np.ndarray,
    sampling_frequency_hz: float = SAMPLING_FREQUENCY_HZ,
    cutoff_frequency_hz: float = 1000.0,
    filter_order: int = 4
) -> np.ndarray:
    """
    Preprocess vibration signal for cycle detection/alignment.
    
    Applies a low-pass Butterworth filter to remove high-frequency noise
    while preserving the vibration cycle structure.
    
    Args:
        raw_signal: Raw vibration signal (1D array)
        sampling_frequency_hz: Sampling frequency in Hz (default: 5e6)
        cutoff_frequency_hz: Low-pass filter cutoff frequency in Hz (default: 1000.0)
        filter_order: Butterworth filter order (default: 4)
    
    Returns:
        Filtered vibration signal (1D array)
    """
    # Calculate normalized cutoff frequency (Nyquist normalized)
    nyquist = sampling_frequency_hz / 2
    normal_cutoff = cutoff_frequency_hz / nyquist
    
    # Design Butterworth low-pass filter
    b, a = scipy.signal.butter(filter_order, normal_cutoff, btype='low', analog=False)
    
    # Apply zero-phase filtering (filtfilt) to avoid phase distortion
    filtered_signal = scipy.signal.filtfilt(b, a, raw_signal)
    
    return filtered_signal

def validate_and_group_cycles(
    cycle_list: list[tuple],
    skip: int,
    cycles_per_frame: int
) -> list[int]:
    """
    Group consecutive cycles and return frame start indices.
    Frames will be created with fixed length based on user input (cycles_length * cycles_per_frame).
    
    Args:
        cycle_list: List of cycle tuples (start_idx, ..., end_idx, cycle_length)
        skip: Number of cycles to skip before creating frames
        cycles_per_frame: Number of consecutive cycles per frame
    
    Returns:
        List of frame start indices (start_idx of first cycle in each group)
    """
    frame_start_idx = []
    
    # Iterate through consecutive groups of cycles (step by cycles_per_frame to avoid overlaps)
    # Example: skip=3, cycles_per_frame=3 -> groups: (3,4,5), (6,7,8), (9,10,11), ...
    for i in range(skip, len(cycle_list) - cycles_per_frame + 1, cycles_per_frame):
        # Check if we have enough cycles for a complete group
        if i + cycles_per_frame > len(cycle_list):
            break
        
        # Verify cycles are consecutive
        # Cycle tuple: (start_idx, first_peak_idx, mid_zero_idx, second_peak_idx, end_idx, cycle_length)
        # end_idx is at index [4], start_idx is at index [0]
        is_consecutive = True
        for j in range(i, i + cycles_per_frame - 1):
            if cycle_list[j][4] != cycle_list[j + 1][0]:  # end of cycle j != start of cycle j+1
                is_consecutive = False
                break
        
        if not is_consecutive:
            logger.warning(
                f"Skipping non-consecutive cycle group starting at index {i}: "
                f"cycle[{i}][4]={cycle_list[i][4]} != cycle[{i+1}][0]={cycle_list[i+1][0]}"
            )
            continue
        
        # Use the start index of the first cycle in the group
        # Frame will be created with fixed length: start_idx + cycles_length * cycles_per_frame
        first_cycle_start = cycle_list[i][0]
        frame_start_idx.append(first_cycle_start)
    
    return frame_start_idx


def segment_stream(stream_data: np.ndarray, frame_length: int, frame_start_idx: list[int]) -> np.ndarray:
    """
    Divide continuous stream into fixed-length frames.
    
    Args:
        stream_data: (N, num_channels) array - multi-channel signal
        frame_length: length of one frame in samples
        frame_start_idx: list of first indices of frames
    
    Returns:
        np.ndarray: array of frames with shape (num_frames, frame_length, num_channels)
    """
    frames = []
    signal_length = len(stream_data)
    for start_idx in frame_start_idx:
        end_idx = start_idx + frame_length
        if end_idx > signal_length:
            # Skip frames that would go beyond signal length
            continue
        frame = stream_data[start_idx:end_idx]
        if len(frame) == frame_length:  # Only add complete frames
            frames.append(frame)
    
    return np.array(frames) if frames else np.empty((0, frame_length, stream_data.shape[1]))


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
        '--cycles-per-frame',
        type=int,
        default=1,
        help='Number of cycles per frame'
    )
    parser.add_argument(
        '--cycle-start-phase',
        type=str,
        default='positive',
        choices=['positive', 'negative', 'both'],
        help='Phase of the cycle to start with'
    )
    parser.add_argument(
        '--cycles-length',
        type=int,
        default=42373, # or 41667 ~ 120Hz
        help='Selected cycle length to use for the dataset'
    )
    parser.add_argument(
        '--skip-cycles',
        type=int,
        nargs='+',
        default=[0],
        help='Number of cycles to skip before creating the first frame. '
             'If --cycle-start-phase is "both", specify 2 values [positive, negative]. '
             'If --cycle-start-phase is "positive" or "negative", specify 1 value.'
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
    cycle_start_phase: str,
    skip_cycles: list[int],
    cycles_per_frame: int,
    cycles_length: int,
    channels: list[str]
):
    """
    Convert raw .mat files to segmented .npy files based on vibration cycles.
    
    Args:
        source_root: Path to source directory containing .mat files
        target_root: Path to output directory
        cycle_start_phase: Phase to start cycles ('positive', 'negative', or 'both')
        skip_cycles: Number of cycles to skip before creating frames (1 or 2 values)
        cycles_per_frame: Number of cycles per frame
        cycles_length: Target cycle length in samples
        channels: List of channel names to extract
    """
    
    # Mapping load string -> (class, value)
    load_map = {
        "05cNm": (0, 0.05), "10cNm": (1, 0.10), "20cNm": (2, 0.20),
        "30cNm": (3, 0.30), "40cNm": (4, 0.40), "50cNm": (5, 0.50), "60cNm": (6, 0.60)
    }

    logger.info(f"Channels: {channels}")
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

            # Preprocess vibrometer signal for cycle detection/alignment
            raw_vibrometer = mat['D'].squeeze()
            vibrometer_signal = preprocess_vibration_signal(
                raw_vibrometer,
                sampling_frequency_hz=SAMPLING_FREQUENCY_HZ,
                cutoff_frequency_hz=1000.0,
                filter_order=4
            )
            if cycle_start_phase == 'positive':
                cycles = [detect_vibration_cycles_with_peaks(vibrometer_signal, start_with_positive=True)]
                # cycles_lengths_avg = np.mean([np.mean([c[5] for c in cycle]) for cycle in cycles])
            elif cycle_start_phase == 'negative':
                cycles = [detect_vibration_cycles_with_peaks(vibrometer_signal, start_with_positive=False)]
                # cycles_lengths_avg = np.mean([np.mean([c[5] for c in cycle]) for cycle in cycles])
            elif cycle_start_phase == 'both':
                cycles_positive = detect_vibration_cycles_with_peaks(vibrometer_signal, start_with_positive=True)
                cycles_negative = detect_vibration_cycles_with_peaks(vibrometer_signal, start_with_positive=False)
                cycles = [cycles_positive, cycles_negative]
                # cycles_lengths_avg = np.mean([np.mean([c[5] for c in cycle]) for cycle in cycles])
            else:
                raise ValueError(f"Invalid cycle start phase: {cycle_start_phase}")
            
            # Check if cycles were found
            if cycle_start_phase == 'both':
                if len(cycles) != 2 or len(cycles[0]) == 0 or len(cycles[1]) == 0:
                    logger.warning(f"No cycles found in {mat_path.name}")
                    continue
                # Check if we have enough cycles after skipping
                if len(cycles[0]) < skip_cycles[0] + cycles_per_frame:
                    logger.warning(f"Not enough positive cycles in {mat_path.name}: {len(cycles[0])} < {skip_cycles[0] + cycles_per_frame}")
                    continue
                if len(cycles[1]) < skip_cycles[1] + cycles_per_frame:
                    logger.warning(f"Not enough negative cycles in {mat_path.name}: {len(cycles[1])} < {skip_cycles[1] + cycles_per_frame}")
                    continue
            else:
                if len(cycles) == 0 or len(cycles[0]) == 0:
                    logger.warning(f"No cycles found in {mat_path.name}")
                    continue
                # Check if we have enough cycles after skipping
                if len(cycles[0]) < skip_cycles[0] + cycles_per_frame:
                    logger.warning(f"Not enough cycles in {mat_path.name}: {len(cycles[0])} < {skip_cycles[0] + cycles_per_frame}")
                    continue
            
            logger.debug(f"Using cycle length: {cycles_length} samples ({cycles_length * 1000 / SAMPLING_FREQUENCY_HZ:.2f} ms)")
            frame_length = cycles_length * cycles_per_frame
            frame_ms = frame_length * 1000 / SAMPLING_FREQUENCY_HZ
            logger.info(f"Frame duration: {frame_ms:.2f} ms")

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

            # Build frame start indices based on cycle boundaries
            frame_start_idx = []
            
            if cycle_start_phase == 'both':
                # Handle both positive and negative cycles
                cycles_positive = cycles[0]
                cycles_negative = cycles[1]
                skip_pos = skip_cycles[0]
                skip_neg = skip_cycles[1]
                
                # Create frames starting at positive cycles (group consecutive cycles)
                pos_starts = validate_and_group_cycles(
                    cycles_positive, skip_pos, cycles_per_frame
                )
                frame_start_idx.extend(pos_starts)
                
                # Create frames starting at negative cycles (group consecutive cycles)
                neg_starts = validate_and_group_cycles(
                    cycles_negative, skip_neg, cycles_per_frame
                )
                frame_start_idx.extend(neg_starts)
            else:
                # Handle single phase (positive or negative)
                cycle_list = cycles[0]
                skip = skip_cycles[0]
                
                # Create frames starting at cycle boundaries, grouping consecutive cycles_per_frame cycles
                frame_start_idx = validate_and_group_cycles(
                    cycle_list, skip, cycles_per_frame
                )

            # Segment stream into frames: (num_frames, frame_length, num_channels)
            segmented_data = segment_stream(signal, frame_length, frame_start_idx)
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
                "frame_ms": frame_ms,
                "frame_length": frame_length,
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
    if len(df) > 0:
        # Get frame info from first entry (all should have same frame_length)
        sample_frame_ms = df.iloc[0]['frame_ms']
        sample_frame_length = df.iloc[0]['frame_length']
        logger.info(f"    Frame duration: {sample_frame_ms:.2f} ms ({sample_frame_length:,} samples)")
    logger.info(f"    Series: {sorted(df['series'].unique().tolist())}")
    logger.info(f"    Loads: {sorted(df['load_class'].unique().tolist())}")
    logger.info(f"    Metadata saved to: {csv_path}")
    
    # Save dataset info (dataset_info.json)
    # Get frame info from first entry if available
    sample_frame_ms = float(df.iloc[0]['frame_ms']) if len(df) > 0 else 0.0
    sample_frame_length = int(df.iloc[0]['frame_length']) if len(df) > 0 else 0
    dataset_info = {
        "frame_duration_ms": sample_frame_ms,
        "frame_length_samples": sample_frame_length,
        "sampling_frequency_hz": int(SAMPLING_FREQUENCY_HZ),
        "num_channels": len(channels),
        "channel_names": channels,
        "num_files": int(len(df)),
        "series": sorted(df['series'].unique().tolist()),
        "cycles_length": int(cycles_length),
        "cycles_per_frame": int(cycles_per_frame),
        "cycles_start_phase": cycle_start_phase,
        "cycles_skip": [int(x) for x in skip_cycles],
        "loads": {
            int(load_class): float(load_val) for load_str, (load_class, load_val) in load_map.items()
        }
    }
    
    info_path = target_root / "dataset_info.json"
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    logger.info(f"    Dataset info saved to: {info_path}")


def main():
    args = parse_args()
    
    # Validate skip-cycles based on cycle-start-phase
    if args.cycle_start_phase == 'both':
        if len(args.skip_cycles) == 1:
            # If only one value provided for 'both', use it for both phases
            args.skip_cycles = [args.skip_cycles[0], args.skip_cycles[0]]
            logger.info(f"Using single skip-cycles value for both phases: {args.skip_cycles}")
        elif len(args.skip_cycles) != 2:
            raise ValueError(
                f"When --cycle-start-phase is 'both', --skip-cycles must have 1 or 2 values, "
                f"got {len(args.skip_cycles)}: {args.skip_cycles}"
            )
    else:  # 'positive' or 'negative'
        if len(args.skip_cycles) == 1:
            # Single value is correct
            pass
        elif len(args.skip_cycles) == 2:
            # If 2 values provided but phase is not 'both', use the first one
            logger.warning(
                f"--cycle-start-phase is '{args.cycle_start_phase}' but --skip-cycles has 2 values. "
                f"Using first value: {args.skip_cycles[0]}"
            )
            args.skip_cycles = [args.skip_cycles[0]]
        else:
            raise ValueError(
                f"When --cycle-start-phase is '{args.cycle_start_phase}', --skip-cycles must have 1 value, "
                f"got {len(args.skip_cycles)}: {args.skip_cycles}"
            )
    
    # Create target_root with format: output-dir / segmented_cycles_{datetime}
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    phase_str = args.cycle_start_phase
    cycles_str = f"c{args.cycles_per_frame}_l{args.cycles_length}"
    channels_str = '_'.join(args.channels)
    target_root = args.output_dir / f"segmented_cycles_{phase_str}_{cycles_str}_c_{channels_str}_{datetime_str}"
    
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Target root: {target_root}")
    logger.info(f"Cycles per frame: {args.cycles_per_frame}")
    logger.info(f"Cycle length: {args.cycles_length} samples")
    logger.info(f"Channels: {args.channels}")
    logger.info(f"Cycle start phase: {args.cycle_start_phase}")
    logger.info(f"Skip cycles: {args.skip_cycles}")
    
    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    
    convert_and_segment_dataset(
        source_root=args.input_dir,
        target_root=target_root,
        cycle_start_phase=args.cycle_start_phase,
        skip_cycles=args.skip_cycles,
        cycles_per_frame=args.cycles_per_frame,
        cycles_length=args.cycles_length,
        channels=args.channels
    )


if __name__ == "__main__":
    main()