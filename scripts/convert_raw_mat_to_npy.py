from pathlib import Path
import logging
import argparse

import scipy
import pandas as pd
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_measurement_files(input_dir: Path) -> list[Path]:
    """
    Get all measurement files from the source directory.

    Args:
        input_dir: Path to the source directory containing the raw .mat files

    Returns:
        list[Path]: List of paths to the measurement files
    """
    return list(input_dir.glob("measurementSeries_*/*/*.mat"))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=Path, default=PROJECT_ROOT / 'data' / 'raw' / 'original')
    parser.add_argument('--output_dir', type=Path, default=PROJECT_ROOT / 'data' / 'raw' / 'converted')
    return parser.parse_args()

def convert_dataset(source_root: Path, target_root: Path):
    """
    Convert raw .mat files to .npy files and save metadata to CSV.

    Args:
        source_root: Path to the source directory containing the raw .mat files
        target_root: Path to the target directory to save the converted .npy files and metadata

    Returns:
        None
    """

    # Mapping load string -> class & value
    load_map = {
        "05cNm": (0, 0.05), "10cNm": (1, 0.10), "20cNm": (2, 0.20),
        "30cNm": (3, 0.30), "40cNm": (4, 0.40), "50cNm": (5, 0.50), "60cNm": (6, 0.60)
    }

    metadata = []
    data_files = get_measurement_files(source_root)
    logger.info(f"Found {len(data_files)} data files")

    for mat_path in tqdm(data_files, desc="Converting dataset"):
        # 1. Parse from path: measurementSeries_B / 05cNm / filename.mat
        series_name = mat_path.parent.parent.name.split('_')[-1]  
        load_str = mat_path.parent.name       
        logger.info(f"Processing {mat_path.name} with series name {series_name} and load string {load_str}") 

        # 2. Load .mat
        try:
            mat = scipy.io.loadmat(mat_path)
            # Stack 4 channels: (N, 4)
            meas_data = np.column_stack([
                mat['A'].reshape(-1),
                mat['B'].reshape(-1),
                mat['C'].reshape(-1),
                mat['D'].reshape(-1)
            ]).astype(np.float32)

            # Validation: check all channels have same length
            # assert all(mat[ch].shape[0] == mat['A'].shape[0] for ch in ['B', 'C', 'D']), \
            #     f"Channel mismatch in {mat_path}"

            # 3. Create unique ID
            # Format: {Series}_{LoadClass}_{SequenceNumber}
            load_class, load_val = load_map[load_str]
            seq_num = len([m for m in metadata if m['series'] == series_name and m['load_class'] == load_class]) + 1
            file_id = f"{series_name}_{load_str.replace('cNm', '')}_{seq_num:03d}"

            # 4. Save as .npy
            npy_path = target_root / f"{file_id}.npy"
            np.save(npy_path, meas_data)

            # 5. Extract timestamp from filename
            # Format: salves_out_XXcNm_Y_Fs5MHz_Tf1s_YYYY-MM-DD-HH-MM-SS_1.mat
            filename_parts = mat_path.stem.split('_')
            timestamp_str = '_'.join(filename_parts[-4:-1])  # YYYY-MM-DD-HH-MM-SS
            timestamp = timestamp_str.replace('-', ':', 2)  # YYYY-MM-DD HH-MM-SS -> YYYY-MM-DD HH:MM:SS

            # 6. Append metadata row
            metadata.append({
                "file_id": file_id,
                "file_path": str(npy_path),
                "series": series_name,
                "load_class": load_class,
                "load_val": load_val,
                "num_samples": len(meas_data),
                "num_channels": 4,
                "channel_names": "A,B,C,D",  # New column
                "timestamp": timestamp,
                "original_file": mat_path.name
            })

        except Exception as e:
            logger.error(f"Error loading {mat_path}: {e}")
            continue

    # 7. Save metadata to CSV
    df = pd.DataFrame(metadata)
    csv_path = target_root / "metadata.csv"
    df.to_csv(csv_path, index=False)    

    logger.info("✓ Conversion complete!")
    logger.info(f"  Samples: {len(df)}")
    logger.info(f"  Series: {df['series'].unique().tolist()}")
    logger.info(f"  Loads: {df['load_class'].unique().tolist()}")
    logger.info(f"  Metadata saved to: {csv_path}")


def main():
    args = parse_args()
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    input_dir = args.input_dir
    output_dir = args.output_dir

    convert_dataset(input_dir, output_dir)


if __name__ == "__main__":
    main()