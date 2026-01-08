# Study on dataset Orion AE

A machine learning framework for acoustic emission (AE) signal classification on the **Orion AE dataset**. This project supports both raw time-series frame processing and pre-computed CWT scalogram features for load classification tasks on Orion AE data.

## Features

- **Flexible Dataset System**: Support for raw frame data (`OrionAEFrameDataset`) and pre-computed CWT scalogram features (`CWTScalogramDataset`) from the Orion AE dataset
- **Configurable Preprocessing**: Pipeline-based preprocessing with filters, normalizations, and transforms (e.g., Hanning window, series-aware Z-score normalization) tailored for Orion AE signals
- **Model Registry**: Easy registration and selection of models (EfficientNet, SimpleCNN, custom architectures) for Orion AE classification
- **Comprehensive Training**: Mixed precision training, early stopping, checkpointing, TensorBoard/MLflow logging
- **Flexible Label Mapping**: Configurable class definitions based on Orion AE load values (e.g., binary classification: loose/tight)

## Project Structure

```
orion-ae-study/
├── configs/          # YAML configuration files
│   ├── dataset/      # Dataset configurations (splits, channels, labels)
│   ├── model/        # Model architectures and hyperparameters
│   ├── train/        # Training configurations
│   ├── preprocess/   # Preprocessing pipelines
│   └── feature/      # Feature extraction configurations
├── src/              # Source code
│   ├── core/         # Trainer and evaluator
│   ├── data/         # Datasets and transforms
│   ├── models/       # Model implementations
│   └── utils/        # Utilities (loss, metrics, config)
├── scripts/          # Utility scripts (feature extraction, analysis)
├── data/             # Data directories (raw, processed)
└── runs/             # Training outputs and checkpoints
```

## Data Processing Pipeline

### 1. Build Frame Cycle (`scripts/build_frame_cycle.py`)

Converts raw Orion AE `.mat` files into segmented frame datasets using **vibration cycle-based segmentation**. This script:

- **Loads raw data**: Each `.mat` file contains a 1-second chunk of multi-channel AE signals (channels A, B, C, D) sampled at 5 MHz
- **Detects vibration cycles**: Uses the laser vibrometer signal (**channel D**) to detect vibration cycles via zero-crossing detection
  - **Note**: Channel D (laser vibrometer) is **only used for cycle detection/alignment**, not for classification
  - Preprocesses vibrometer signal with low-pass Butterworth filter (default: 1000 Hz cutoff) to remove high-frequency noise
  - Identifies cycle boundaries (positive-going and/or negative-going zero-crossings)
- **Phase-aligns frames**: Creates frames that start at detected cycle boundaries, ensuring consistent phase relationship across frames
- **Fixed-length frames**: Uses fixed frame length (`cycles_length × cycles_per_frame` samples) for consistent ML model inputs
- **Saves segmented data**: Each `.mat` file is converted to a `.npy` file containing all frames from that chunk, along with metadata
  - **Classification channels**: Only channels A, B, C are used for classification (channel D is excluded from the output frames)

**Usage:**
```bash
python scripts/build_frame_cycle.py \
    --input-dir data/raw/original \
    --output-dir data/raw \
    --cycles-per-frame 1 \
    --cycle-start-phase positive \
    --cycles-length 42373 \
    --channels A B C D  # D is used for cycle detection only
```

**Note**: When specifying channels for classification in dataset configs, use only channels A, B, C (channel D is not included in classification data).

**Output structure:**
```
data/raw/segmented_cycles_<phase>_c<cycles>_l<length>_c_<channels>_<datetime>/
├── data/
│   ├── B_05_001.npy      # All frames from one 1-second chunk
│   ├── B_05_002.npy
│   └── ...
├── metadata.csv           # File metadata (series, load, chunk, num_frames, etc.)
└── dataset_info.json      # Dataset-level information
```

### 2. Feature Extraction (`scripts/build_feature_set.py`)

Extracts features (e.g., CWT scalograms) from the segmented frame dataset for use with image-based models.

**Usage:**
```bash
python scripts/build_feature_set.py \
    --frame-path data/raw/segmented_cycles_... \
    --dataset-config-path configs/dataset/example_1.yaml \
    --feature-config-path configs/feature/feature_cwt_scalogram_gmw.yaml
```

## Scripts

Detailed usage information for all utility scripts:

### `scripts/build_frame_cycle.py`

Converts raw `.mat` files to segmented frame datasets using **vibration cycle-based segmentation** (phase-aligned to cycles).

**Arguments:**
- `--input-dir` (default: `data/raw/original`) - Directory containing raw `.mat` files
- `--output-dir` (default: `data/raw`) - Output base directory
- `--cycles-per-frame` (int, default: 1) - Number of cycles per frame
- `--cycle-start-phase` (str, default: `positive`) - `positive`, `negative`, or `both`
- `--cycles-length` (int, default: 42373) - Fixed cycle length in samples (~120 Hz)
- `--skip-cycles` (int list, default: [0]) - Cycles to skip before first frame
  - For `both` phase: specify 2 values `[positive, negative]`
- `--channels` (str list, default: `['A', 'B', 'C', 'D']`) - Channels to extract (D used for cycle detection only)
- `--filter-cutoff-hz` (float, default: 1000.0) - Low-pass filter cutoff for vibrometer signal
- `--filter-order` (int, default: 4) - Butterworth filter order
- `--no-save-events` - Disable saving cycle event timestamps

**Example:**
```bash
python scripts/build_frame_cycle.py \
    --input-dir data/raw/original \
    --output-dir data/raw \
    --cycles-per-frame 1 \
    --cycle-start-phase positive \
    --cycles-length 42373 \
    --channels A B C D
```

### `scripts/build_frame_sliding_window.py`

Alternative segmentation method using **sliding window** approach (fixed-duration frames with optional overlap).

**Arguments:**
- `--input-dir` (default: `data/raw/original`) - Directory containing raw `.mat` files
- `--output-dir` (default: `data/raw`) - Output base directory
- `--frame-duration-ms` (float, required) - Frame duration in milliseconds
- `--overlap` (float, default: 0.0) - Overlap ratio between frames [0.0, 1.0)
- `--channels` (str list, default: `['A', 'B', 'C', 'D']`) - Channels to extract

**Example:**
```bash
python scripts/build_frame_sliding_window.py \
    --input-dir data/raw/original \
    --output-dir data/raw \
    --frame-duration-ms 100.0 \
    --overlap 0.5 \
    --channels A B C
```

### `scripts/build_feature_set.py`

Extracts features (e.g., CWT scalograms) from segmented frame datasets for image-based models.

**Arguments:**
- `--frame-path` (required) - Path to segmented frame dataset directory
- `--dataset-config-path` (required) - Path to dataset config YAML
- `--feature-config-path` (required) - Path to feature extraction config YAML
- `--preprocess-config-path` (optional) - Path to preprocessing config YAML
- `--save-path` (optional) - Output directory (default: `data/processed`)

**Example:**
```bash
python scripts/build_feature_set.py \
    --frame-path data/raw/segmented_cycles_positive_c1_l42373_c_A_B_C_D_20260107_205752 \
    --dataset-config-path configs/dataset/example_1.yaml \
    --feature-config-path configs/feature/feature_cwt_scalogram_gmw.yaml \
    --preprocess-config-path configs/preprocess/preprocess_serienorm_hanning.yaml
```

**Output:**
```
data/processed/{dataset_name}_features_{timestamp}/
├── data/
│   ├── B_05_001_features.npy
│   └── ...
├── metadata.csv
├── dataset_info.json
└── feature_info.json
```

### `scripts/build_feature_set_optimized.py`

Optimized version of feature extraction with improved performance for large datasets.

**Usage:** Same as `build_feature_set.py` with additional optimizations.

### `scripts/analyze_class_statistics.py`

Calculates statistical information (mean, std, min, max) of raw signals for specified class groups per series. Useful for computing normalization parameters.

**Arguments:**
- `--frame-path` (required) - Path to segmented frame dataset directory
- `--classes` (int list, required) - Space-separated class IDs to analyze (e.g., `0 1 2`)
- `--channels` (str list, optional) - Channels to analyze (default: all available)
- `--output` (optional) - Path to save results as JSON file

**Example:**
```bash
# Analyze classes 0, 1, 2 (baseline classes) for all channels
python scripts/analyze_class_statistics.py \
    --frame-path data/raw/segmented_cycles_positive_c1_l42373_c_A_B_C_D_20260107_205752 \
    --classes 0 1 2

# Analyze specific channels A and B for classes 5 and 6
python scripts/analyze_class_statistics.py \
    --frame-path data/raw/segmented_cycles_positive_c1_l42373_c_A_B_C_D_20260107_205752 \
    --classes 5 6 \
    --channels A B

# Save results to JSON for later use (e.g., for normalization configs)
python scripts/analyze_class_statistics.py \
    --frame-path data/raw/segmented_cycles_positive_c1_l42373_c_A_B_C_D_20260107_205752 \
    --classes 0 1 2 \
    --output baseline_stats.json
```

## Quick Start

### Training (`train.py`)

Trains a model on the Orion AE dataset with configurable preprocessing, model architecture, and training parameters.

**Required Arguments:**
- `--train_config` - Path to training configuration YAML file
- `--dataset_config` - Path to dataset configuration YAML file
- `--model_config` - Path to model configuration YAML file
- `--data_path` - Path to the data directory (segmented frame dataset)

**Optional Arguments:**
- `--preprocess_config` - Path to preprocessing configuration YAML (only for `OrionAEFrameDataset`)
- `--feature_config` - Path to feature configuration YAML (currently unused in training)

**Example:**
```bash
python train.py \
    --train_config configs/train/train_example_cfg.yaml \
    --dataset_config configs/dataset/example_1.yaml \
    --model_config configs/model/simple_cnn.yaml \
    --data_path data/raw/segmented_cycles_positive_c1_l42373_c_A_B_C_D_20260107_205752 \
    --preprocess_config configs/preprocess/preprocess_serienorm_hanning.yaml
```

**Output:**
- Training checkpoints saved to `runs/{experiment_name}/checkpoints/`
- TensorBoard logs (if enabled) in `runs/{experiment_name}/logs/`
- All configs saved to `runs/{experiment_name}/config/` for reproducibility

### Evaluation (`eval.py`)

Evaluates a trained model on test/validation splits with comprehensive metrics and visualizations.

**Required Arguments:**
- `--checkpoint` - Path to model checkpoint file (e.g., `runs/experiment/checkpoints/best_model.pt`)
- `--dataset_config` - Path to dataset configuration YAML file
- `--model_config` - Path to model configuration YAML file
- `--data_path` - Path to the data directory

**Optional Arguments:**
- `--split` (default: `test`) - Dataset split to evaluate: `test`, `val`, or `all`
- `--preprocess_config` - Path to preprocessing configuration YAML (only for `OrionAEFrameDataset`)
- `--output_dir` - Directory to save evaluation results (default: checkpoint directory / eval_results)
- `--device` (default: `auto`) - Device to use: `auto`, `cuda`, or `cpu`
- `--batch_size` (default: 32) - Batch size for evaluation
- `--num_workers` (default: 4) - Number of workers for data loading
- `--train_config` - Path to training config (optional, for loss function config)

**Example:**
```bash
python eval.py \
    --checkpoint runs/experiment_name/checkpoints/best_model.pt \
    --dataset_config configs/dataset/example_1.yaml \
    --model_config configs/model/simple_cnn.yaml \
    --data_path data/raw/segmented_cycles_positive_c1_l42373_c_A_B_C_D_20260107_205752 \
    --split test \
    --preprocess_config configs/preprocess/preprocess_serienorm_hanning.yaml
```

**Output:**
- `metrics.json` - Comprehensive metrics (accuracy, precision, recall, F1, confusion matrix)
- `classification_report.txt` - Detailed per-class metrics
- `predictions.csv` - All predictions with probabilities
- `confusion_matrix.png` - Visual confusion matrix
- `eval_config.yaml` - Evaluation configuration for reproducibility

## Configuration

The framework is fully config-driven via YAML files:

- **Dataset config**: Defines data splits (train/val/test), selected channels (A, B, C - **not D**), and label mappings (load values → classes) for the Orion AE dataset
- **Model config**: Specifies model architecture and hyperparameters
- **Train config**: Training parameters (optimizer, scheduler, loss functions, epochs, etc.)
- **Preprocess config**: Preprocessing pipeline (filters, norms, misc transforms) for Orion AE signals

## Requirements

See `requirements.txt` and `requirements-cuda.txt` for dependencies.
