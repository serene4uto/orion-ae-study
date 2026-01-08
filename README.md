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

## Quick Start

### Training

```bash
python train.py \
    --train_config configs/train/train_example_cfg.yaml \
    --dataset_config configs/dataset/example_1.yaml \
    --model_config configs/model/simple_cnn.yaml \
    --data_path data/raw/segmented_cycles_... \
    --preprocess_config configs/preprocess/preprocess_example.yaml
```

### Evaluation

```bash
python eval.py \
    --checkpoint runs/experiment_name/checkpoints/best_model.pt \
    --dataset_config configs/dataset/example_1.yaml \
    --model_config configs/model/simple_cnn.yaml \
    --data_path data/raw/segmented_cycles_... \
    --split test
```

## Configuration

The framework is fully config-driven via YAML files:

- **Dataset config**: Defines data splits (train/val/test), selected channels (A, B, C - **not D**), and label mappings (load values → classes) for the Orion AE dataset
- **Model config**: Specifies model architecture and hyperparameters
- **Train config**: Training parameters (optimizer, scheduler, loss functions, epochs, etc.)
- **Preprocess config**: Preprocessing pipeline (filters, norms, misc transforms) for Orion AE signals

## Requirements

See `requirements.txt` and `requirements-cuda.txt` for dependencies.
