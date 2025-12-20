# Scripts

## build_frame_cycle.py

Builds AE frame dataset from raw .mat files using vibration cycle-based segmentation. Frames are phase-aligned to detected vibration cycles.

### Usage

```bash
python scripts/build_frame_cycle.py
```

### Arguments

**Input/Output:**
- `--input-dir` (default: `data/raw/original`) - Directory with .mat files
- `--output-dir` (default: `data/raw`) - Output base directory

**Cycle Configuration:**
- `--cycles-per-frame` (int, default: 1) - Number of cycles per frame
- `--cycle-start-phase` (str, default: `positive`) - `positive`, `negative`, or `both`
- `--cycles-length` (int, default: 42373) - Fixed cycle length in samples (≈120 Hz)
- `--skip-cycles` (int, nargs='+', default: [0]) - Cycles to skip before first frame
  - For `both` phase: specify 2 values `[positive, negative]`

**Signal Processing:**
- `--filter-cutoff-hz` (float, default: 1000.0) - Low-pass filter cutoff frequency
- `--filter-order` (int, default: 4) - Butterworth filter order

**Data Selection:**
- `--channels` (str, nargs='+', default: `['A', 'B', 'C', 'D']`) - Channels to extract

### Examples

```bash
# Basic usage
python scripts/build_frame_cycle.py

# Custom configuration
python scripts/build_frame_cycle.py \
  --cycles-per-frame 2 \
  --cycle-start-phase positive \
  --skip-cycles 0

# Both phases
python scripts/build_frame_cycle.py \
  --cycle-start-phase both \
  --skip-cycles 3 5
```

### Output

```
<output-dir>/segmented_cycles_<phase>_c<cycles>_l<length>_c_<channels>_<datetime>/
  data/
    <series>_<load>_<chunk>.npy
  metadata.csv
  dataset_info.json
```

### Notes

- Frames are phase-aligned to cycles with fixed length: `cycles_length * cycles_per_frame`
- Vibrometer signal (channel D) used for detection, not included in output unless specified
