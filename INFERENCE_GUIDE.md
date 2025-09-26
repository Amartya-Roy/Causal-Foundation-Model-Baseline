# Inference quickstart

The `src/inference.sh` helper script wraps `src/inference.py` and lets you run
SEA on either a dataset manifest CSV (with columns named `fp_data`, `fp_graph`,
`fp_regime`, and `split`) or a plain matrix-style CSV containing your
observations. Plain CSVs are automatically converted into the manifest format
and stored under your run's `SAVE_PATH`.

## 1. Prepare your dataset CSV

### Option A: Manifest CSV (advanced)

1. Create a CSV file (for example `data.csv`) with the following columns:
   - `fp_data`: absolute path to the `.npy` data matrix to evaluate.
   - `fp_graph`: absolute path to the `.npy` ground-truth adjacency matrix.
   - `fp_regime` (GIES/interventional only): absolute path to the intervention
     metadata `.csv` file. Use an empty string for observational datasets.
   - `split`: one of `train`, `val`, or `test`. At inference time we read the
     `test` split.
2. Ensure every referenced file is accessible from the machine that will run
   the script.

### Option B: Raw observational CSV (quick start)

1. Export your measurements to a CSV file where each column corresponds to a
   variable and each row is an observation. Index columns produced by tools
   like pandas (named `index`, `Unnamed: 0`, or empty) are ignored
   automatically.
2. Point `DATA_FILE` at this CSV. During argument processing the script will
   generate the required `.npy` tensors, dummy graph/regime files, and a
   manifest under `${SAVE_PATH}/converted_csv/<csv-name>/` before launching
   inference.

## 2. Run inference

Use the environment variables below to customise the run without editing the
script. Any value that is omitted falls back to the defaults shown.

```bash
# Activate your conda/virtualenv before running the command
export CUDA=0                                 # GPU id (-1 for CPU)
export TAG="aggregator_tf_gies"              # Chooses the config/checkpoint
export DATA_FILE="/abs/path/to/data.csv"     # Dataset description CSV
export SAVE_PATH="/abs/path/to/output_dir"   # Where to place args/results files
export RESULTS_FILE="predictions.npy"        # Output file name (inside SAVE_PATH)
export CHECKPOINT_PATH="/abs/path/to.ckpt"   # Optional: override pretrained ckpt

./src/inference.sh
```

The script automatically:
- picks the matching default checkpoint for the chosen `TAG` (unless you set
  `CHECKPOINT_PATH`),
- writes the aggregated predictions to
  `${SAVE_PATH}/${RESULTS_FILE}` as a NumPy `.npy` file, and
- stores the resolved CLI/config arguments alongside the results for
  reproducibility.

## 3. Reading the results

The saved `.npy` file contains the Python dictionary returned by the inference
loop. Load it with NumPy and access the metrics/predictions per dataset key:

```python
import numpy as np

results = np.load("/abs/path/to/output_dir/predictions.npy", allow_pickle=True).item()
print(results["<dataset-key>"]["pred"][0])  # Example: first predicted graph
```
