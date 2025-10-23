# NeuTraL-AD-on-TimeSeAD-data

The process consists of the following steps:
1) transform a **TimeSeAD** dataset (e.g., **SMD**) into the 4-NumPy-file format expected by **NeuTraL-AD**,  
2) create a minimal **config**, **register** the dataset name, and  
3) **run** the NeuTraL-AD experiment and locate the results.
 
> NeuTraL-AD expects time series data as **sequence-level** samples (windows). We convert point-wise labels to **window labels**.

---

## 1) Transform a TimeSeAD dataset into NeuTraL-AD format

### Mandatory command (SMD example)

```bash
python timesead_to_neutralad.py --timesead smd --neutral_name timesead_smd_w128_s64 --win 128 --stride 64
```

This creates:
```
NeuTraL-AD/
└─ DATA/
   └─ timesead_smd_w128_s64/
      ├─ train_array.npy   # (N_train, T, D)  float32
      ├─ train_label.npy   # (N_train,)       int64 — all zeros (normal only)
      ├─ test_array.npy    # (N_test,  T, D)  float32
      └─ test_label.npy    # (N_test,)        int64 — 0/1 window labels
```

#### What the converter does

- Loads the TimeSeAD dataset split (`training=True/False`).
- **Slides fixed-length windows** over each time series:
  - `--win` controls the **window length** (T).
  - `--stride` controls **overlap** (smaller stride → more windows).
- **Window labels**: A window is anomalous if it contains anomalous points (configurable via `--min_anom_frac`; see below).
- For **training**, only **normal** windows (label 0) are kept (one-class setup).
- Arrays are saved as **(N, T, D)**. NeuTraL-AD’s time-series loader will transpose internally to **(N, D, T)**, so you don’t need to.

### All converter arguments (reference)

| Argument | Required | Default | Notes |
|---|---:|---:|---|
| `--timesead {smd,minismd,smap,swat,wadi,tep,exathlon}` | ✅ | – | Which TimeSeAD dataset wrapper to use. |
| `--neutral_name <str>` | ✅ | – | Output subfolder name under `DATA/`. You’ll pass this again as `--dataset-name` when running. |
| `--win <int>` |  | `128` | Window length in timesteps. Must match `x_length` in your config. |
| `--stride <int>` |  | `64` | Step between window starts. Smaller = more windows. |
| `--min_anom_frac <float>` |  | `0.0` | Window labeled anomalous if **fraction of anomalous points** ≥ this value. With `0.0`, **any** anomaly point marks the window anomalous. |
| `--standardize` |  | off | If set, z-score features (per sensor) using **training windows only**, then apply to test. |

---

## 2) Create a minimal config file

1) **Copy** a time-series config (e.g., `config_files/config_arabic.yml`) to  
   `config_files/config_smd.yml`

2) **Edit only what you need**:
   - Set **`x_length: 128`** (must match the `--win` used during export).
   - Set **`result_folder:`** to where you want outputs, e.g.  
     `RESULTS/`
   - (Optionally) tune training hparams (`batch_size`, `training_epochs`, `learning_rate`, `num_trans`, `trans_type`, `optimizer`, `early_stopper`, …).

---

## 3) Register the dataset name

The config parser asserts that `--dataset-name` is known. Add **two small entries**.

### 3.1 `config/Dataset_Class.py`
```python
# Example entry for exported SMD windows
class timesead_smd_w128_s64:
    data_name = "timesead_smd_w128_s64"
    num_cls = 1  # training labels contain a single class (normal-only => label 0)
```

- `num_cls = 1` ensures the training/eval loop iterates exactly once (class index 0).

### 3.2 `config/base.py`
Import the class and register it in the dictionary that the parser uses:
```python
class Config:

    datasets = {
        ...
        "timesead_smd_w128_s64": timesead_smd_w128_s64,
    }
```

---

## 4) Run the NeuTraL-AD experiment

Exactly as requested:
```bash
python Launch_Exps.py --config-file config_smd.yml --dataset-name timesead_smd_w128_s64
```

- `--config-file`: path to your YAML (e.g., under `config_files/`).
- `--dataset-name`: **must equal** the folder name under `DATA/` created in step 1.

---

## 5) Results

- NeuTraL-AD writes outputs to the path you set in **`result_folder`** inside your YAML (e.g., `RESULTS/TimeSeAD_SMD_w128_s64/`)
- Contains:
  - **Training log** with per-epoch metrics (TR/VAL loss, validation AUC/AP/F1).
  - **Final test metrics** (AUC/AP/F1) summary.

---
