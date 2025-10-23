import os
import argparse
import numpy as np
import torch

# --- TimeSeAD imports ---
from timesead.data.smd_dataset import SMDDataset
from timesead.data.smap_dataset import SMAPDataset, MSLDataset
from timesead.data.swat_dataset import SWaTDataset
from timesead.data.wadi_dataset import WADIDataset
from timesead.data.tep_dataset import TEPDataset
from timesead.data.exathlon_dataset import ExathlonDataset
from timesead.data.minismd_dataset import MiniSMDDataset


def sliding_windows(x, y, win, stride, min_anom_frac=0.0):
    """
    x: (T, D), y: (T,) in {0,1}
    returns windows_X: (N, win, D), windows_y: (N,)
    """
    T = x.shape[0]
    idxs = range(0, max(T - win + 1, 0), stride)
    Xs, Ys = [], []
    for s in idxs:
        e = s + win
        if e > T: break
        w = x[s:e]
        ly = y[s:e]
        is_anom = (ly.sum() > 0) if min_anom_frac == 0.0 else (float(ly.sum()) / win) >= min_anom_frac
        label = 1 if is_anom else 0
        Xs.append(w)
        Ys.append(label)
    if not Xs:
        return np.empty((0, win, x.shape[1])), np.empty((0,), dtype=int)
    return np.stack(Xs), np.asarray(Ys, dtype=int)

def stack_dataset(ds, win, stride, min_anom_frac):
    X_all, y_all = [], []
    for i in range(len(ds)):
        (xs_tuple, ys_tuple) = ds[i]          # TimeSeAD: inputs tuple, targets tuple
        x = xs_tuple[0].numpy()               # (T, D)
        y = ys_tuple[0].numpy().astype(int)   # (T,), 0 normal / 1 anomaly
        Xw, yw = sliding_windows(x, y, win, stride, min_anom_frac)
        if Xw.size:
            X_all.append(Xw)
            y_all.append(yw)
    if not X_all:
        return np.empty((0, win, ds.num_features)), np.empty((0,), dtype=int)
    return np.concatenate(X_all, axis=0), np.concatenate(y_all, axis=0)

def zscore_fit(X):   # X: (N, T, D)
    mu = X.reshape(-1, X.shape[-1]).mean(axis=0)
    sd = X.reshape(-1, X.shape[-1]).std(axis=0)
    sd[sd == 0] = 1.0
    return mu, sd

def zscore_apply(X, mu, sd):
    return (X - mu) / sd

def build_timesead_dataset(name, training):
    name = name.lower()
    if name == 'smd':
        return SMDDataset(server_id=17, training=training)
    if name == 'minismd':
        return MiniSMDDataset(training=training)
    if name == 'smap':
        return SMAPDataset(training=training)
    if name == 'swat':
        return SWaTDataset(training=training)
    if name == 'wadi':
        return WADIDataset(training=training)
    if name == 'tep':
        return TEPDataset(training=training)  # requires manual download/preprocess per docs
    if name == 'exathlon':
        return ExathlonDataset(training=training)
    raise ValueError(f"Unknown TimeSeAD dataset: {name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--timesead', required=True,
                    help="TimeSeAD dataset key: smd|minismd|smap|swat|wadi|tep|exathlon")
    ap.add_argument('--neutral_name', required=True,
                    help="Folder name under NeuTraL-AD/DATA/ to write .npy files, e.g. timesead_smd")
    ap.add_argument('--win', type=int, default=128)
    ap.add_argument('--stride', type=int, default=64)
    ap.add_argument('--min_anom_frac', type=float, default=0.0,
                    help="Window labeled anomalous if frac of anomaly points >= this")
    ap.add_argument('--standardize', action='store_true', help="Z-score by train windows")
    args = ap.parse_args()

    # Load splits
    ds_train = build_timesead_dataset(args.timesead, training=True)
    ds_test  = build_timesead_dataset(args.timesead, training=False)

    # Build windows
    Xtr, ytr = stack_dataset(ds_train, args.win, args.stride, args.min_anom_frac)
    Xte, yte = stack_dataset(ds_test,  args.win, args.stride, args.min_anom_frac)

    # Train on only normal windows (label==0)
    normal_mask = (ytr == 0)
    Xtr, ytr = Xtr[normal_mask], ytr[normal_mask]
    # labels stay (all zeros) for train; keep 0/1 on test

    # Optional: z-score by training windows
    if args.standardize and Xtr.shape[0] > 0:
        mu, sd = zscore_fit(Xtr)
        Xtr = zscore_apply(Xtr, mu, sd)
        Xte = zscore_apply(Xte, mu, sd)

    # NeuTraL-AD expects (batch, #channels, sequence length) for time series, i.e. (N, D, T)  -> it transposes by itself (load_data() inside LoadData.py)
    #Xtr = Xtr.transpose(0, 2, 1)  # (N, D, T)
    #Xte = Xte.transpose(0, 2, 1)  # (N, D, T)

    # Cast + save
    Xtr = Xtr.astype(np.float32)
    Xte = Xte.astype(np.float32)
    ytr = np.zeros_like(ytr, dtype=np.int64)   # enforce all-normal in train
    yte = yte.astype(np.int64)

    out_dir = os.path.join('DATA', args.neutral_name)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'train_array.npy'), Xtr)
    np.save(os.path.join(out_dir, 'train_label.npy'), ytr)
    np.save(os.path.join(out_dir, 'test_array.npy'),  Xte)
    np.save(os.path.join(out_dir, 'test_label.npy'),  yte)
    print(f"Wrote {out_dir} : train {Xtr.shape}/{ytr.shape}, test {Xte.shape}/{yte.shape}")

if __name__ == "__main__":
    main()
