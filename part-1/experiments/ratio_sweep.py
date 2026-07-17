"""
F1 is computed only on Left (pos_label=1). So we can trade Stayed precision
for Left recall. Sweep training class ratio to find F1(Left)-optimal composition.
Run from part-1/: python experiments/ratio_sweep.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from utils import (
    ID_COLUMN,
    TARGET_COLUMN,
    _get_test_employee_ids,
    _load_pool_labels,
    evaluate_model,
    load_pool,
    prepare_xy,
    train_model,
)

SEEDS = [1, 2, 3]
# fraction of Left in the training set
LEFT_FRACS = [0.42, 0.46, 0.50, 0.54, 0.58]


def full_labeled_df(exclude_ids):
    pool = load_pool()
    labels_map = _load_pool_labels()
    ids = pool[ID_COLUMN].astype(str)
    mask = ids.isin(set(labels_map.keys()) - exclude_ids)
    df = pool[mask].copy()
    df[TARGET_COLUMN] = df[ID_COLUMN].astype(str).map(labels_map).astype(int)
    return df


results = {f: [] for f in LEFT_FRACS}
for seed in SEEDS:
    test_ids = _get_test_employee_ids(seed)
    df = full_labeled_df(test_ids)
    left = df[df[TARGET_COLUMN] == 1]
    stay = df[df[TARGET_COLUMN] == 0]

    for frac in LEFT_FRACS:
        # keep all Left, subsample Stayed to hit target Left fraction
        n_left = len(left)
        n_stay = int(round(n_left * (1 - frac) / frac))
        n_stay = min(n_stay, len(stay))
        comp = pd.concat([left, stay.sample(n_stay, random_state=seed)])
        X, y, ids = prepare_xy(comp)
        f1 = evaluate_model(train_model(X, y, ids, seed), seed)
        results[frac].append(f1)

print("Left_frac | seed1  seed2  seed3  | mean")
for frac in LEFT_FRACS:
    s = results[frac]
    print(f"  {frac:.2f}    | {s[0]:.4f} {s[1]:.4f} {s[2]:.4f} | {np.mean(s):.4f}")
