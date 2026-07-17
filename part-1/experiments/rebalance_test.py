"""
Test whether the '0.574 ceiling' is real.
Hypothesis: full-pool training is NOT F1-optimal under imbalance.
Rebalancing the training set (oversample minority / balanced subset) should beat it.
Run from part-1/: python experiments/rebalance_test.py
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
    load_test,
    prepare_xy,
    train_model,
)

SEEDS = [1, 2, 3]


def full_labeled_df(exclude_ids):
    pool = load_pool()
    labels_map = _load_pool_labels()
    ids = pool[ID_COLUMN].astype(str)
    mask = ids.isin(set(labels_map.keys()) - exclude_ids)
    df = pool[mask].copy()
    df[TARGET_COLUMN] = df[ID_COLUMN].astype(str).map(labels_map).astype(int)
    return df


for seed in SEEDS:
    test_ids = _get_test_employee_ids(seed)
    df = full_labeled_df(test_ids)

    # test-set balance
    test_df = load_test(seed)
    _, y_test, _ = prepare_xy(test_df)

    # (1) full imbalanced training = my old "ceiling"
    X, y, ids = prepare_xy(df)
    f1_full = evaluate_model(train_model(X, y, ids, seed), seed)

    # (2) class-balanced subset (undersample majority to minority count)
    left = df[df[TARGET_COLUMN] == 1]
    stay = df[df[TARGET_COLUMN] == 0]
    n = min(len(left), len(stay))
    bal_df = pd.concat([left.sample(n, random_state=seed), stay.sample(n, random_state=seed)])
    Xb, yb, idb = prepare_xy(bal_df)
    f1_bal = evaluate_model(train_model(Xb, yb, idb, seed), seed)

    # (3) oversample minority via duplication up to majority count
    reps = len(stay) // len(left)
    over_df = pd.concat([df] + [left] * (reps - 1), ignore_index=True)
    Xo, yo, ido = prepare_xy(over_df)
    f1_over = evaluate_model(train_model(Xo, yo, ido, seed), seed)

    print(
        f"Seed {seed}: test_Left={y_test.mean():.3f} | "
        f"full={f1_full:.4f}  balanced={f1_bal:.4f}  oversample={f1_over:.4f}"
    )
