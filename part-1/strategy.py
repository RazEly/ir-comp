from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from utils import (
    call_oracle,
    get_oracle_usage,
    load_initial_labeled,
    load_pool,
    prepare_xy,
    train_model,
)

BATCH_SIZE = 500

MINORITY_RATIO = 1.0
GAMMA = 0.5
OVERSELECT = 3

RATIO_GRID = (None, 0.5, 0.6, 0.7, 0.8)
FIXED_FRACTION = 0.5
CV_FOLDS = 3


def _oversample(labeled: pd.DataFrame, ratio: float, seed: int) -> pd.DataFrame:
    """(A) Duplicate minority (Left) rows to rebalance training set.

    RandomForest hyperparameters are fixed (no class_weight), and the staff
    score uses model.predict at the 0.5 threshold. Oversampling the minority
    shifts the decision boundary toward higher Left recall -> better F1(Left).
    Duplicate Employee IDs are allowed (train_model only warns).
    """
    pos = labeled[labeled["Attrition"] == 1]
    neg = labeled[labeled["Attrition"] == 0]
    if len(pos) == 0 or len(neg) == 0:
        return labeled
    target = int(len(neg) * ratio)
    if target <= len(pos):
        return labeled
    extra = pos.sample(n=target - len(pos), replace=True, random_state=seed)
    return pd.concat([labeled, extra], ignore_index=True)


def _query_scores(model, X_pool: pd.DataFrame) -> np.ndarray:
    """Positive-biased uncertainty: predictive entropy + GAMMA * P(Left).

    (C) The P(Left) bias steers budget toward likely-positive rows -> more true
    Lefts labeled, enriching the minority signal the RF needs for F1(Left).
    """
    probs = model.predict_proba(X_pool)  # [n, 2], column 1 = P(Left)
    unc = entropy(probs.T, base=2)  # per-row entropy, peaks at p=0.5
    return unc + GAMMA * probs[:, 1]


def _kmeans_select(
    scores: np.ndarray, X: np.ndarray, n_query: int, seed: int
) -> np.ndarray:
    """(B) Raw-feature KMeans diversity: cluster candidates, keep top-score each."""
    Xc = StandardScaler().fit_transform(X)
    labels = MiniBatchKMeans(
        n_clusters=n_query, random_state=seed, n_init=3, batch_size=1024
    ).fit_predict(Xc)
    chosen: list[int] = []
    for c in range(n_query):
        members = np.where(labels == c)[0]
        if len(members):
            chosen.append(int(members[np.argmax(scores[members])]))
    if len(chosen) < n_query:  # backfill empty clusters by score
        taken = set(chosen)
        for idx in np.argsort(scores)[::-1]:
            idx = int(idx)
            if idx not in taken:
                chosen.append(idx)
                taken.add(idx)
                if len(chosen) == n_query:
                    break
    return np.array(chosen[:n_query])


def _select(scores: np.ndarray, X: np.ndarray, n_query: int, seed: int) -> np.ndarray:
    """Over-select the most uncertain candidates, then diversify the batch."""
    n_cand = min(len(scores), n_query * OVERSELECT)
    cand = np.argsort(scores)[-n_cand:]
    if len(cand) <= n_query:
        return cand
    sub = _kmeans_select(scores[cand], X[cand], n_query, seed)
    return cand[sub]


def _train(labeled: pd.DataFrame, seed: int):
    """Loop model used for query scoring: plain duplicate oversampling at ratio 1.0."""
    balanced = _oversample(labeled, MINORITY_RATIO, seed)
    X_train, y_train, train_ids = prepare_xy(balanced)
    return train_model(X_train, y_train, train_ids, seed)


def _resample(
    X: pd.DataFrame, y: np.ndarray, ids: np.ndarray, frac: float | None, seed: int
):
    """Duplicate minority rows until Left reaches positive fraction `frac` (never drops data)."""
    if frac is None:
        return X, y, ids
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return X, y, ids
    target_pos = int(frac * n_neg / (1.0 - frac))
    n_add = target_pos - n_pos
    if n_add <= 0:
        return X, y, ids

    pos_idx = np.where(y == 1)[0]
    take = np.random.default_rng(seed).choice(pos_idx, size=n_add, replace=True)
    X_out = pd.concat([X, X.iloc[take].reset_index(drop=True)], ignore_index=True)
    y_out = np.concatenate([y, np.ones(n_add, dtype=int)])
    ids_out = np.concatenate([ids, np.array([f"syn_{i}" for i in range(n_add)])])
    return X_out, y_out, ids_out


def _best_fraction(X: pd.DataFrame, y: np.ndarray, ids: np.ndarray, seed: int):
    """(F) Choose the positive fraction maximizing out-of-fold F1(Left) — no test leakage."""
    if int((y == 1).sum()) < CV_FOLDS or int((y == 0).sum()) < CV_FOLDS:
        return FIXED_FRACTION
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    best_f, best_s = FIXED_FRACTION, -1.0
    for f in RATIO_GRID:
        fold_scores = []
        for tr, va in skf.split(X, y):
            Xtr, ytr, itr = _resample(X.iloc[tr], y[tr], ids[tr], f, seed)
            m = train_model(Xtr, ytr, itr, seed)
            fold_scores.append(f1_score(y[va], m.predict(X.iloc[va]), pos_label=1))
        s = float(np.mean(fold_scores))
        if s > best_s:
            best_s, best_f = s, f
    return best_f


def _train_final(labeled: pd.DataFrame, seed: int):
    """Final returned model: resample to the CV-tuned positive fraction, then fit."""
    X, y, ids = prepare_xy(labeled)
    X = X.astype(float)
    frac = _best_fraction(X, y, ids, seed)
    X, y, ids = _resample(X, y, ids, frac, seed)
    return train_model(X, y, ids, seed)


def run_active_learning(seed: int):
    """
    Run active learning for the given seed and return a trained RandomForestClassifier.

    Parameters
    ----------
    seed : int
        One of {1, 2, 3}. Controls randomness and selects the initial labeled set.

    Returns
    -------
    sklearn.ensemble.RandomForestClassifier
        Trained model to be evaluated on the hidden test set.
    """
    labeled = load_initial_labeled(seed)
    pool = load_pool()
    labeled_ids = set(labeled["Employee ID"].astype(str))
    budget_left = get_oracle_usage()["remaining"]  # oracle tracker is authoritative

    while budget_left > 0:
        model = _train(labeled, seed)

        unlabeled = pool[~pool["Employee ID"].astype(str).isin(labeled_ids)]
        if unlabeled.empty:
            break

        # prepare_xy needs Attrition column; add dummy since _encode_features drops it
        X_pool, _, pool_ids = prepare_xy(unlabeled.assign(Attrition=0))

        scores = _query_scores(model, X_pool)
        n_query = min(BATCH_SIZE, budget_left, len(unlabeled))
        top_idx = _select(scores, X_pool.to_numpy(dtype=float), n_query, seed)
        query_ids = pool_ids[top_idx].tolist()

        new_labeled = call_oracle(query_ids)
        labeled = pd.concat([labeled, new_labeled], ignore_index=True)
        labeled_ids.update(query_ids)
        budget_left = get_oracle_usage()["remaining"]

    return _train_final(labeled, seed)
