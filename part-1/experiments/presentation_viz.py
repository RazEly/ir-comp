"""
Presentation visuals: one figure per design decision in strategy.py.

Diagnostic only — uses the leaked pool labels (_load_pool_labels) to avoid
oracle bookkeeping and to compute ceilings. Not submittable.

Run from part-1/:  python experiments/presentation_viz.py

Outputs (experiments/plots/):
  p1_class_imbalance.png       -> why minority handling at all (A, F)
  p2_query_bias.png            -> entropy + GAMMA*P(Left) scoring (C)
  p3_batch_diversity.png       -> over-select + KMeans batch diversity (B)
  p4_learning_curve.png        -> AL loop vs random querying
  p5_fraction_sweep.png        -> CV-tuned training prevalence (F)
  p6_supervision_comparison.png-> headline: budget is not the bottleneck
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from strategy import (
    BATCH_SIZE,
    GAMMA,
    OVERSELECT,
    RATIO_GRID,
    CV_FOLDS,
    _best_fraction,
    _query_scores,
    _resample,
    _select,
    _train,
    _train_final,
)
from utils import (
    ID_COLUMN,
    TARGET_COLUMN,
    _get_test_employee_ids,
    _load_pool_labels,
    evaluate_model,
    load_initial_labeled,
    load_pool,
    prepare_xy,
    train_model,
)

warnings.filterwarnings("ignore", message="Duplicate Employee IDs")

PLOTS = Path(__file__).resolve().parent / "plots"
PLOTS.mkdir(exist_ok=True)
SEEDS = [1, 2, 3]
N_ROUNDS = 10
SEED_COLORS = {1: "#4e79a7", 2: "#f28e2b", 3: "#59a14f"}


def labels_map() -> dict[str, int]:
    return {str(k): int(v) for k, v in _load_pool_labels().items()}


def label_rows(pool: pd.DataFrame, ids: list[str], lmap: dict[str, int]) -> pd.DataFrame:
    rows = pool[pool[ID_COLUMN].astype(str).isin(set(ids))].copy()
    rows[TARGET_COLUMN] = rows[ID_COLUMN].astype(str).map(lmap).astype(int)
    return rows


def run_al(seed: int, strategy: str):
    """Replay the AL loop (oracle bypassed via label map). strategy: 'ours'|'random'.

    Returns (history, labeled) where history = list of (n_labeled, loop-model test F1).
    """
    lmap = labels_map()
    labeled = load_initial_labeled(seed)
    pool = load_pool()
    labeled_ids = set(labeled[ID_COLUMN].astype(str))
    history = []

    for r in range(N_ROUNDS + 1):
        model = _train(labeled, seed)
        history.append((len(labeled), evaluate_model(model, seed)))
        if r == N_ROUNDS:
            break
        unlabeled = pool[~pool[ID_COLUMN].astype(str).isin(labeled_ids)]
        X_pool, _, pool_ids = prepare_xy(unlabeled.assign(**{TARGET_COLUMN: 0}))
        if strategy == "ours":
            scores = _query_scores(model, X_pool)
            top = _select(scores, X_pool.to_numpy(dtype=float), BATCH_SIZE, seed)
            query_ids = pool_ids[top].tolist()
        else:
            rng = np.random.default_rng(seed * 1000 + r)
            query_ids = rng.choice(pool_ids, size=BATCH_SIZE, replace=False).tolist()
        labeled = pd.concat([labeled, label_rows(pool, query_ids, lmap)], ignore_index=True)
        labeled_ids.update(query_ids)

    return history, labeled


def full_pool_df(seed: int) -> pd.DataFrame:
    """All pool rows with labels, excluding this seed's test IDs."""
    lmap = labels_map()
    pool = load_pool()
    test_ids = _get_test_employee_ids(seed)
    ids = pool[ID_COLUMN].astype(str)
    df = pool[ids.isin(set(lmap.keys()) - test_ids)].copy()
    df[TARGET_COLUMN] = df[ID_COLUMN].astype(str).map(lmap).astype(int)
    return df


# ── p1: class imbalance ──────────────────────────────────────────────────────
def p1_class_imbalance():
    lmap = labels_map()
    pool_frac = np.mean(list(lmap.values()))
    init_fracs = [load_initial_labeled(s)[TARGET_COLUMN].mean() for s in SEEDS]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Class imbalance: F1(Left) is the metric, but Left is the minority",
                 fontsize=13, fontweight="bold")

    counts = [1 - pool_frac, pool_frac]
    bars = ax1.bar(["Stayed (0)", "Left (1)"], counts, color=["#4e79a7", "#e15759"],
                   edgecolor="white")
    for b, v in zip(bars, counts):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.1%}",
                 ha="center", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 0.85)
    ax1.set_ylabel("Fraction of pool")
    ax1.set_title(f"Pool prevalence (n={len(lmap):,})")

    x = np.arange(len(SEEDS))
    bars = ax2.bar(x, init_fracs, color=[SEED_COLORS[s] for s in SEEDS], edgecolor="white")
    for b, v in zip(bars, init_fracs):
        ax2.text(b.get_x() + b.get_width() / 2, v - 0.03, f"{v:.1%}", ha="center",
                 fontsize=10, color="white", fontweight="bold")
    ax2.axhline(pool_frac, color="black", ls="--", lw=1.2, label=f"pool = {pool_frac:.1%}")
    ax2.set_ylim(0, 0.42)
    ax2.set_xticks(x, [f"seed {s}" for s in SEEDS])
    ax2.set_ylabel("Left fraction in initial 500")
    ax2.set_title("Initial labeled sets mirror the imbalance")
    ax2.legend(loc="upper right")

    fig.text(0.5, -0.03,
             "RF at the fixed 0.5 threshold under-predicts the 1:2 minority -> every design "
             "decision (oversampling A, positive bias C, prevalence CV F) targets F1(Left).",
             ha="center", fontsize=9, style="italic")
    plt.tight_layout()
    plt.savefig(PLOTS / "p1_class_imbalance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("saved p1_class_imbalance.png")


# ── p2: positive-biased uncertainty (C) ──────────────────────────────────────
def p2_query_bias(model, X_pool):
    p = np.linspace(0.001, 0.999, 400)
    H = entropy(np.vstack([1 - p, p]), base=2)
    score = H + GAMMA * p

    probs = model.predict_proba(X_pool)[:, 1]
    scores = _query_scores(model, X_pool)
    top = _select(scores, X_pool.to_numpy(dtype=float), BATCH_SIZE, seed=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f"Query score (C): entropy + {GAMMA}·P(Left) — uncertainty with a minority bias",
                 fontsize=13, fontweight="bold")

    ax1.plot(p, H, label="entropy (uncertainty)", color="#4e79a7", lw=2)
    ax1.plot(p, GAMMA * p, label=f"{GAMMA}·P(Left) (positive bias)", color="#e15759", lw=2, ls="--")
    ax1.plot(p, score, label="query score (sum)", color="black", lw=2.5)
    pk = p[np.argmax(score)]
    ax1.axvline(0.5, color="#4e79a7", ls=":", lw=1)
    ax1.axvline(pk, color="black", ls=":", lw=1)
    ax1.annotate(f"peak shifts 0.50 → {pk:.2f}", xy=(pk, score.max()),
                 xytext=(0.05, 1.18), fontsize=9,
                 arrowprops=dict(arrowstyle="->", lw=1))
    ax1.set_xlabel("P(Left)")
    ax1.set_ylabel("score")
    ax1.set_title("Analytic score: still boundary-seeking, tilted toward Left")
    ax1.legend(fontsize=9)

    bins = np.linspace(0, 1, 41)
    ax2.hist(probs, bins=bins, density=True, alpha=0.55, color="#9aa5b1",
             label=f"all pool candidates (n={len(probs):,})")
    ax2.hist(probs[top], bins=bins, density=True, alpha=0.65, color="#e15759",
             label=f"selected batch (n={len(top)})")
    ax2.axvline(0.5, color="black", ls=":", lw=1)
    ax2.set_xlabel("P(Left) under the loop model")
    ax2.set_ylabel("density")
    ax2.set_title("Selected batch (seed 1, round 1): boundary + likely-Left region")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOTS / "p2_query_bias.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("saved p2_query_bias.png")


# ── p3: KMeans batch diversity (B) ───────────────────────────────────────────
def p3_batch_diversity(model, X_pool):
    scores = _query_scores(model, X_pool)
    n_cand = BATCH_SIZE * OVERSELECT
    cand = np.argsort(scores)[-n_cand:]
    Xc = StandardScaler().fit_transform(X_pool.to_numpy(dtype=float)[cand])
    Z = PCA(n_components=2, random_state=0).fit_transform(Xc)

    topk_local = np.argsort(scores[cand])[-BATCH_SIZE:]
    km_local = _select(scores, X_pool.to_numpy(dtype=float), BATCH_SIZE, seed=1)
    # map kmeans-chosen global indices into candidate-local positions
    pos = {g: i for i, g in enumerate(cand)}
    km_local = np.array([pos[g] for g in km_local if g in pos])

    # quantify redundancy: how many of the BATCH_SIZE clusters does each batch cover?
    from sklearn.cluster import MiniBatchKMeans

    clusters = MiniBatchKMeans(n_clusters=BATCH_SIZE, random_state=1, n_init=3,
                               batch_size=1024).fit_predict(Xc)
    cov_topk = len(np.unique(clusters[topk_local]))
    cov_km = len(np.unique(clusters[km_local]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, sharey=True)
    fig.suptitle(f"Batch diversity (B): over-select {OVERSELECT}× most uncertain, "
                 "then 1 point per KMeans cluster", fontsize=13, fontweight="bold")

    for ax, sel, cov, name, color in [
        (axes[0], topk_local, cov_topk, f"pure top-{BATCH_SIZE} by score", "#e15759"),
        (axes[1], km_local, cov_km, f"KMeans pick ({BATCH_SIZE} clusters)", "#4e79a7"),
    ]:
        ax.scatter(Z[:, 0], Z[:, 1], s=8, c="#d0d4da",
                   label=f"candidate pool ({n_cand} most uncertain)")
        ax.scatter(Z[sel, 0], Z[sel, 1], s=14, c=color, alpha=0.8, label=name)
        ax.set_title(name)
        ax.set_xlabel("PCA 1")
        ax.legend(fontsize=9, loc="upper right")
        ax.text(0.02, 0.02, f"feature-space clusters covered: {cov}/{BATCH_SIZE}",
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor=color))
    axes[0].set_ylabel("PCA 2")

    fig.text(0.5, -0.02,
             "Pure top-k concentrates the batch in a few high-score regions; KMeans keeps the "
             "highest-score point per cluster, covering the whole uncertain frontier "
             "(ablation: dropping diversity costs ~0.008 F1).",
             ha="center", fontsize=9, style="italic")
    plt.tight_layout()
    plt.savefig(PLOTS / "p3_batch_diversity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("saved p3_batch_diversity.png")


# ── p4: learning curve, ours vs random ───────────────────────────────────────
def p4_learning_curve(histories, final_f1):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.suptitle("Active learning loop: loop-model F1(Left) per round vs random querying",
                 fontsize=13, fontweight="bold")

    for strat, ls, lw_m in [("ours", "-", 2.5), ("random", "--", 2.0)]:
        curves = np.array([[f for _, f in histories[(s, strat)]] for s in SEEDS])
        n = [n for n, _ in histories[(SEEDS[0], strat)]]
        for i, s in enumerate(SEEDS):
            ax.plot(n, curves[i], ls=ls, color=SEED_COLORS[s], alpha=0.30, lw=1)
        ax.plot(n, curves.mean(axis=0), ls=ls, color="black", lw=lw_m,
                label=f"{'biased-uncertainty + diversity (ours)' if strat == 'ours' else 'random queries'} — mean")

    mean_final = np.mean(list(final_f1.values()))
    ax.scatter([n[-1]] * len(SEEDS), [final_f1[s] for s in SEEDS],
               marker="*", s=140, c=[SEED_COLORS[s] for s in SEEDS], zorder=5,
               label="final model (CV-tuned prevalence, F)")
    ax.axhline(mean_final, color="#e15759", ls=":", lw=1.2)
    ax.annotate(f"final mean = {mean_final:.3f}", xy=(n[1], mean_final),
                xytext=(n[1], mean_final + 0.004), fontsize=9, color="#e15759")

    ax.set_xlabel("labeled training rows (500 initial + 500/round, budget 5,000)")
    ax.set_ylabel("test F1(Left)")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(PLOTS / "p4_learning_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("saved p4_learning_curve.png")


# ── p5: training prevalence sweep (F) ────────────────────────────────────────
def p5_fraction_sweep(labeled_sets):
    fracs = list(RATIO_GRID) + [0.9]
    xlabels = ["natural" if f is None else f"{f:.1f}" for f in fracs]
    xs = np.arange(len(fracs))

    test_f1 = {s: [] for s in SEEDS}
    oof_f1 = {s: [] for s in SEEDS}
    chosen = {}
    for s in SEEDS:
        X, y, ids = prepare_xy(labeled_sets[s])
        X = X.astype(float)
        chosen[s] = _best_fraction(X, y, ids, s)
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=s)
        for f in fracs:
            Xr, yr, ir = _resample(X, y, ids, f, s)
            test_f1[s].append(evaluate_model(train_model(Xr, yr, ir, s), s))
            folds = []
            for tr, va in skf.split(X, y):
                Xt, yt, it = _resample(X.iloc[tr], y[tr], ids[tr], f, s)
                m = train_model(Xt, yt, it, s)
                folds.append(f1_score(y[va], m.predict(X.iloc[va]), pos_label=1))
            oof_f1[s].append(np.mean(folds))
        print(f"  seed {s}: CV picks frac={chosen[s]}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.suptitle("Training prevalence (F): duplicate Left rows until fraction f, pick f by CV",
                 fontsize=13, fontweight="bold")

    for s in SEEDS:
        ax.plot(xs, test_f1[s], color=SEED_COLORS[s], alpha=0.35, lw=1, marker="o", ms=3)
        ax.plot(xs, oof_f1[s], color=SEED_COLORS[s], alpha=0.20, lw=1, ls="--", marker="s", ms=3)
    ax.plot(xs, np.mean([test_f1[s] for s in SEEDS], axis=0), color="black", lw=2.5,
            marker="o", label="test F1(Left) — mean over seeds")
    ax.plot(xs, np.mean([oof_f1[s] for s in SEEDS], axis=0), color="black", lw=2, ls="--",
            marker="s", label="CV out-of-fold F1 — what the strategy sees")
    for s in SEEDS:
        ci = fracs.index(chosen[s])
        ax.scatter([ci], [test_f1[s][ci]], marker="*", s=180, color=SEED_COLORS[s],
                   zorder=5, label=f"CV choice seed {s} (f={xlabels[ci]})")

    ax.set_xticks(xs, xlabels)
    ax.set_xlabel("Left fraction in the final training matrix")
    ax.set_ylabel("F1(Left)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    fig.text(0.5, -0.02,
             "Resampling to ~0.6-0.8 gains +0.02-0.03 over natural prevalence at the frozen 0.5 "
             "threshold — a legal, leakage-free proxy for threshold tuning. CV is optimistic "
             "(AL-enriched folds) but agrees on the region.",
             ha="center", fontsize=9, style="italic")
    plt.tight_layout()
    plt.savefig(PLOTS / "p5_fraction_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("saved p5_fraction_sweep.png")


# ── p6: headline supervision comparison ──────────────────────────────────────
def p6_supervision_comparison(final_f1):
    res = {"full pool\nnatural balance\n(~14.4k labels)": [],
           "full pool\n+ duplication 0.7\n(~14.4k labels)": [],
           "ours: AL 5k budget\n+ CV prevalence\n(5.5k labels)": [final_f1[s] for s in SEEDS],
           "class_weight 1:4\n(forbidden —\nmodel ceiling)": []}
    for s in SEEDS:
        df = full_pool_df(s)
        X, y, ids = prepare_xy(df)
        X = X.astype(float)
        res["full pool\nnatural balance\n(~14.4k labels)"].append(
            evaluate_model(train_model(X, y, ids, s), s))
        Xr, yr, ir = _resample(X, y, ids, 0.7, s)
        res["full pool\n+ duplication 0.7\n(~14.4k labels)"].append(
            evaluate_model(train_model(Xr, yr, ir, s), s))
        cw = RandomForestClassifier(n_estimators=100, random_state=s, n_jobs=-1,
                                    class_weight={0: 1, 1: 4}).fit(X, y)
        res["class_weight 1:4\n(forbidden —\nmodel ceiling)"].append(evaluate_model(cw, s))
        print(f"  seed {s}: comparison bars done")

    names = list(res.keys())
    means = [np.mean(res[k]) for k in names]
    colors = ["#9aa5b1", "#4e79a7", "#e15759", "#d0d4da"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.suptitle("Budget is not the bottleneck: 5k tuned labels ≈ 14.4k labels; "
                 "the frozen RF is the ceiling", fontsize=13, fontweight="bold")
    bars = ax.bar(names, means, color=colors, edgecolor="white",
                  hatch=["", "", "", "//"])
    for b, k in zip(bars, names):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.006,
                f"{np.mean(res[k]):.3f}", ha="center", fontsize=11, fontweight="bold")
        for s, v in zip(SEEDS, res[k]):
            ax.scatter(b.get_x() + b.get_width() / 2, v, s=25, color=SEED_COLORS[s], zorder=5)
    for s in SEEDS:
        ax.scatter([], [], s=25, color=SEED_COLORS[s], label=f"seed {s}")
    ax.axhline(0.55, color="black", ls=":", lw=1.2)
    ax.text(0.02, 0.552, "0.55 guaranteed-points floor", fontsize=8,
            transform=ax.get_yaxis_transform())
    ax.set_ylabel("test F1(Left) @ fixed 0.5 threshold")
    ax.set_ylim(0.5, max(means) + 0.05)
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    plt.savefig(PLOTS / "p6_supervision_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("saved p6_supervision_comparison.png")


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    only = set(sys.argv[1:])  # e.g. `python experiments/presentation_viz.py p1 p3`

    def want(tag):
        return not only or tag in only

    if want("p1"):
        print("p1: class imbalance ...")
        p1_class_imbalance()

    if want("p2") or want("p3"):
        # shared round-1 state (seed 1) for p2 & p3
        print("p2/p3: round-1 model state (seed 1) ...")
        init = load_initial_labeled(1)
        model0 = _train(init, 1)
        pool = load_pool()
        unl = pool[~pool[ID_COLUMN].astype(str).isin(set(init[ID_COLUMN].astype(str)))]
        X_pool, _, _ = prepare_xy(unl.assign(**{TARGET_COLUMN: 0}))
        if want("p2"):
            p2_query_bias(model0, X_pool)
        if want("p3"):
            p3_batch_diversity(model0, X_pool)

    if not (want("p4") or want("p5") or want("p6")):
        print("done (subset)")
        sys.exit(0)

    print("p4: AL replays (ours + random, 3 seeds) ...")
    histories, labeled_sets, final_f1 = {}, {}, {}
    for s in SEEDS:
        for strat in ("ours", "random"):
            histories[(s, strat)], lab = run_al(s, strat)
            if strat == "ours":
                labeled_sets[s] = lab
        final_f1[s] = evaluate_model(_train_final(labeled_sets[s], s), s)
        print(f"  seed {s}: final F1 = {final_f1[s]:.4f}")
    p4_learning_curve(histories, final_f1)

    print("p5: prevalence sweep ...")
    p5_fraction_sweep(labeled_sets)

    print("p6: supervision comparison ...")
    p6_supervision_comparison(final_f1)

    print(f"\nAll figures in {PLOTS}")
