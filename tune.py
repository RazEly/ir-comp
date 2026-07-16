import json
import os

import optuna

import retrieval

MU, FB_DOCS, FB_TERMS, ORIG_WEIGHT = 1000, 10, 50, 0.5

# search spaces
K1_RANGE, K1_STEP = (0.4, 2.0), 0.1
B_RANGE, B_STEP = (0.1, 1.0), 0.05
RRF_K_RANGE = (10, 200)
BM25_TRIALS, RRF_TRIALS = 30, 20

optuna.logging.set_verbosity(optuna.logging.WARNING)


def parse_qrels(path):
    qrels = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _, docid, rel = parts[:4]
            if int(rel) > 0:
                qrels.setdefault(qid, set()).add(docid)
    return qrels


def average_precision(ranked, relevant):
    hits, total = 0, 0.0
    for i, (docid, _) in enumerate(ranked, start=1):
        if docid in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant) if relevant else 0.0


def mean_ap(lists, qrels):
    aps = [average_precision(lists.get(qid, []), rels) for qid, rels in qrels.items()]
    return sum(aps) / len(aps)


def cached(name, fn):
    path = os.path.join("cache", name + ".json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    lists = fn()
    os.makedirs("cache", exist_ok=True)
    with open(path, "w") as f:
        json.dump(lists, f)
    return lists


def optimize(objective, n_trials):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


# ranked lists over the full train set, cached by parameters
def bm25_lists(train, k1, b):
    k1, b = round(k1, 1), round(b, 2)  # stable cache names
    return cached(
        f"bm25_k1={k1}_b={b}", lambda: retrieval.bm25(train, retrieval.INDEX, k1, b)
    )


def rm3_lists(train):
    return cached(
        f"rm3n_mu={MU}_fd={FB_DOCS}_ft={FB_TERMS}_w={ORIG_WEIGHT}",
        lambda: retrieval.rm3(
            train, retrieval.INDEX, MU, FB_DOCS, FB_TERMS, ORIG_WEIGHT
        ),
    )


def tune(train, qrels):
    def bm25_objective(trial):
        k1 = trial.suggest_float("k1", *K1_RANGE, step=K1_STEP)
        b = trial.suggest_float("b", *B_RANGE, step=B_STEP)
        return mean_ap(bm25_lists(train, k1, b), qrels)

    p = optimize(bm25_objective, n_trials=BM25_TRIALS)
    best_bm25 = bm25_lists(train, p["k1"], p["b"])
    best_rm3 = rm3_lists(train)

    def rrf_objective(trial):
        k = trial.suggest_int("k", *RRF_K_RANGE)
        return mean_ap(retrieval.rrf(best_bm25, best_rm3, k), qrels)

    k = optimize(rrf_objective, n_trials=RRF_TRIALS)["k"]
    return best_bm25, best_rm3, retrieval.rrf(best_bm25, best_rm3, k)


def main():
    queries = retrieval.parse_queries(retrieval.QUERIES)
    train = dict(list(queries.items())[: retrieval.N_TRAIN])
    qrels = parse_qrels(os.path.join(retrieval.DATA, "qrels_50_Queries"))

    for name, lists in zip(("bm25", "rm3", "rrf"), tune(train, qrels)):
        print(f"{name}: map={mean_ap(lists, qrels):.4f}")


if __name__ == "__main__":
    main()
