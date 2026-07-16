import json
import os

import optuna

import retrieval

N_FOLDS = 10
MU_GRID = range(250, 3250, 250)

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


def ql_lists(train, mu):
    return cached(f"ql_mu={mu}", lambda: retrieval.ql(train, retrieval.INDEX, mu))


def rm3_lists(train, mu, fb_docs, fb_terms, w):
    w = round(w, 2)
    return cached(
        f"rm3n_mu={mu}_fd={fb_docs}_ft={fb_terms}_w={w}",
        lambda: retrieval.rm3(train, retrieval.INDEX, mu, fb_docs, fb_terms, w),
    )


def tune(train, qrels):
    # stage 1: tune mu on the plain QL run, then freeze it for feedback
    mu = max(MU_GRID, key=lambda m: mean_ap(ql_lists(train, m), qrels))

    def bm25_objective(trial):
        k1 = trial.suggest_float("k1", 0.4, 2.0, step=0.1)
        b = trial.suggest_float("b", 0.1, 1.0, step=0.05)
        return mean_ap(bm25_lists(train, k1, b), qrels)

    def rm3_objective(trial):
        fb_docs = trial.suggest_int("fb_docs", 10, 100, step=10)
        fb_terms = trial.suggest_int("fb_terms", 10, 100, step=10)
        w = trial.suggest_float("orig_weight", 0.1, 0.9, step=0.05)
        return mean_ap(rm3_lists(train, mu, fb_docs, fb_terms, w), qrels)

    bm25_params = optimize(bm25_objective, n_trials=30)
    best_bm25 = bm25_lists(train, bm25_params["k1"], bm25_params["b"])

    rm3_params = optimize(rm3_objective, n_trials=40)
    best_rm3 = rm3_lists(
        train,
        mu,
        rm3_params["fb_docs"],
        rm3_params["fb_terms"],
        rm3_params["orig_weight"],
    )

    def rrf_objective(trial):
        k = trial.suggest_int("k", 10, 200)
        return mean_ap(retrieval.rrf(best_bm25, best_rm3, k), qrels)

    rrf_params = optimize(rrf_objective, n_trials=20)
    params = {"mu": mu, "bm25": bm25_params, "rm3": rm3_params, "rrf": rrf_params}
    return params, best_bm25, best_rm3


def cross_validate(train, qrels):
    qids = [qid for qid in train if qid in qrels]
    folds = [qids[i::N_FOLDS] for i in range(N_FOLDS)]

    scores = {"bm25": [], "rm3": [], "rrf": []}
    for i, test_qids in enumerate(folds):
        train_qrels = {qid: qrels[qid] for qid in qids if qid not in test_qids}
        test_qrels = {qid: qrels[qid] for qid in test_qids}

        params, best_bm25, best_rm3 = tune(train, train_qrels)
        fused = retrieval.rrf(best_bm25, best_rm3, params["rrf"]["k"])
        scores["bm25"].append(mean_ap(best_bm25, test_qrels))
        scores["rm3"].append(mean_ap(best_rm3, test_qrels))
        scores["rrf"].append(mean_ap(fused, test_qrels))

        print(
            f"fold {i}: mu={params['mu']} bm25={scores['bm25'][-1]:.4f} "
            f"rm3={scores['rm3'][-1]:.4f} rrf={scores['rrf'][-1]:.4f}"
        )

    for name, s in scores.items():
        print(f"{name}: cv map={sum(s) / len(s):.4f}")


def main():
    queries = retrieval.parse_queries(retrieval.QUERIES)
    train = dict(list(queries.items())[: retrieval.N_TRAIN])
    qrels = parse_qrels(os.path.join(retrieval.DATA, "qrels_50_Queries"))

    cross_validate(train, qrels)


if __name__ == "__main__":
    main()
