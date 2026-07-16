import json
import os

import optuna

import bm25
import fusion
import rm3
import utils


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


def optimize(name, objective, n_trials):
    study = optuna.create_study(direction="maximize", study_name=name)
    study.optimize(objective, n_trials=n_trials)
    print(f"best {name}: {study.best_params} map={study.best_value:.4f}")
    return study.best_params


def main():
    queries = utils.parse_queries(utils.QUERIES)
    train = dict(list(queries.items())[: utils.N_TRAIN])

    qrels = parse_qrels(os.path.join(utils.DATA, "qrels_50_Queries"))

    def bm25_lists(k1, b):
        k1, b = round(k1, 1), round(b, 2)  # stable cache names
        return cached(
            f"bm25_k1={k1}_b={b}", lambda: bm25.rank(train, utils.INDEX, k1, b)
        )

    def ql_lists(mu):
        return cached(f"ql_mu={mu}", lambda: rm3.initial_run(train, utils.INDEX, mu))

    # stage 1: tune mu on the plain QL run, then freeze it for feedback
    mu = max(range(250, 3250, 250), key=lambda m: mean_ap(ql_lists(m), qrels))
    print(f"best mu: {mu} map={mean_ap(ql_lists(mu), qrels):.4f}")

    def rm3_lists(fb_docs, fb_terms, w):
        w = round(w, 2)
        return cached(
            f"rm3n_mu={mu}_fd={fb_docs}_ft={fb_terms}_w={w}",
            lambda: rm3.rank_native(train, utils.INDEX, mu, fb_docs, fb_terms, w),
        )

    def bm25_objective(trial):
        k1 = trial.suggest_float("k1", 0.4, 2.0, step=0.1)
        b = trial.suggest_float("b", 0.1, 1.0, step=0.05)
        return mean_ap(bm25_lists(k1, b), qrels)

    def rm3_objective(trial):
        fb_docs = trial.suggest_int("fb_docs", 10, 100, step=10)
        fb_terms = trial.suggest_int("fb_terms", 10, 100, step=10)
        w = trial.suggest_float("orig_weight", 0.1, 0.9, step=0.05)
        return mean_ap(rm3_lists(fb_docs, fb_terms, w), qrels)

    p = optimize("bm25", bm25_objective, n_trials=30)
    best_bm25 = bm25_lists(p["k1"], p["b"])

    p = optimize("rm3", rm3_objective, n_trials=40)
    best_rm3 = rm3_lists(p["fb_docs"], p["fb_terms"], p["orig_weight"])

    def rrf_objective(trial):
        k = trial.suggest_int("k", 10, 200)
        return mean_ap(fusion.rrf(best_bm25, best_rm3, k), qrels)

    optimize("rrf", rrf_objective, n_trials=20)


if __name__ == "__main__":
    main()
