import os

import optuna

import bm25
import fusion
import rm3
import utils
from evaluate import cached, mean_ap, parse_qrels

N_FOLDS = 10

optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize(objective, n_trials):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
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

    def rm3_lists(mu, fb_docs, fb_terms, w):
        w = round(w, 2)
        return cached(
            f"rm3n_mu={mu}_fd={fb_docs}_ft={fb_terms}_w={w}",
            lambda: rm3.rank(train, utils.INDEX, mu, fb_docs, fb_terms, w),
        )

    qids = [qid for qid in train if qid in qrels]
    folds = [qids[i::N_FOLDS] for i in range(N_FOLDS)]

    scores = {"bm25": [], "rm3": [], "rrf": []}
    for i, test_qids in enumerate(folds):
        train_qrels = {qid: qrels[qid] for qid in qids if qid not in test_qids}
        test_qrels = {qid: qrels[qid] for qid in test_qids}

        def bm25_objective(trial):
            k1 = trial.suggest_float("k1", 0.4, 2.0, step=0.1)
            b = trial.suggest_float("b", 0.1, 1.0, step=0.05)
            return mean_ap(bm25_lists(k1, b), train_qrels)

        # stage 1: tune mu on the plain QL run over the train fold
        mu = max(
            range(250, 3250, 250), key=lambda m: mean_ap(ql_lists(m), train_qrels)
        )

        def rm3_objective(trial):
            fb_docs = trial.suggest_int("fb_docs", 10, 100, step=10)
            fb_terms = trial.suggest_int("fb_terms", 10, 100, step=10)
            w = trial.suggest_float("orig_weight", 0.1, 0.9, step=0.05)
            return mean_ap(rm3_lists(mu, fb_docs, fb_terms, w), train_qrels)

        p = optimize(bm25_objective, n_trials=30)
        best_bm25 = bm25_lists(p["k1"], p["b"])
        scores["bm25"].append(mean_ap(best_bm25, test_qrels))

        p = optimize(rm3_objective, n_trials=40)
        best_rm3 = rm3_lists(mu, p["fb_docs"], p["fb_terms"], p["orig_weight"])
        scores["rm3"].append(mean_ap(best_rm3, test_qrels))

        def rrf_objective(trial):
            k = trial.suggest_int("k", 10, 200)
            return mean_ap(fusion.rrf(best_bm25, best_rm3, k), train_qrels)

        p = optimize(rrf_objective, n_trials=20)
        scores["rrf"].append(mean_ap(fusion.rrf(best_bm25, best_rm3, p["k"]), test_qrels))

        print(
            f"fold {i}: mu={mu} bm25={scores['bm25'][-1]:.4f} "
            f"rm3={scores['rm3'][-1]:.4f} rrf={scores['rrf'][-1]:.4f}"
        )

    for name, s in scores.items():
        print(f"{name}: cv map={sum(s) / len(s):.4f}")


if __name__ == "__main__":
    main()
