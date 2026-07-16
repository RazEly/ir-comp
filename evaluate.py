import json
import os
import bm25
import rm3
import fusion
import utils

QRELS = os.path.join(utils.DATA, "qrels_50_Queries")
CACHE = "cache"


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
    path = os.path.join(CACHE, name + ".json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    lists = fn()
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "w") as f:
        json.dump(lists, f)
    return lists


def main():
    queries = utils.parse_queries(utils.QUERIES)
    train = dict(list(queries.items())[:utils.N_TRAIN])
    qrels = parse_qrels(QRELS)

    best_bm25, bm25_lists = (0.0, None, None), None
    for k1 in (0.6, 0.9, 1.2):
        for b in (0.25, 0.4, 0.75):
            name = f"bm25_k1={k1}_b={b}"
            lists = cached(name, lambda: bm25.rank(train, utils.INDEX, k1, b))
            m = mean_ap(lists, qrels)
            print(name, round(m, 4))
            if m > best_bm25[0]:
                best_bm25, bm25_lists = (m, k1, b), lists

    vec_cache = {}
    best_rm3, rm3_lists = (0.0, None, None, None, None), None
    for mu in (500, 1000, 2000):
        initial = cached(f"ql_mu={mu}", lambda: rm3.initial_run(train, utils.INDEX, mu))
        for fb_docs in (25, 50):
            for fb_terms in (25, 50):
                for w in (0.3, 0.5, 0.7):
                    name = f"rm3_mu={mu}_fd={fb_docs}_ft={fb_terms}_w={w}"
                    lists = cached(name, lambda: rm3.rank(
                        train, utils.INDEX, mu, fb_docs, fb_terms, w,
                        initial=initial, vec_cache=vec_cache))
                    m = mean_ap(lists, qrels)
                    print(name, round(m, 4))
                    if m > best_rm3[0]:
                        best_rm3, rm3_lists = (m, mu, fb_docs, fb_terms, w), lists

    best_rrf = (0.0, None)
    for k in (20, 60, 100):
        m = mean_ap(fusion.rrf(bm25_lists, rm3_lists, k), qrels)
        print(f"rrf_k={k}", round(m, 4))
        if m > best_rrf[0]:
            best_rrf = (m, k)

    m, k1, b = best_bm25
    print(f"best bm25: k1={k1} b={b} map={m:.4f}")
    m, mu, fb_docs, fb_terms, w = best_rm3
    print(f"best rm3: mu={mu} fb_docs={fb_docs} fb_terms={fb_terms} orig_weight={w} map={m:.4f}")
    m, k = best_rrf
    print(f"best rrf: k={k} map={m:.4f}")


if __name__ == "__main__":
    main()
