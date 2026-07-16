import retrieval

# K1, B, RRF_K set from the best values printed by tune.py;
# RM3 uses standard defaults (Indri/Anserini), not tuned
K1, B = 0.9, 0.4
MU, FB_DOCS, FB_TERMS, ORIG_WEIGHT = 2500, 10, 10, 0.5
RRF_K = 60


def main():
    queries = retrieval.parse_queries(retrieval.QUERIES)
    test = dict(list(queries.items())[retrieval.N_TRAIN:])
    bm25_lists = retrieval.bm25(test, retrieval.INDEX, K1, B)
    rm3_lists = retrieval.rm3(test, retrieval.INDEX, MU, FB_DOCS, FB_TERMS, ORIG_WEIGHT)
    fused = retrieval.rrf(bm25_lists, rm3_lists, RRF_K)
    retrieval.write_run(fused, "run_1.res", "run_1")
    retrieval.write_run(rm3_lists, "run_2.res", "run_2")
    retrieval.write_run(bm25_lists, "run_3.res", "run_3")


if __name__ == "__main__":
    main()
