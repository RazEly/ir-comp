import bm25
import rm3
import fusion
import utils

# set from the best values printed by evaluate.py
K1, B = 0.9, 0.4
MU, FB_DOCS, FB_TERMS, ORIG_WEIGHT = 1000, 25, 25, 0.5
RRF_K = 60


def main():
    queries = utils.parse_queries(utils.QUERIES)
    test = dict(list(queries.items())[utils.N_TRAIN:])
    bm25_lists = bm25.rank(test, utils.INDEX, K1, B)
    rm3_lists = rm3.rank(test, utils.INDEX, MU, FB_DOCS, FB_TERMS, ORIG_WEIGHT)
    fused = fusion.rrf(bm25_lists, rm3_lists, RRF_K)
    fusion.write_run(fused, "run_1.res", "run_1")
    fusion.write_run(rm3_lists, "run_2.res", "run_2")
    fusion.write_run(bm25_lists, "run_3.res", "run_3")


if __name__ == "__main__":
    main()
