import utils


def initial_run(queries, index, mu, count=1000):
    return utils.indri_run(queries, index, [f"-rule=method:dirichlet,mu:{mu}"], count)


def rank(queries, index, mu=1000, fb_docs=25, fb_terms=25, orig_weight=0.5, count=1000):
    return utils.indri_run(
        queries,
        index,
        [
            f"-rule=method:dirichlet,mu:{mu}",
            f"-fbDocs={fb_docs}",
            f"-fbTerms={fb_terms}",
            f"-fbOrigWeight={orig_weight}",
        ],
        count,
    )
