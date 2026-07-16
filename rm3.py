import math

import utils


def initial_run(queries, index, mu, count=1000):
    return utils.indri_run(queries, index, [f"-rule=method:dirichlet,mu:{mu}"], count)


def rank_native(queries, index, mu, fb_docs, fb_terms, orig_weight, count=1000):
    # Indri's built-in relevance-model feedback; fbMu defaults to 0,
    # i.e. unsmoothed ML term estimates, matching rm1() below.
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


def doc_vector(index, docid):
    internal = utils.dumpindex(index, "di", "docno", docid).strip()
    vec = {}
    for line in utils.dumpindex(index, "dv", internal).splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0].isdigit():
            term = parts[2]
            if term != "[OOV]":
                vec[term] = vec.get(term, 0) + 1
    return vec


def rm1(doc_vecs, log_scores):
    # Indri LM scores are log P(q|M_d), so posteriors p(M_d|q) ~ exp(score)
    m = max(log_scores)
    ps = [math.exp(s - m) for s in log_scores]
    z = sum(ps)
    rm = {}
    for vec, p in zip(doc_vecs, ps):
        dl = sum(vec.values())
        w = p / z
        for t, c in vec.items():
            rm[t] = rm.get(t, 0.0) + w * (c / dl)
    return rm


def interpolate_clip(rm, qtext, qweight, k):
    qterms = qtext.split()
    model = {t: (1 - qweight) * p for t, p in rm.items()}
    for t in qterms:
        model[t] = model.get(t, 0.0) + qweight / len(qterms)
    top = sorted(model.items(), key=lambda x: x[1], reverse=True)[:k]
    z = sum(p for _, p in top)
    return {t: p / z for t, p in top}


def rank(
    queries,
    index,
    mu=1000,
    fb_docs=25,
    fb_terms=25,
    orig_weight=0.5,
    count=1000,
    initial=None,
    vec_cache=None,
):
    if initial is None:
        initial = initial_run(queries, index, mu, count)
    if vec_cache is None:
        vec_cache = {}
    expanded = {}
    for qid, qtext in queries.items():
        top = initial.get(qid, [])[:fb_docs]
        if not top:
            expanded[qid] = qtext
            continue
        for docid, _ in top:
            if docid not in vec_cache:
                vec_cache[docid] = doc_vector(index, docid)
        vecs = [vec_cache[docid] for docid, _ in top]
        scores = [s for _, s in top]
        model = interpolate_clip(rm1(vecs, scores), qtext, orig_weight, fb_terms)
        expanded[qid] = utils.weight_query(model)
    return utils.indri_run(expanded, index, [f"-rule=method:dirichlet,mu:{mu}"], count)
