def rrf(lists_a, lists_b, k=60, count=1000):
    fused = {}
    for qid in set(lists_a) | set(lists_b):
        scores = {}
        for lists in (lists_a, lists_b):
            for rank, (docid, _) in enumerate(lists.get(qid, []), start=1):
                scores[docid] = scores.get(docid, 0.0) + 1.0 / (k + rank)
        fused[qid] = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:count]
    return fused


def write_run(lists, path, tag):
    with open(path, "w") as f:
        for qid in sorted(lists, key=int):
            for rank, (docid, score) in enumerate(lists[qid], start=1):
                f.write(f"{qid} Q0 {docid} {rank} {score} {tag}\n")
