import os
import re
import subprocess
import xml.etree.ElementTree as ET

DATA = "/data/IRCompetition"
INDEX = os.path.join(DATA, "ROBUSTindex")
QUERIES = os.path.join(DATA, "queriesROBUST.xml")
N_TRAIN = 50


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def parse_queries(path):
    root = ET.parse(path).getroot()
    queries = {}
    for q in root.iter("query"):
        qid = q.find("number").text.strip()
        queries[qid] = clean(q.find("text").text)
    return queries


def clean(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def indri_run(queries, index, extra_args, count=1000):
    body = "".join(
        f"<query><number>{qid}</number><text>{text}</text></query>"
        for qid, text in queries.items()
    )
    params_path = "indri_params.xml"
    with open(params_path, "w") as f:
        f.write(f"<parameters>{body}</parameters>")
    out = _run(
        [
            "IndriRunQuery",
            params_path,
            f"-index={index}",
            f"-count={count}",
            "-trecFormat=true",
        ]
        + list(extra_args)
    )
    return parse_trec(out)


def parse_trec(text):
    lists = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 6:
            continue
        qid, _, docid, _, score, _ = parts
        lists.setdefault(qid, []).append((docid, float(score)))
    return lists


def bm25(queries, index, k1, b, count=1000):
    baseline = f"-baseline=okapi,k1:{k1},b:{b},k3:{count}"
    return indri_run(queries, index, [baseline], count)


def rm3(queries, index, mu, fb_docs, fb_terms, orig_weight, count=1000):
    return indri_run(
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


def rrf(lists_a, lists_b, k, count=1000):
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
