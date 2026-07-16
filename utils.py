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


def weight_query(model):
    return "#weight(" + " ".join(f'{w} "{t}"' for t, w in model.items()) + ")"


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


def dumpindex(index, *args):
    return _run(["dumpindex", index] + list(args))
