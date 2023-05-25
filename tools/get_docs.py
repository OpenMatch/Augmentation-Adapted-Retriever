from utils import load_from_trec
import argparse
import csv
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=str, required=False)
    parser.add_argument("--ra_name", type=str, required=False)
    parser.add_argument("--FiD", action="store_true")

    parser.add_argument("run")
    args = parser.parse_args()

    collection = {}
    csv.field_size_limit(500 * 1024 * 1024)
    with open(args.collection, "r") as f:
        reader = csv.DictReader(f, fieldnames=["id", "title", "text"], delimiter="\t")
        for row in reader:
            # The id_ is the same as the index
            id_ = row.pop("id")
            collection[id_] = row

    run = load_from_trec(args.run, as_list=True)

    dataset = []
    data_names = {
        "mmlu": "mmlu_msmarco_ra_ance_aar",
        "popQA": "popQA_kilt_wikipedia_ra_ance_aar",
        "marco_qa": "marco_qa_msmarco_ra_ance",
        "kilt": "kilt_kilt_wikipedia_ra_ance",
    }
    for name in data_names:
        if name in args.ra_name:
            data_name = data_names[name]
            break
    with open(f"data/{data_name}/cache/validation.jsonl") as f:
        lines = f.readlines()
        for line in lines:
            dataset.append(json.loads(line))

    id = 0
    for qid, rank_list in run.items():
        texts = []
        for doc_rank, (docid, _) in enumerate(rank_list):
            text = collection[docid]["text"]
            texts.append(text)
            if len(texts) == 10:
                break
        if args.FiD:
            dataset[id].pop("mmlu_demo", None)
            dataset[id]["passages"] = texts
        else:
            dataset[id]["mmlu_demo"] = " ".join(texts)
        id += 1
    os.makedirs(f"data/{args.ra_name}/cache/", exist_ok=True)
    g = open(
        f"data/{args.ra_name}/cache/validation.jsonl",
        "w",
    )
    for data in dataset:
        g.write(json.dumps(data) + "\n")
