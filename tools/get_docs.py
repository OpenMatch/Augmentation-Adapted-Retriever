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
    data_names = [
        "popQA",
        "marco_qa",
        "mmlu_all_train_msmarco_ra_FiD",
        "mmlu_all_train_train_msmarco_ra_FiD",
        "mmlu_msmarco_ra_FiD",
        "mmlu_kilt_wikipedia_ra_FiD",
    ]
    for name in data_names:
        if name in args.ra_name:
            data_name = name
            break
    split_name = "validation"
    if data_name == "mmlu_all_train_msmarco_ra_FiD" and "mmlu_all_train" in args.run:
        split_name = "train"
    with open(
        f"/data/private/yuzc/Flan-T5-RA/data_hf/{data_name}/cache/{split_name}.jsonl"
    ) as f:
        lines = f.readlines()
        for line in lines:
            dataset.append(json.loads(line))

    id = 0
    for qid, rank_list in run.items():
        texts = []
        for doc_rank, (docid, _) in enumerate(rank_list):
            text = collection[docid]["text"]
            texts.append(text)
            if len(texts) == 20:
                break
        if args.FiD:
            dataset[id].pop("mmlu_demo", None)
            # random.shuffle(texts)
            dataset[id]["passages"] = texts
        else:
            dataset[id]["mmlu_demo"] = " ".join(texts)
        id += 1
    os.makedirs(
        f"/data/private/yuzc/Flan-T5-RA/data_hf/{args.ra_name}/cache/", exist_ok=True
    )
    g = open(
        f"/data/private/yuzc/Flan-T5-RA/data_hf/{args.ra_name}/cache/{split_name}.jsonl",
        "w",
    )
    for data in dataset:
        g.write(json.dumps(data) + "\n")
    # print(np.mean([len(t) for t in texts])) # 370
