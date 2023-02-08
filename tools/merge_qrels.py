from utils import load_from_trec
import argparse
import torch
import csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_path", type=str)
    parser.add_argument("--save_path", type=str)

    parser.add_argument("run")
    args = parser.parse_args()

    scores = torch.load(args.scores_path)
    print(scores.size())
    run = load_from_trec(args.run, as_list=True)

    g = open(args.save_path, "w")

    qrels = {}
    with open("marco/qrels.train.tsv", encoding="utf8") as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [qid, _, docid, rel] in tsvreader:
            assert rel == "1"
            if qid in qrels:
                qrels[qid].append(docid)
            else:
                qrels[qid] = [docid]

    id = 0
    sum, overlap = 0, 0
    for qid, rank_list in run.items():
        # if id in ignore_index:
        #     id += 1
        #     continue
        docids = []
        for doc_rank, (docid, _) in enumerate(rank_list):
            docids.append(docid)
            if len(docids) == 10:
                break
        sort_scores, sort_index = torch.sort(scores[id], descending=True)
        for docid in qrels[qid]:
            # pass
            g.write(f"{qid}\t0\t{docid}\t1\n")
        sum += len(qrels[qid])
        for i in sort_index[:5]:
            if docids[i] not in qrels[qid]:
                # pass
                g.write(f"{qid}\t0\t{docids[i]}\t1\n")
            else:
                overlap += 1
        id += 1
        if id >= scores.size(0):
            break
    print(overlap, sum, overlap / sum)
