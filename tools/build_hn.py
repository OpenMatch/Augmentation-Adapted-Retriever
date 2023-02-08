# Adapted from Tevatron (https://github.com/texttron/tevatron)

from utils import SimpleTrainPreProcessor as TrainPreProcessor
from argparse import ArgumentParser
from transformers import AutoTokenizer
import os

import random
from tqdm import tqdm
from datetime import datetime
from multiprocessing import Pool


def load_ranking(rank_file, relevance, n_sample, depth, skip_sample):
    with open(rank_file) as rf:
        lines = iter(rf)
        q_0, _, p_0, rank, _, _ = next(lines).strip().split()
        rank = int(rank)

        curr_q = q_0
        negatives = (
            []
            if q_0 not in relevance or p_0 in relevance[q_0] or rank <= skip_sample
            else [p_0]
        )

        while True:
            try:
                q, _, p, rank, _, _ = next(lines).strip().split()
                rank = int(rank)
                if q != curr_q:
                    negatives = negatives[:depth]
                    random.shuffle(negatives)
                    if curr_q in relevance and len(relevance[curr_q]) == 6:
                        yield curr_q, relevance[curr_q], negatives[:n_sample]
                    curr_q = q
                    negatives = (
                        []
                        if q not in relevance
                        or p in relevance[q]
                        or rank <= skip_sample
                        else [p]
                    )
                else:
                    if q in relevance and p not in relevance[q] and rank > skip_sample:
                        negatives.append(p)
            except StopIteration:
                negatives = negatives[:depth]
                random.shuffle(negatives)
                if curr_q in relevance and len(relevance[curr_q]) == 6:
                    yield curr_q, relevance[curr_q], negatives[:n_sample]
                return


random.seed(datetime.now())
parser = ArgumentParser()
parser.add_argument("--tokenizer_name", required=True)
parser.add_argument("--hn_file", required=True)
parser.add_argument("--qrels", required=True)
parser.add_argument("--queries", required=True)
parser.add_argument("--collection", required=True)
parser.add_argument("--save_to", required=True)
parser.add_argument("--doc_template", type=str, default=None)
parser.add_argument("--query_template", type=str, default=None)
parser.add_argument("--query_max_len", type=int, default=32)

parser.add_argument("--truncate", type=int, default=128)
parser.add_argument("--n_sample", type=int, default=30)
parser.add_argument("--depth", type=int, default=200)
parser.add_argument("--skip_sample", type=int, default=0)
parser.add_argument("--mp_chunk_size", type=int, default=500)
parser.add_argument("--shard_size", type=int, default=45000)

args = parser.parse_args()

qrel = TrainPreProcessor.read_qrel(args.qrels)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
processor = TrainPreProcessor(
    query_file=args.queries,
    query_max_len=args.query_max_len,
    collection_file=args.collection,
    tokenizer=tokenizer,
    doc_max_len=args.truncate,
    doc_template=args.doc_template,
    query_template=args.query_template,
    allow_not_found=True,
)

counter = 0
shard_id = 0
f = None
os.makedirs(args.save_to, exist_ok=True)

pbar = tqdm(
    load_ranking(args.hn_file, qrel, args.n_sample, args.depth, args.skip_sample)
)
with Pool() as p:
    for x in p.imap(processor.process_one, pbar, chunksize=args.mp_chunk_size):
        counter += 1
        if f is None:
            f = open(os.path.join(args.save_to, f"split{shard_id:02d}.hn.jsonl"), "w")
            pbar.set_description(f"split - {shard_id:02d}")
        f.write(x + "\n")

        if counter == args.shard_size:
            f.close()
            f = None
            shard_id += 1
            counter = 0

if f is not None:
    f.close()
