bash scripts/LM/get_scores.sh

python tools/merge_qrels.py \
    --scores_path results/flan-t5-base/fp16/zs/marco_qa_msmarco_ra_ance/stored_FiD_scores.pt \
    --qrels_path data/msmarco/qrels.train.tsv \
    --save_path data/msmarco/qrels.marco_qa.tsv \
    data/msmarco/t5-ance/marco_qa.trec

python tools/build_hn.py  \
    --tokenizer_name checkpoints/t5-ance  \
    --hn_file data/msmarco/t5-ance/marco_qa.trec  \
    --qrels data/msmarco/qrels.marco_qa.tsv  \
    --queries data/msmarco/marco_qa.csv  \
    --collection data/msmarco/corpus.tsv  \
    --save_to data/msmarco/t5-ance/marco_qa/  \
    --doc_template "Title: <title> Text: <text>"

cat data/msmarco/t5-ance/marco_qa/*.hn.jsonl > data/msmarco/t5-ance/marco_qa/train.hn.jsonl
tail -n 500 data/msmarco/t5-ance/marco_qa/train.hn.jsonl > data/msmarco/t5-ance/marco_qa/val.hn.jsonl
head -n -500 data/msmarco/t5-ance/marco_qa/train.hn.jsonl > data/msmarco/t5-ance/marco_qa/train.new.hn.jsonl