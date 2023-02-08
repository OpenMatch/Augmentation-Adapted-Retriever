python scripts/msmarco/merge_qrels.py \
    --scores_path /data/private/yuzc/Flan-T5-RA/results/flan-t5-large/fp16/zs/marco_qa_msmarco_ra_FiD/stored_FiD_scores.pt \
    --save_path marco/qrels.marco_qa.tsv \
    msmarco/t5-ance/marco_qa.trec

python scripts/msmarco/build_hn.py  \
    --tokenizer_name checkpoints/t5-ance  \
    --hn_file msmarco/t5-ance/marco_qa.trec  \
    --qrels marco/qrels.marco_qa.tsv  \
    --queries marco/marco_qa.csv  \
    --collection marco/corpus.tsv  \
    --save_to msmarco/t5-ance/marco_qa/  \
    --doc_template "Title: <title> Text: <text>"

cat msmarco/t5-ance/marco_qa/*.hn.jsonl > msmarco/t5-ance/marco_qa/train.hn.jsonl
tail -n 500 msmarco/t5-ance/marco_qa/train.hn.jsonl > msmarco/t5-ance/marco_qa/val.hn.jsonl
head -n -500 msmarco/t5-ance/marco_qa/train.hn.jsonl > msmarco/t5-ance/marco_qa/train.new.hn.jsonl