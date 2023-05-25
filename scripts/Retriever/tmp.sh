python tools/merge_qrels.py \
    --scores_path results/flan-t5-base/fp16/zs/kilt_kilt_wikipedia_ra_ance/stored_FiD_scores.pt \
    --qrels_path /data/private/yuzc/openmatch-research/data_hf/KILT/qrels.kilt.tsv \
    --save_path /data/private/yuzc/openmatch-research/marco/qrels.kilt.tsv \
    /data/private/yuzc/openmatch-research/msmarco/t5-ance/kilt_wikipedia/kilt.trec

python scripts/msmarco/build_hn.py  \
    --tokenizer_name checkpoints/t5-ance  \
    --hn_file msmarco/t5-ance/kilt_wikipedia/kilt.trec  \
    --qrels marco/qrels.kilt.tsv  \
    --queries data_hf/KILT/kilt.csv  \
    --collection data_hf/kilt_wikipedia.csv  \
    --save_to msmarco/t5-ance/kilt/  \
    --doc_template "Title: <title> Text: <text>"

cat msmarco/t5-ance/kilt/*.hn.jsonl > msmarco/t5-ance/kilt/train.hn.jsonl
tail -n 500 msmarco/t5-ance/kilt/train.hn.jsonl > msmarco/t5-ance/kilt/val.hn.jsonl
head -n -500 msmarco/t5-ance/kilt/train.hn.jsonl > msmarco/t5-ance/kilt/train.new.hn.jsonl

python src/openmatch/driver/train_dr.py  \
    --output_dir msmarco/t5-ance_aar/output  \
    --model_name_or_path checkpoints/t5-ance  \
    --do_train  \
    --save_steps 1000  \
    --eval_steps 400  \
    --train_path msmarco/t5-ance/kilt/train.new.hn.jsonl  \
    --eval_path msmarco/t5-ance/kilt/val.hn.jsonl  \
    --fp16  \
    --per_device_train_batch_size 8  \
    --train_n_passages 8  \
    --use_all_positive_passages  \
    --learning_rate 5e-6  \
    --q_max_len 32  \
    --p_max_len 128  \
    --num_train_epochs 6  \
    --logging_dir msmarco/t5-ance_aar/log  \
    --evaluation_strategy steps