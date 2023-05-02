python src/Retriever/driver/build_index.py  \
    --output_dir data/msmarco/t5-ance_aar/output/checkpoint-70000  \
    --model_name_or_path data/msmarco/t5-ance_aar/output/checkpoint-70000  \
    --per_device_eval_batch_size 256  \
    --corpus_path data/msmarco/corpus.tsv  \
    --doc_template "Title: <title> Text: <text>"  \
    --doc_column_names id,title,text  \
    --q_max_len 32  \
    --p_max_len 128  \
    --fp16  \
    --dataloader_num_workers 1

python src/Retriever/driver/retrieve.py  \
    --output_dir data/msmarco/t5-ance_aar/output/checkpoint-70000  \
    --model_name_or_path data/msmarco/t5-ance_aar/output/checkpoint-70000  \
    --per_device_eval_batch_size 256  \
    --query_path data/msmarco/mmlu_val.csv  \
    --query_template "<text>"  \
    --query_column_names id,text  \
    --q_max_len 512  \
    --fp16  \
    --trec_save_path data/msmarco/t5-ance_aar/output/checkpoint-70000/mmlu_val.trec  \
    --dataloader_num_workers 1

python tools/get_docs.py \
    --collection data/msmarco/corpus.tsv \
    --ra_name mmlu_msmarco_ra_ance_aar \
    --FiD \
    data/msmarco/t5-ance_aar/output/checkpoint-70000/mmlu_val.trec

python src/Retriever/driver/build_index.py  \
    --output_dir data/msmarco/t5-ance_aar/output/checkpoint-70000  \
    --model_name_or_path data/msmarco/t5-ance_aar/output/checkpoint-70000 \
    --per_device_eval_batch_size 1024  \
    --corpus_path data/msmarco/kilt_wikipedia.csv  \
    --doc_template "Title: <title> Text: <text>"  \
    --doc_column_names id,title,text  \
    --q_max_len 32  \
    --p_max_len 128  \
    --fp16  \
    --dataloader_num_workers 1

python src/Retriever/driver/retrieve.py  \
    --output_dir data/msmarco/t5-ance_aar/output/checkpoint-70000  \
    --model_name_or_path data/msmarco/t5-ance_aar/output/checkpoint-70000  \
    --per_device_eval_batch_size 256  \
    --query_path data/msmarco/popQA.csv  \
    --query_template "<text>"  \
    --query_column_names id,text  \
    --q_max_len 512  \
    --fp16  \
    --trec_save_path data/msmarco/t5-ance_aar/output/checkpoint-70000/popQA.trec  \
    --dataloader_num_workers 1

python tools/get_docs.py \
    --collection data/msmarco/kilt_wikipedia.csv \
    --ra_name popQA_kilt_wikipedia_ra_ance_aar \
    --FiD \
    data/msmarco/t5-ance_aar/output/checkpoint-70000/popQA.trec