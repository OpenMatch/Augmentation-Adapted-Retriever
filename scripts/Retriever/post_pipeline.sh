# Here is the step:
# 1. First retrieve initial training docs for reader.
# 2. Reader generate attention file (train and dev).
# 3. Generate qrels based on the attention.
# 4. Build hard negative.
# 5. Train the retriever with the hard negative.
# 6. Run this file.

python src/openmatch/driver/build_index.py  \
    --output_dir msmarco/t5-ance_s2_qa/output/checkpoint-$checkpoint_num  \
    --model_name_or_path msmarco/t5-ance_s2_qa/output/checkpoint-$checkpoint_num  \
    --per_device_eval_batch_size 256  \
    --corpus_path marco/corpus.tsv  \
    --doc_template "Title: <title> Text: <text>"  \
    --doc_column_names id,title,text  \
    --q_max_len 32  \
    --p_max_len 128  \
    --fp16  \
    --dataloader_num_workers 1

python src/openmatch/driver/retrieve.py  \
    --output_dir msmarco/t5-ance_s2_qa/output/checkpoint-$checkpoint_num  \
    --model_name_or_path msmarco/t5-ance_s2_qa/output/checkpoint-$checkpoint_num  \
    --per_device_eval_batch_size 256  \
    --query_path marco/dev.query.txt  \
    --query_template "<text>"  \
    --query_column_names id,text  \
    --q_max_len 32  \
    --fp16  \
    --trec_save_path msmarco/t5-ance_s2_qa/output/checkpoint-$checkpoint_num/dev.query.trec  \
    --dataloader_num_workers 1

# python src/openmatch/driver/retrieve.py  \
#     --output_dir msmarco/t5-ance_s2_qa/output/checkpoint-$checkpoint_num  \
#     --model_name_or_path msmarco/t5-ance_s2_qa/output/checkpoint-$checkpoint_num  \
#     --per_device_eval_batch_size 256  \
#     --query_path marco/mmlu_val.csv  \
#     --query_template "<text>"  \
#     --query_column_names id,text  \
#     --q_max_len 512  \
#     --fp16  \
#     --trec_save_path msmarco/t5-ance_s2_qa/output/checkpoint-$checkpoint_num/mmlu_val.trec  \
#     --dataloader_num_workers 1

# python scripts/msmarco/get_docs.py \
#     --collection marco/corpus.tsv \
#     --ra_name mmlu_msmarco_ra_FiD_MoMA_all_qa_$checkpoint_num \
#     --FiD \
#     msmarco/t5-ance_s2_qa/output/checkpoint-$checkpoint_num/mmlu_val.trec

# python src/openmatch/driver/build_index.py  \
#     --output_dir msmarco/contriever_s2_qa/output/checkpoint-$checkpoint_num  \
#     --model_name_or_path msmarco/contriever_s2_qa/output/checkpoint-$checkpoint_num  \
#     --pooling mean  \
#     --per_device_eval_batch_size 1024  \
#     --corpus_path marco/corpus.tsv  \
#     --doc_template "<title>[SEP]<text>"  \
#     --doc_column_names id,title,text  \
#     --q_max_len 32  \
#     --p_max_len 128  \
#     --fp16  \
#     --dataloader_num_workers 1

# python src/openmatch/driver/retrieve.py  \
#     --output_dir msmarco/contriever_s2_qa/output/checkpoint-$checkpoint_num  \
#     --model_name_or_path msmarco/contriever_s2_qa/output/checkpoint-$checkpoint_num  \
#     --pooling mean  \
#     --per_device_eval_batch_size 256  \
#     --query_path marco/mmlu_val.csv  \
#     --query_template "<text>"  \
#     --query_column_names id,text  \
#     --q_max_len 512  \
#     --fp16  \
#     --trec_save_path msmarco/contriever_s2_qa/output/checkpoint-$checkpoint_num/mmlu_val.trec  \
#     --dataloader_num_workers 1

# python scripts/msmarco/get_docs.py \
#     --collection marco/corpus.tsv \
#     --ra_name mmlu_msmarco_ra_FiD_contriever_MoMA_all_qa_$checkpoint_num \
#     --FiD \
#     msmarco/contriever_s2_qa/output/checkpoint-$checkpoint_num/mmlu_val.trec

# python src/openmatch/driver/build_index.py  \
#     --output_dir msmarco/contriever_s2_qa/kilt_wikipedia/checkpoint-$checkpoint_num  \
#     --model_name_or_path msmarco/contriever_s2_qa/output/checkpoint-$checkpoint_num \
#     --pooling mean  \
#     --per_device_eval_batch_size 1024  \
#     --corpus_path data_hf/kilt_wikipedia.csv  \
#     --doc_template "<title>[SEP]<text>"  \
#     --doc_column_names id,title,text  \
#     --q_max_len 32  \
#     --p_max_len 128  \
#     --fp16  \
#     --dataloader_num_workers 1

# python src/openmatch/driver/retrieve.py  \
#     --output_dir msmarco/contriever_s2_qa/kilt_wikipedia/checkpoint-$checkpoint_num  \
#     --model_name_or_path msmarco/contriever_s2_qa/output/checkpoint-$checkpoint_num \
#     --pooling mean  \
#     --per_device_eval_batch_size 256  \
#     --query_path marco/popQA.csv  \
#     --query_template "<text>"  \
#     --query_column_names id,text  \
#     --q_max_len 32  \
#     --fp16  \
#     --trec_save_path msmarco/contriever_s2_qa/kilt_wikipedia/checkpoint-$checkpoint_num/popQA.trec  \
#     --dataloader_num_workers 1