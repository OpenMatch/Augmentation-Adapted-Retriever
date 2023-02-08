# python src/openmatch/driver/retrieve.py  \
#     --output_dir msmarco/t5-ance_s2_qa/kilt_wikipedia/checkpoint-70000  \
#     --model_name_or_path msmarco/t5-ance_s2_qa/output/checkpoint-70000  \
#     --per_device_eval_batch_size 256  \
#     --query_path marco/mmlu_val.csv  \
#     --query_template "<text>"  \
#     --query_column_names id,text  \
#     --q_max_len 512  \
#     --fp16  \
#     --trec_save_path msmarco/t5-ance_s2_qa/kilt_wikipedia/checkpoint-70000/mmlu_val.trec  \
#     --dataloader_num_workers 1

# python src/openmatch/driver/retrieve.py  \
#     --output_dir msmarco/t5-ance_s2_qa/output/checkpoint-70000  \
#     --model_name_or_path msmarco/t5-ance_s2_qa/output/checkpoint-70000 \
#     --per_device_eval_batch_size 256  \
#     --query_path marco/popQA.csv  \
#     --query_template "<text>"  \
#     --query_column_names id,text  \
#     --q_max_len 32  \
#     --fp16  \
#     --trec_save_path msmarco/t5-ance_s2_qa/output/checkpoint-70000/popQA.trec  \
#     --dataloader_num_workers 1

# python src/openmatch/driver/retrieve.py  \
#     --output_dir msmarco/contriever/output  \
#     --model_name_or_path checkpoints/contriever  \
#     --pooling mean  \
#     --per_device_eval_batch_size 256  \
#     --query_path marco/mmlu_val.csv  \
#     --query_template "<text>"  \
#     --query_column_names id,text  \
#     --q_max_len 512  \
#     --fp16  \
#     --trec_save_path msmarco/contriever/output/mmlu_val.trec  \
#     --dataloader_num_workers 1

# python src/openmatch/driver/retrieve.py  \
#     --output_dir msmarco/contriever/output  \
#     --model_name_or_path checkpoints/contriever  \
#     --pooling mean  \
#     --per_device_eval_batch_size 256  \
#     --query_path marco/marco_qa.csv  \
#     --query_template "<text>"  \
#     --query_column_names id,text  \
#     --q_max_len 32  \
#     --fp16  \
#     --trec_save_path msmarco/contriever/output/marco_qa.trec  \
#     --dataloader_num_workers 1

# python src/openmatch/driver/retrieve.py  \
#     --output_dir msmarco/contriever/output  \
#     --model_name_or_path checkpoints/contriever  \
#     --pooling mean  \
#     --per_device_eval_batch_size 256  \
#     --query_path marco/dev.query.txt  \
#     --query_template "<text>"  \
#     --query_column_names id,text  \
#     --q_max_len 32  \
#     --fp16  \
#     --trec_save_path msmarco/contriever/output/dev.query.trec  \
#     --dataloader_num_workers 1

# python src/openmatch/driver/retrieve.py  \
#     --output_dir msmarco/t5-ance_s2_qa/output/checkpoint-$checkpoint_num  \
#     --model_name_or_path msmarco/t5-ance_s2_qa/output/checkpoint-$checkpoint_num  \
#     --per_device_eval_batch_size 256  \
#     --query_path marco/mmlu_all_train.csv  \
#     --query_template "<text>"  \
#     --query_column_names id,text  \
#     --q_max_len 512  \
#     --fp16  \
#     --trec_save_path msmarco/t5-ance_s2_qa/output/checkpoint-$checkpoint_num/mmlu_all_train.trec  \
#     --dataloader_num_workers 1

# python src/openmatch/driver/retrieve.py  \
#     --output_dir msmarco/t5-ance_s2_qa/output/checkpoint-$checkpoint_num  \
#     --model_name_or_path msmarco/t5-ance_s2_qa/output/checkpoint-$checkpoint_num  \
#     --per_device_eval_batch_size 256  \
#     --query_path marco/mmlu_all_validation.csv  \
#     --query_template "<text>"  \
#     --query_column_names id,text  \
#     --q_max_len 512  \
#     --fp16  \
#     --trec_save_path msmarco/t5-ance_s2_qa/output/checkpoint-$checkpoint_num/mmlu_all_validation.trec  \
#     --dataloader_num_workers 1

python src/openmatch/driver/retrieve.py  \
    --output_dir msmarco/t5-ance_s2_qa/output/checkpoint-$checkpoint_num  \
    --model_name_or_path msmarco/t5-ance_s2_qa/output/checkpoint-$checkpoint_num  \
    --per_device_eval_batch_size 256  \
    --query_path marco/marco_qa_dev.csv  \
    --query_template "<text>"  \
    --query_column_names id,text  \
    --q_max_len 32  \
    --fp16  \
    --trec_save_path msmarco/t5-ance_s2_qa/output/checkpoint-$checkpoint_num/marco_qa_dev.trec  \
    --dataloader_num_workers 1

# python src/openmatch/driver/build_index.py  \
#     --output_dir msmarco/t5-ance_s2_qa/output/checkpoint-$checkpoint_num  \
#     --model_name_or_path msmarco/t5-ance_s2_qa/output/checkpoint-$checkpoint_num  \
#     --per_device_eval_batch_size 256  \
#     --corpus_path marco/corpus.tsv  \
#     --doc_template "Title: <title> Text: <text>"  \
#     --doc_column_names id,title,text  \
#     --q_max_len 32  \
#     --p_max_len 128  \
#     --fp16  \
#     --dataloader_num_workers 1