# python scripts/msmarco/get_docs.py \
#     --collection marco/corpus.tsv \
#     --ra_name mmlu_msmarco_ra_FiD_contriever \
#     --FiD \
#     msmarco/contriever/output/mmlu_val.trec

python scripts/msmarco/get_docs.py \
    --collection data_hf/kilt_wikipedia.csv \
    --ra_name popQA_kilt_wikipedia_ra_FiD_contriever_MoMA_all_qa_$checkpoint_num \
    --FiD \
    msmarco/contriever_s2_qa/kilt_wikipedia/checkpoint-$checkpoint_num/popQA.trec

# python scripts/msmarco/get_docs.py \
#     --collection data_hf/kilt_wikipedia.csv \
#     --ra_name mmlu_kilt_wikipedia_ra_FiD_MoMA_all_qa_70000 \
#     --FiD \
#     msmarco/t5-ance_s2_qa/kilt_wikipedia/checkpoint-70000/mmlu_val.trec

# python scripts/msmarco/get_docs.py \
#     --collection marco/corpus.tsv \
#     --ra_name marco_qa_msmarco_ra_FiD_contriever \
#     --FiD \
#     msmarco/contriever/output/marco_qa.trec

# python scripts/msmarco/get_docs.py \
#     --collection marco/corpus.tsv \
#     --ra_name mmlu_all_train_msmarco_ra_FiD_MoMA_all_qa_70000 \
#     --FiD \
#     msmarco/t5-ance_s2_qa/output/checkpoint-$checkpoint_num/mmlu_all_train.trec