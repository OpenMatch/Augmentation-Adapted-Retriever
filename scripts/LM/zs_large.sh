#! /bin/bash

WORKING_DIR="YOUR_WORKING_DIR"

MP_SIZE=1

NUM_GPUS_PER_WORKER=1 # number of gpus used on one node

DATA_EXT=".jsonl"
DATA_NAMES="mmlu_msmarco_ra_FiD_MoMA_all_qa_70000"
# DATA_NAMES="mmlu_msmarco_ra_FiD_contriever_MoMA_all_qa_90000"
# DATA_NAMES="popQA_kilt_wikipedia_ra_FiD_MoMA_all_qa_70000"

MASTER_PORT=${1-$(expr $RANDOM + 1000)}
# CKPT=${1}
SEED=10

CONFIG_PATH="${WORKING_DIR}/configs/model/t5_large_config.json"
CKPT_PATH="${WORKING_DIR}/checkpoints/flan-t5-large/t5-MP1/"

SAVE_PATH="${WORKING_DIR}/results/flan-t5-large/fp16/zs/${DATA_NAMES}"

LOG_FILE="${SAVE_PATH}/log.txt"
DS_CONFIG="${WORKING_DIR}/configs/deepspeed/ds_fp16.json"
TOKENIZER_PATH="${WORKING_DIR}/vocab_en"

BATCH_SIZE=9 # 1530
# BATCH_SIZE=11


OPTS=""
OPTS+=" --model-config ${CONFIG_PATH}"
OPTS+=" --model-parallel-size ${MP_SIZE}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --dev-batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${BATCH_SIZE}"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --log-file ${LOG_FILE}"
OPTS+=" --load ${CKPT_PATH}"
OPTS+=" --data-names ${DATA_NAMES}"
OPTS+=" --FiD"
OPTS+=" --passage_num 6"
OPTS+=" --distributed-backend nccl"
OPTS+=" --no-load-optim"
OPTS+=" --lr-decay-style constant"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --tokenizer-path ${TOKENIZER_PATH}"
OPTS+=" --checkpoint-activations"
OPTS+=" --deepspeed-activation-checkpointing"
OPTS+=" --fp16"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${DS_CONFIG}"
OPTS+=" --do-eval"
OPTS+=" --test-num 100000"
OPTS+=" --seed ${SEED}"
OPTS+=" --eval-per-prompt"

CMD="torchrun --master_port ${MASTER_PORT} --nproc_per_node ${NUM_GPUS_PER_WORKER} ${WORKING_DIR}/train_t0.py ${OPTS}"

echo ${CMD}
mkdir -p ${SAVE_PATH}
${CMD} 2>&1 | tee ${SAVE_PATH}/train_log
