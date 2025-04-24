#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_IF_BASE_PORT=60050

MindSpeed_LLM_PATH=../MindSpeed-Core-MS/MindSpeed-LLM

NPUS_PER_NODE=8
MASTER_PORT=6015
NNODES=1
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
DATA_PATH="/home/jenkins/mindspore/testcases/testcases/tests/st/networks/test_qwen_sft/dataset/finetune_dataset/alpaca"
TOKENIZER_PATH="/home/workspace/mindspore_dataset/msadapter/test_input/net/test_qwen_sft/R1"
DATA_CACHE_PATH="./alpaca/"

TP=2
PP=4
SEQ_LEN=8192
MBS=2
GBS=32

DISTRIBUTED_ARGS="
    --worker_num $WORLD_SIZE \
    --local_worker_num $NPUS_PER_NODE \
    --master_port $MASTER_PORT \
    --log_dir=msrun_log_qwen_sft \
    --join=True \
    --cluster_time_out=300 \
    --bind_core=True \
"

TUNE_ARGS="
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --tokenizer-not-use-fast \
    --variable-seq-lengths \
    --prompt-type qwen_r1 \
"

GPT_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers 4  \
    --hidden-size 1792  \
    --ffn-hidden-size 18944 \
    --num-attention-heads 28  \
    --max-position-embeddings ${SEQ_LEN} \
    --seq-length ${SEQ_LEN} \
    --disable-bias-linear \
    --add-qkv-bias \
    --group-query-attention \
    --num-query-groups 4 \
    --use-flash-attn \
    --swiglu \
    --use-fused-swiglu \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --use-fused-rmsnorm \
    --position-embedding-type rope \
    --rotary-base 1000000 \
    --use-fused-rotary-pos-emb \
    --untie-embeddings-and-output-weights \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 152064 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --train-iters 5 \
    --lr 1.25e-6 \
    --lr-decay-style cosine \
    --min-lr 1.25e-7 \
    --lr-warmup-fraction 0.03 \
    --init-method-std 0.01 \
    --weight-decay 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --no-gradient-accumulation-fusion \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --use-distributed-optimizer \
    --sequence-parallel \
    --reuse-fp32-param \
    --bf16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --data-cache-path $DATA_CACHE_PATH \
    --split 100,0,0
"

CKPT_ARGS="
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --seed 1234 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2500 \
    --eval-interval 2000 \
    --eval-iters 0 \
    --log-throughput \
"

msrun $DISTRIBUTED_ARGS ${MindSpeed_LLM_PATH}/posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $CKPT_ARGS \
    $OUTPUT_ARGS \
    $TUNE_ARGS \
    --distributed-backend nccl \
    | tee qwen_sft.txt