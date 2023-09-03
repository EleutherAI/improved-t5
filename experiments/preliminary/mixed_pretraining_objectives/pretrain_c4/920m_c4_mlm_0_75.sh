#!/bin/bash

ADDR=$1
MODEL_DIR=$2

python -m t5x.train \
    --gin_file="models/scalable_t5/t5_1_1/base.gin" \
    --gin_file="configs/size/920m/vanilla.gin" \
    --gin_file="configs/pretrain/c4_mlm.gin" \
    --gin.MIXTURE_OR_TASK_NAME="c4_mlm_0_75" \
    --gin.MODEL_DIR=\"'/fsx/lintangsutawika/ckpts/base_c4_mlm_0_75/'\" \
    --gin.USE_CACHED_TASKS=False \
    --gin.TRAIN_STEPS=256000 \
    --gin.SAVING_PERIOD=32000 \
    --gin.BATCH_SIZE=2048 \
    --alsologtostderr \
    --multiprocess_gpu \
    --coordinator_address=${ADDR} \
    --process_count=${SLURM_NTASKS} \
    --process_index=${SLURM_PROCID}
