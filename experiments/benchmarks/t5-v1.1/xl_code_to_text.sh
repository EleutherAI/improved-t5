#!/bin/bash

CODE_LANG=$1

if [[ $CODE_LANG == "python" ]]; then
    # python 251,820 -> 7869
    STEPS=7869
elif [[ $CODE_LANG == "php" ]]; then
    # php 241,241 -> 7538
    STEPS=7538
elif [[ $CODE_LANG == "go" ]]; then
    # go 167,288 -> 5227
    STEPS=5227
elif [[ $CODE_LANG == "java" ]]; then
    # java 164,923 -> 5154
    STEPS=5154
elif [[ $CODE_LANG == "javascript" ]]; then
    # javascript 58,025 -> 1813
    STEPS=1813
elif [[ $CODE_LANG == "ruby" ]]; then
    # ruby 24,927 -> 779
    STEPS=779
fi

SAVING_PERIOD=$STEPS
TRAIN_STEPS=$(( ${STEPS} * 10 + 1000000 ))

python -m t5x.train \
    --gin_file="../t5x/t5x/examples/t5/t5_1_1/xl.gin" \
    --gin_file="configs/task/finetune/codexglue/code_to_text_${CODE_LANG}.gin" \
    --gin.MIXTURE_OR_TASK_NAME=\""code_to_text_${CODE_LANG}_t5"\" \
    --gin.partitioning.PjitPartitioner.model_parallel_submesh="(1, 1, 2, 1)" \
    --gin.TRAIN_STEPS=${TRAIN_STEPS} \
    --gin.SAVING_PERIOD=${SAVING_PERIOD} \
    --gin.MODEL_DIR=\"gs://improved-t5/ckpts/t5_1_1_xl/codexglue_${CODE_LANG}_finetune\" \
    --gin.INITIAL_CHECKPOINT_PATH=\"gs://t5-data/pretrained_models/t5x/t5_1_1_xl/checkpoint_1000000\" \
    --gin.USE_CACHED_TASKS=False \
    --alsologtostderr
