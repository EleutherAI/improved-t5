#!/bin/bash
MODEL_PATH=$1
MODEL=$2
LM_EVAL=$3
EXTRA=$4

export TASK="super-glue-t5-prompt"

${LM_EVAL} \
    --model hf \
    --model_args "pretrained=${MODEL_PATH}${MODEL}${EXTRA},truncation=True,max_length=512" \
    --tasks ${TASK} \
    --batch_size 8 \
    --output "OUTPUT/${TASK}/${MODEL}" \
    --log_samples
