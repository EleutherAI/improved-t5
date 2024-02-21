#!/bin/bash 
PATH=$1
MODEL=$2
LM_EVAL=$3
EXTRA=$4

${LM_EVAL} \
    --model hf \
    --model_args "pretrained=${PATH}${MODEL}${EXTRA},truncation=True,max_length=512" \
    --tasks ${TASK} \
    --batch_size 8 \
    --output "output/${TASK}/${MODEL}" \
    --log_samples
