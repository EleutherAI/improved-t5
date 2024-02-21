#!/bin/bash 
PATH=$1
MODEL=$2
LM_EVAL=$3
EXTRA=$4

TASK=flan_held_in

${LM_EVAL} \
    --model hf \
    --model_args "pretrained=${PATH}${MODEL}${EXTRA}" \
    --tasks ${TASK} \
    --batch_size 8 \
    --output "output/${TASK}/${MODEL}" \
    --log_samples
