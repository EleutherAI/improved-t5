#!/bin/bash 
MODEL=$1

export TASK=flan_held_in
accelerate launch --no_python lm-eval \
    --model hf \
    --model_args "pretrained=${MODEL}" \
    --tasks ${TASK} \
    --batch_size 4 \
    --output "output/${TASK}/${MODEL}" \
    --log_samples
