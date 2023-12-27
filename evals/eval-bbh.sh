#!/bin/bash 
MODEL=$1

for TASK in bbh_flan_zeroshot bbh_flan_fewshot
do
    accelerate launch --no_python lm-eval \
        --model hf \
        --model_args "pretrained=${MODEL}" \
        --tasks ${TASK} \
        --batch_size 4 \
        --output "output/${TASK}/${MODEL}" \
        --log_samples
done
