#!/bin/bash 
MODEL=$1

# for TASK in bbh_zeroshot bbh_fewshot
for TASK in bbh_fewshot
do
    # accelerate launch --no_python lm-eval \
    lm-eval \
        --model hf \
        --model_args "pretrained=${MODEL},parallelize=True" \
        --tasks ${TASK} \
        --batch_size 1 \
        --output "output/${TASK}/${MODEL}" \
        --log_samples
done
