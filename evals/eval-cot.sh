#!/bin/bash 
MODEL=$1

for TASK in bbh_flan_cot_zeroshot bbh_flan_cot_fewshot mmlu_flan_cot_zeroshot mmlu_flan_cot_fewshot 
do
    accelerate launch --no_python lm-eval \
        --model hf \
        --model_args "pretrained=${MODEL}" \
        --tasks ${TASK} \
        --batch_size 4 \
        --output "output/${TASK}/${MODEL}" \
        --log_samples
    done
done
