#!/bin/bash 
PATH=$1
MODEL=$2
LM_EVAL=$3
EXTRA=$4

for TASK in bbh_cot_zeroshot bbh_cot_fewshot mmlu_flan_cot_zeroshot mmlu_flan_cot_fewshot 
do
    ${LM_EVAL} \
        --model hf \
        --model_args "pretrained=${PATH}${MODEL}${EXTRA}" \
        --tasks ${TASK} \
        --batch_size 4 \
        --output "output/${TASK}/${MODEL}" \
        --log_samples
    done
done
