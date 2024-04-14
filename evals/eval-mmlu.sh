#!/bin/bash
MODEL_PATH=$1
MODEL=$2
LM_EVAL=$3
EXTRA=$4

for TASK in mmlu_generative mmlu_flan_n_shot_generative
do
    for NUM in 0 5
    do
    ${LM_EVAL} \
        --model hf \
        --model_args "pretrained=${MODEL_PATH}${MODEL}${EXTRA}" \
        --tasks ${TASK} \
        --batch_size 8 \
        --output "OUTPUT/${TASK}/${MODEL}-${NUM}-shot" \
        --num_fewshot $NUM \
        --log_samples
    done
done

for TASK in mmlu mmlu_flan_n_shot_loglikelihood
do
    for NUM in 0 5
    do
    ${LM_EVAL} \
        --model hf \
        --model_args "pretrained=${MODEL_PATH}${MODEL}${EXTRA}" \
        --tasks ${TASK} \
        --batch_size 8 \
        --output "OUTPUT/${TASK}/${MODEL}-${NUM}-shot" \
        --num_fewshot $NUM \
        --log_samples
    done
done
