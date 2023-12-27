#!/bin/bash 
MODEL=$1
for TASK in mmlu mmlu_flan_n_shot_generative mmlu_flan_n_shot_loglikelihood
do
    for NUM in 0 5
    do
    accelerate launch --no_python lm-eval \
        --model hf \
        --model_args "pretrained=${MODEL}" \
        --tasks ${TASK} \
        --batch_size 4 \
        --output "output/${TASK}/${MODEL}-${NUM}-shot" \
        --num_fewshot $NUM \
        --log_samples
    done
done


# for TASK in mmlu_flan_cot_zeroshot mmlu_flan_cot_fewshot
# do
#     accelerate launch --no_python lm-eval \
#         --model hf \
#         --model_args "pretrained=${MODEL}" \
#         --tasks ${TASK} \
#         --batch_size 4 \
#         --output "output/${TASK}/${MODEL}"
# done
