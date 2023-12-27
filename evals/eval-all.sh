#!/bin/bash 

# export MODEL_PATH="checkpoints/hf/"
export MODEL_PATH="/fsx/lintangsutawika/00-improved-t5/checkpoints/hf/"
# export PATH="lintang"

ALL_MODEL=(
    # "lintang/t5-v2-base-flan"
    # "lintang/t5-v2-large-flan"
    # "lintang/t5-v2-xl-flan"
    # "lintang/t5-v1_1-base-flan"
    # "lintang/t5-v1_1-large-flan"
    # "lintang/t5-v1_1-xl-flan"
    "google/flan-t5-base"
    # "google/flan-t5-large"
    # "google/flan-t5-xl"
    # "${MODEL_PATH}t5-v1_1-xl-flan2021_submix"
    # "${MODEL_PATH}t5-v2-xl-flan2021_submix"
    # "${MODEL_PATH}t5-v2-xl-ul2c-flan2021"
    # "${MODEL_PATH}t5-v2-xl-flan2022-76k"
    # "${MODEL_PATH}t5-v2-xl-flan2022-36k"
    # "${MODEL_PATH}t5-v2-xl-flan2022-38k"
    "${MODEL_PATH}t5-v2-base-2M-flan"
)

for MODEL in ${ALL_MODEL[@]}; do
    echo $MODEL
    echo "BBH"
    bash eval-bbh.sh ${MODEL}
    echo "MMLU"
    bash eval-mmlu.sh ${MODEL}
    echo "Held In"
    bash eval-held_in.sh ${MODEL}
    echo "CoT"
    bash eval-cot.sh ${MODEL}
done
