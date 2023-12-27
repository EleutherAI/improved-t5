#!/bin/bash 
export PATH="lintang"

export TASK="super-glue-t5-prompt"
for SIZE in base large xl
do
    for VERSION in "t5-v2" "t5-v1_1"
    do
        export MODEL="${MODEL_PATH}/${VERSION}-${SIZE}-sglue/"
        echo $MODEL
        accelerate launch --no_python lm-eval \
            --model hf \
            --model_args "pretrained=${MODEL}" \
            --tasks ${TASK} \
            --batch_size 4 \
            --output "output/${TASK}/${MODEL}" \
            --log_samples
    done
done
