export MODEL_PATH="lintang"
export TASK="super-glue-t5-prompt"

for SIZE in base large xl xxl
do
    for VERSION in "t5-v2" "t5-v1_1"
    do
        export MODEL="${MODEL_PATH}/${VERSION}-${SIZE}-sglue"
        echo $MODEL
        # accelerate launch --no_python lm-eval \
        lm-eval \
            --model hf \
            --model_args "pretrained=${MODEL},truncation=True,max_length=512" \
            --tasks ${TASK} \
            --batch_size 4 \
            --output "output/${TASK}/${MODEL}" \
            --log_samples
    done
done
