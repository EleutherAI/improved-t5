#ROOT_PATH="/mnt/ssd-2/lintangsutawika/00-improved-t5/checkpoints/hf"
export ROOT_PATH="checkpoints/hf"

ALL_MODEL=(
    # "t5-v2-base"
    # "t5-v2-large"
    # "t5-v2-xl"
    "t5-v1_1-base"
    "t5-v1_1-large"
    "t5-v1_1-xl"
)

for MODEL in ${ALL_MODEL[@]}; do
    for LANG in go java php python ruby javascript; do
        accelerate launch --no_python \
            lm-eval --model hf \
            --model_args "pretrained=${ROOT_PATH}/${MODEL}-codexglue-${LANG}" \
            --tasks "code2text_${LANG}" \
            --batch_size 1 \
            --output "output/codexglue_code2text/${MODEL}/${LANG}/" \
            --log_samples
    done
done
