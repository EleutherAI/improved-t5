MODEL_PATH=$1
MODEL=$2
LM_EVAL=$3
EXTRA=$4

for LANG in go java php python ruby javascript; do
    ${LM_EVAL} \
        --model hf \
        --model_args "pretrained=${MODEL_PATH}${MODEL}${EXTRA},revision=${LANG}" \
        --tasks "code2text_${LANG}" \
        --batch_size 8 \
        --output "output/codexglue_code2text/${MODEL}/${LANG}/" \
        --log_samples
done
