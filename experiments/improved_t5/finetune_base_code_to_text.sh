#!/bin/bash

CODE_LANG=$1

if [[ $CODE_LANG == "python" ]]; then
    # python 251,820 -> 7869
    STEPS=7869
elif [[ $CODE_LANG == "php" ]]; then
    # php 241,241 -> 7538
    STEPS=7538
elif [[ $CODE_LANG == "go" ]]; then
    # go 167,288 -> 5227
    STEPS=5227
elif [[ $CODE_LANG == "java" ]]; then
    # java 164,923 -> 5154
    STEPS=5154
elif [[ $CODE_LANG == "javascript" ]]; then
    # javascript 58,025 -> 1813
    STEPS=1813
elif [[ $CODE_LANG == "ruby" ]]; then
    # ruby 24,927 -> 779
    STEPS=779
fi

SAVING_PERIOD=$STEPS
TRAIN_STEPS=$((${STEPS} * 10 + 1000000))

python -m t5x.train \
    --gin_file="models/scalable_t5/t5_1_1/base.gin" \
    --gin.seqio.SentencePieceVocabulary.sentencepiece_model_file=\""gs://improved-t5/vocabs/tokenizer.model"\" \
    --gin_file="configs/task/finetune/codexglue/code_to_text_${CODE_LANG}.gin" \
    --gin.TRAIN_STEPS=${TRAIN_STEPS} \
    --gin.SAVING_PERIOD=${SAVING_PERIOD} \
    --gin.MODEL_DIR=\"gs://improved-t5/ckpts/t5v2/vanilla_base/checkpoint_1000000/codexglue_${CODE_LANG}_finetune\" \
    --gin.INITIAL_CHECKPOINT_PATH=\"gs://improved-t5/ckpts/t5v2/vanilla_base/checkpoint_1000000\" \
    --seqio_additional_cache_dirs=\"gs://improved-t5/data\" \
    --alsologtostderr
