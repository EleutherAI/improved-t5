#!/bin/bash

SIZE=$1
START_STEP=$2
CODE_LANG=$3
INIT_DIR=$4
MODEL_DIR=$5

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
TRAIN_STEPS=$(( ${STEPS} * 10 + ${START_STEP} ))

python -m t5x.train \
    --gin_file="models/scalable_t5/t5_1_1/${SIZE}.gin" \
    --gin.seqio.SentencePieceVocabulary.sentencepiece_model_file=\""${GCP_BUCKET}/vocabs/tokenizer.model"\" \
    --gin.seqio.SentencePieceVocabulary.extra_ids=100 \
    --gin_file="configs/task/finetune/codexglue/code_to_text_${CODE_LANG}.gin" \
    --gin.train.use_orbax=False \
    --gin.TRAIN_STEPS=${TRAIN_STEPS} \
    --gin.SAVING_PERIOD=${SAVING_PERIOD} \
    --gin.INITIAL_CHECKPOINT_PATH=\"${INIT_DIR}\" \
    --gin.MODEL_DIR=\"${MODEL_DIR}\" \
    --seqio_additional_cache_dirs=\"${GCP_BUCKET}/data\" \
    --gin.Trainer.num_microbatches=32 \
    --gin.partitioning.PjitPartitioner.num_partitions=8 \
    --alsologtostderr
    # --gin.partitioning.PjitPartitioner.model_parallel_submesh="(1, 1, 8, 1)" \
