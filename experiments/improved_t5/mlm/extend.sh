SIZE=$1
START_STEP=$2
EXTENSION=$3
EXTEND_FROM=$4
EXTEND_TO=$5
NUM_PARTITIONS=$6
NUM_MICROBATCHES=$7

if [[ $NUM_PARTITIONS == "" ]]; then
    NUM_PARTITIONS=2
fi

if [[ $NUM_MICROBATCHES == "" ]]; then
    NUM_MICROBATCHES=8
fi

TRAIN_STEPS=$(( ${START_STEP} + 5000 ))

python -m t5x.train \
    --gin_file="models/scalable_t5/t5_1_1/${SIZE}.gin" \
    --gin_file="configs/task/finetune/extend_${EXTENSION}.gin" \
    --gin.seqio.SentencePieceVocabulary.sentencepiece_model_file=\""${GCP_BUCKET}/vocabs/tokenizer.model"\" \
    --gin.seqio.SentencePieceVocabulary.extra_ids=100 \
    --gin.TRAIN_STEPS=${TRAIN_STEPS} \
    --gin.SAVING_PERIOD=1000 \
    --gin.INITIAL_CHECKPOINT_PATH=\"${EXTEND_FROM}\" \
    --gin.MODEL_DIR=\"${EXTEND_TO}\" \
    --gin.Trainer.num_microbatches=${NUM_MICROBATCHES} \
    --gin.partitioning.PjitPartitioner.num_partitions=${NUM_PARTITIONS} \
    --gin.USE_CACHED_TASKS=False \
    --alsologtostderr
