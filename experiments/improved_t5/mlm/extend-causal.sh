SIZE=$1
TRAIN_STEPS=$2
EXTENSION=$3
EXTEND_FROM=$4
EXTEND_TO=$5
NUM_PARTITIONS=$6
NUM_MICROBATCHES=$7
CAUSAL=$8

if [[ $NUM_PARTITIONS == "" ]]; then
    NUM_PARTITIONS=2
fi

if [[ $NUM_MICROBATCHES == "" ]]; then
    NUM_MICROBATCHES=8
fi

python -m t5x.train \
    --gin_file="models/scalable_t5/t5_1_1/${SIZE}.gin" \
    --gin_file="configs/task/finetune/extend_${EXTENSION}-causal.gin" \
    --gin.seqio.SentencePieceVocabulary.sentencepiece_model_file=\""${GCP_BUCKET}/vocabs/tokenizer.model"\" \
    --gin.seqio.SentencePieceVocabulary.extra_ids=100 \
    --gin.TRAIN_STEPS=${TRAIN_STEPS} \
    --gin.SAVING_PERIOD=1000 \
    --gin.train.use_orbax=False \
    --gin.train.infer_eval_dataset_cfg=None \
    --gin.INITIAL_CHECKPOINT_PATH=\"${EXTEND_FROM}\" \
    --gin.MODEL_DIR=\"${EXTEND_TO}\" \
    --gin.Trainer.num_microbatches=${NUM_MICROBATCHES} \
    --gin.partitioning.PjitPartitioner.num_partitions=${NUM_PARTITIONS} \
    --gin.USE_CACHED_TASKS=False \
    --alsologtostderr
