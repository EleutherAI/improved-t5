SIZE=$1
START_STEP=$2
MODEL_DIR=$3

python -m t5x.train \
    --gin_file="models/scalable_t5/t5_1_1/${SIZE}.gin" \
    --gin_file="configs/task/pretrain/pile_mlm.gin" \
    --gin.seqio.SentencePieceVocabulary.sentencepiece_model_file=\""${GCP_BUCKET}/vocabs/tokenizer.model"\" \
    --gin.seqio.SentencePieceVocabulary.extra_ids=100 \
    --gin.TRAIN_STEPS=${TRAIN_STEPS} \
    --gin.SAVING_PERIOD=10000 \
    --gin.MODEL_DIR=\"${MODEL_DIR}\" \
    --gin.USE_CACHED_TASKS=False \
    --alsologtostderr
    # --gin.Trainer.num_microbatches=2 \
    # --gin.partitioning.standard_logical_axis_rules.activation_partitioning_dims=2 \
    # --gin.partitioning.PjitPartitioner.model_parallel_submesh="(1, 1, 8, 1)" \
    # --gin.partitioning.standard_logical_axis_rules.parameter_partitioning_dims=2 \