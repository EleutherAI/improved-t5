SIZE=$1
STEP=$2
INIT_DIR=$3
MODEL_DIR=$4

python -m t5x.train \
    --gin_file="models/scalable_t5/t5_1_1/${SIZE}.gin" \
    --gin.seqio.SentencePieceVocabulary.sentencepiece_model_file=\""${GCP_BUCKET}/vocabs/tokenizer.model"\" \
    --gin.seqio.SentencePieceVocabulary.extra_ids=100 \
    --gin_file="configs/task/finetune/flan2022.gin" \
    --gin.TRAIN_STEPS=${STEP} \
    --gin.SAVING_PERIOD=2000 \
    --gin.INITIAL_CHECKPOINT_PATH=\"${INIT_DIR}\" \
    --gin.MODEL_DIR=\"${MODEL_DIR}\" \
    --gin.USE_CACHED_TASKS=False \
    --alsologtostderr
    # --multiprocess_gpu \
    # --coordinator_address=${ADDR} \
    # --process_count=${SLURM_NTASKS} \
    # --process_index=${SLURM_PROCID}