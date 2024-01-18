ADDR=$1
MODEL_DIR=$2

python -m t5x.train \
    --gin_file="models/scalable_t5/t5_1_1/base.gin" \
    --gin.seqio.SentencePieceVocabulary.sentencepiece_model_file=\""gs://improved-t5/vocabs/tokenizer.model"\" \
    --gin.seqio.SentencePieceVocabulary.extra_ids=100 \
    --gin_file="configs/task/finetune/flan2022.gin" \
    --gin.TRAIN_STEPS=1_084_000 \
    --gin.SAVING_PERIOD=2000 \
    --gin.MODEL_DIR=\"gs://improved-t5/ckpts/v2_base_mlm/_finetune/checkpoint_1000000/flan2022_finetune\" \
    --gin.INITIAL_CHECKPOINT_PATH=\"gs://improved-t5/ckpts/v2_base_mlm/checkpoint_1000000\" \
    --gin.USE_CACHED_TASKS=False \
    --alsologtostderr
    # --multiprocess_gpu \
    # --coordinator_address=${ADDR} \
    # --process_count=${SLURM_NTASKS} \
    # --process_index=${SLURM_PROCID}
