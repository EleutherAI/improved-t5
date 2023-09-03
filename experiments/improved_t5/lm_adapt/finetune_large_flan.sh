ADDR=$1
MODEL_DIR=$2

python -m t5x.train \
    --gin_file="models/scalable_t5/t5_1_1/large.gin" \
    --gin.seqio.SentencePieceVocabulary.sentencepiece_model_file=\""gs://improved-t5/vocabs/tokenizer.model"\" \
    --gin.seqio.SentencePieceVocabulary.extra_ids=100 \
    --gin_file="configs/task/finetune/flan.gin" \
    --gin.TRAIN_STEPS=1_164_000 \
    --gin.SAVING_PERIOD=10_000 \
    --gin.MODEL_DIR=\"gs://improved-t5/ckpts/v2_large_mlm/checkpoint_1000000/lm_adapt_flan_finetune\" \
    --gin.INITIAL_CHECKPOINT_PATH=\"gs://improved-t5/ckpts/v2_large_mlm/checkpoint_1000000/lm_adapt/checkpoint_1100000\" \
    --gin.USE_CACHED_TASKS=False \
    --alsologtostderr
    # --multiprocess_gpu \
    # --coordinator_address=${ADDR} \
    # --process_count=${SLURM_NTASKS} \
    # --process_index=${SLURM_PROCID}
