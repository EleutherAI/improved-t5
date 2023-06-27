ADDR=$1
MODEL_DIR=$2

python -m t5x.train \
    --gin_file="models/scalable_t5/t5_1_1/base.gin" \
    --gin.seqio.SentencePieceVocabulary.sentencepiece_model_file=\""gs://improved-t5/vocabs/tokenizer.model"\" \
    --gin_file="configs/task/finetune/sglue.gin" \
    --gin.TRAIN_STEPS=1_128_000 \
    --gin.SAVING_PERIOD=5000 \
    --gin.MODEL_DIR=\"${MODEL_DIR}'sglue_finetune'\" \
    --gin.INITIAL_CHECKPOINT_PATH=\"gs://improved-t5/ckpts/t5v2/vanilla_base/checkpoint_1000000\" \
    --seqio_additional_cache_dirs="${CACHED_DATA_DIR}" \
    --alsologtostderr
    # --multiprocess_gpu \
    # --coordinator_address=${ADDR} \
    # --process_count=${SLURM_NTASKS} \
    # --process_index=${SLURM_PROCID}