ADDR=$1
MODEL_DIR=$2

python -m t5x.train \
    --gin_file="models/scalable_t5/t5_1_1/base.gin" \
    --gin.seqio.SentencePieceVocabulary.sentencepiece_model_file=\""gs://improved-t5/vocabs/tokenizer.model"\" \
    --gin.seqio.SentencePieceVocabulary.extra_ids=100 \
    --gin_file="configs/task/finetune/sglue.gin" \
    --gin.TRAIN_STEPS=1_128_000 \
    --gin.SAVING_PERIOD=5000 \
    --gin.MODEL_DIR=\"gs://improved-t5/ckpts/v2_base/checkpoint_1000000/sglue_finetune\" \
    --gin.INITIAL_CHECKPOINT_PATH=\"gs://improved-t5/ckpts/v2_base/checkpoint_1000000\" \
    --seqio_additional_cache_dirs=\"gs://improved-t5/data\" \
    --alsologtostderr
    # --multiprocess_gpu \
    # --coordinator_address=${ADDR} \
    # --process_count=${SLURM_NTASKS} \
    # --process_index=${SLURM_PROCID}
