ADDR=$1
MODEL_DIR=$2

python -m t5x.train \
    --gin_file="models/scalable_t5/t5_1_1/xl.gin" \
    --gin_file="configs/task/finetune/extend_2048.gin" \
    --gin.seqio.SentencePieceVocabulary.sentencepiece_model_file=\""gs://improved-t5/vocabs/tokenizer.model"\" \
    --gin.seqio.SentencePieceVocabulary.extra_ids=100 \
    --gin.TRAIN_STEPS=1000 \
    --gin.SAVING_PERIOD=1000 \
    --gin.INITIAL_CHECKPOINT_PATH=\"'gs://improved-t5/ckpts/v2_xl_mlm/checkpoint_1000000'\" \
    --gin.MODEL_DIR=\"'gs://improved-t5/ckpts/v2_xl_mlm/_extend_2048'\" \
    --gin.USE_CACHED_TASKS=False \
    --alsologtostderr
    # --gin.partitioning.PjitPartitioner.model_parallel_submesh="(1, 1, 2, 1)" \