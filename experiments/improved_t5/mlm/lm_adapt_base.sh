ADDR=$1
MODEL_DIR=$2

python -m t5x.train \
    --gin_file="models/scalable_t5/t5_1_1/base.gin" \
    --gin_file="configs/task/pretrain/pile_prefix_lm.gin" \
    --gin.seqio.SentencePieceVocabulary.sentencepiece_model_file=\""gs://improved-t5/vocabs/tokenizer.model"\" \
    --gin.seqio.SentencePieceVocabulary.extra_ids=100 \
    --gin.TRAIN_STEPS=1100000 \
    --gin.SAVING_PERIOD=10000 \
    --gin.MODEL_DIR=\"gs://improved-t5/ckpts/v2_base_mlm/checkpoint_1000000/lm_adapt\" \
    --gin.INITIAL_CHECKPOINT_PATH=\"gs://improved-t5/ckpts/v2_base_mlm/checkpoint_1000000\" \
    --gin.USE_CACHED_TASKS=False \
    --alsologtostderr
