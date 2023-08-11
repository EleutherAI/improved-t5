ADDR=$1
MODEL_DIR=$2

python -m t5x.train \
    --gin_file="models/scalable_t5/t5_1_1/base.gin" \
    --gin_file="configs/task/finetune/pile_mlm.gin" \
    --gin.seqio.SentencePieceVocabulary.sentencepiece_model_file=\""gs://improved-t5/vocabs/tokenizer.model"\" \
    --gin.seqio.SentencePieceVocabulary.extra_ids=100 \
    --gin.TRAIN_STEPS=1000000 \
    --gin.SAVING_PERIOD=10000 \
    --gin.network.T5Config.dtype=\"bfloat16\" \
    --gin.MODEL_DIR=\"gs://improved-t5/ckpts/v2_base_mlm/checkpoint_1000000/pile_bf16\" \
    --gin.INITIAL_CHECKPOINT_PATH=\"gs://improved-t5/ckpts/v2_base_mlm/checkpoint_1000000\" \
    --gin.USE_CACHED_TASKS=False \
    --alsologtostderr