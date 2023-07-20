ADDR=$1
MODEL_DIR=$2

python -m t5x.train \
    --gin_file="models/scalable_t5/t5_1_1/xl.gin" \
    --gin_file="configs/task/pretrain/pile_mixed_objective.gin" \
    --gin.seqio.SentencePieceVocabulary.sentencepiece_model_file=\""gs://improved-t5/vocabs/tokenizer.model"\" \
    --gin.seqio.SentencePieceVocabulary.extra_ids=100 \
    --gin.MIXTURE_OR_TASK_NAME=\""pile_ul2_causal_0_50"\" \
    --gin.TRAIN_STEPS=1000000 \
    --gin.SAVING_PERIOD=10000 \
    --gin.MODEL_DIR=\"'gs://improved-t5/ckpts/v2_xl/'\" \
    --gin.USE_CACHED_TASKS=False \
    --alsologtostderr
