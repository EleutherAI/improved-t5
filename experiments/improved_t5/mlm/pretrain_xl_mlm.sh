ADDR=$1
MODEL_DIR=$2

python -m t5x.train \
    --gin_file="models/scalable_t5/t5_1_1/xl.gin" \
    --gin_file="configs/task/pretrain/pile_mlm.gin" \
    --gin.partitioning.PjitPartitioner.num_partitions=2 \
    --gin.seqio.SentencePieceVocabulary.sentencepiece_model_file=\""gs://improved-t5/vocabs/tokenizer.model"\" \
    --gin.seqio.SentencePieceVocabulary.extra_ids=100 \
    --gin.TRAIN_STEPS=2000000 \
    --gin.SAVING_PERIOD=10000 \
    --gin.MODEL_DIR=\"'gs://improved-t5/ckpts/v2_xl_mlm/'\" \
    --gin.USE_CACHED_TASKS=False \
    --alsologtostderr
    # --gin.partitioning.PjitPartitioner.model_parallel_submesh="(1, 1, 2, 1)" \