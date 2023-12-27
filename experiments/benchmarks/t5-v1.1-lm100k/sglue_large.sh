
python3 -m t5x.train \
    --gin_file="../t5x/t5x/examples/t5/t5_1_1/large.gin" \
    --gin_file="configs/task/finetune/sglue.gin" \
    --gin.TRAIN_STEPS=1_128_000 \
    --gin.SAVING_PERIOD=4000 \
    --gin.MODEL_DIR=\"gs://improved-t5/ckpts/t5_1_1_large/sglue_finetune\" \
    --gin.INITIAL_CHECKPOINT_PATH=\"gs://t5-data/pretrained_models/t5x/t5_1_1_large/checkpoint_1000000\" \
    --seqio_additional_cache_dirs=\"gs://improved-t5/data\" \
    --alsologtostderr