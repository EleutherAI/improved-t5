
python3 -m t5x.train \
    --gin_file="../t5x/t5x/examples/t5/t5_1_1/xl.gin" \
    --gin_file="configs/task/finetune/sglue_t5.gin" \
    --gin.TRAIN_STEPS=1_262_144 \
    --gin.SAVING_PERIOD=5000 \
    --gin.MODEL_DIR=\"gs://improved-t5/ckpts/t5_1_1_xl/sglue_finetune\" \
    --gin.INITIAL_CHECKPOINT_PATH=\"gs://t5-data/pretrained_models/t5x/t5_1_1_xl/checkpoint_1000000\" \
    --gin.USE_CACHED_TASKS=False \
    --alsologtostderr