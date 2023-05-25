MODEL_DIR=$1
CACHED_DATA_DIR=$2
USE_GPU=$3
ADDR=$4

python3 -m t5x.train \
    --gin_file="/home/lintangsutawika/t5x/t5x/examples/t5/t5_1_1/large.gin" \
    --gin_file="configs/task/finetune/t0_train.gin" \
    --gin.TRAIN_STEPS=1_128_000 \
    --gin.SAVING_PERIOD=4000 \
    --gin.MODEL_DIR=\"${MODEL_DIR}'t0-train_finetune'\" \
    --gin.INITIAL_CHECKPOINT_PATH=\"gs://t5-data/pretrained_models/t5x/t5_1_1_large/checkpoint_1000000\" \
    --seqio_additional_cache_dirs="${CACHED_DATA_DIR}" \
    --alsologtostderr