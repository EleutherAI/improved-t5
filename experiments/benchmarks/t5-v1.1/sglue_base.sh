MODEL_DIR=$1
CACHED_DATA_DIR=$2
USE_GPU=$3
ADDR=$4

python3 -m t5x.train \
    --gin_file="models/scalable_t5/t5_1_1/base.gin" \
    --gin_file="configs/task/finetune/sglue.gin" \
    --gin.TRAIN_STEPS=1_128_000 \
    --gin.EVAL_PERIOD=4000 \
    --gin.MODEL_DIR=\"${MODEL_DIR}'sglue_finetune'\" \
    --gin.INITIAL_CHECKPOINT_PATH=\"gs://t5-data/pretrained_models/t5x/t5_1_1_base/checkpoint_1000000\" \
    --seqio_additional_cache_dirs="${CACHED_DATA_DIR}" \
    --alsologtostderr \
    --multiprocess_gpu=${USE_GPU} \
    --coordinator_address=${ADDR} \
    --process_count=${SLURM_NTASKS} \
    --process_index=${SLURM_PROCID}