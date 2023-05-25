export CACHED_DATA_DIR="/fsx/lintangsutawika/data"
export PREFIX="/fsx/lintangsutawika/improved_t5/ckpts/LayerNorm/with_abs_pos/"

ADDR=$1
MODEL_DIR=$2

python -m t5x.train \
    --gin_file="models/scalable_t5/t5_1_1/base.gin" \
    --gin_file="configs/task/finetune/t0_train.gin" \
    --gin_file="configs/exp/LayerNorm/reset_optim.gin" \
    --gin.TRAIN_STEPS=1110000 \
    --gin.SAVING_PERIOD=2000 \
    --gin.EVAL_PERIOD=10000 \
    --gin.MODEL_DIR=\"${PREFIX}'base_lm100k/finetune_t0_train/'\" \
    --gin.INITIAL_CHECKPOINT_PATH=\"'gs://t5-data/pretrained_models/t5x/t5_1_1_lm100k_base/checkpoint_1100000/'\" \
    --seqio_additional_cache_dirs="${CACHED_DATA_DIR}" \
    --gin.USE_CACHED_TASKS=True \
    --alsologtostderr \
    --multiprocess_gpu \
    --coordinator_address=${ADDR} \
    --process_count=${SLURM_NTASKS} \
    --process_index=${SLURM_PROCID}

