export T5X_DIR="/fsx/lintangsutawika/t5x/"
export CONFIG_PATH="/fsx/aran/jax/t5x_2/architecture-objective/experiments/configs"
export CACHED_DATA_DIR="/fsx/lintangsutawika/data"

ADDR=$1
MODEL_DIR=$2

python -m t5x.train \
    --gin_file="configs/finetune_sglue.gin" \
    --gin_file="configs/size/110m/vanilla.gin" \
    --gin_file="${CONFIG_PATH}/mode/gpu.gin" \
    --gin.TRAIN_STEPS=208_000 \
    --gin.MODEL_DIR=\"'/fsx/aran/jax/ckpts/scaling/110m/vanilla_80k_finetune'\" \
    --gin.INITIAL_CHECKPOINT_PATH=\"'/fsx/aran/jax/ckpts/scaling/110m/checkpoint_80000'\" \
    --seqio_additional_cache_dirs="${CACHED_DATA_DIR}" \
    --alsologtostderr \
    --multiprocess_gpu \
    --coordinator_address=${ADDR} \
    --process_count=${SLURM_NTASKS} \
    --process_index=${SLURM_PROCID}