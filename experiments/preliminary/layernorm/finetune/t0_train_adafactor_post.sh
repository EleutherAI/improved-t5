# export CACHED_DATA_DIR="/fsx/lintangsutawika/data"
export CACHED_DATA_DIR="gs://improved-t5/data"

export PREFIX="gs://improved-t5/ckpts/"
export PREFIX="/fsx/lintangsutawika/improved_t5/ckpts/"

ADDR=$1
MODEL_DIR=$2

python3 -m t5x.train \
    --gin_file="models/scalable_t5/t5_1_1/base.gin" \
    --gin_file="configs/task/finetune/t0_train.gin" \
    --gin_file="configs/exp/LayerNorm/post_layernorm.gin" \
    --gin_file="configs/exp/LayerNorm/reset_optim.gin" \
    --gin.TRAIN_STEPS=135000 \
    --gin.SAVING_PERIOD=2000 \
    --gin.EVAL_PERIOD=10000 \
    --gin.MODEL_DIR=\"'${PREFIX}LayerNorm/pile_mlm_adafactor_post/finetune_t0_train/'\" \
    --gin.INITIAL_CHECKPOINT_PATH=\"'${PREFIX}LayerNorm/pile_mlm_adafactor_post/checkpoint_125000/'\" \
    --seqio_additional_cache_dirs="${CACHED_DATA_DIR}" \
    --gin.USE_CACHED_TASKS=True \
    --alsologtostderr
    #  \
    # --multiprocess_gpu \
    # --coordinator_address=${ADDR} \
    # --process_count=${SLURM_NTASKS} \
    # --process_index=${SLURM_PROCID}