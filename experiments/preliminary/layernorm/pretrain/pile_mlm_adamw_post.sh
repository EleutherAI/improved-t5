ADDR=$1
MODEL_DIR=$2

export CACHED_DATA_DIR="/fsx/lintangsutawika/data"
export PREFIX="/fsx/lintangsutawika/improved_t5/ckpts/LayerNorm/with_abs_pos/"

python -m t5x.train \
    --gin_file="models/scalable_t5/t5_1_1/base.gin" \
    --gin_file="configs/task/pretrain/pile_mlm.gin" \
    --gin_file="configs/exp/LayerNorm/training.gin" \
    --gin_file="configs/exp/LayerNorm/post_layernorm.gin" \
    --gin.TRAIN_STEPS=125000 \
    --gin.SAVING_PERIOD=25000 \
    --gin.MODEL_DIR=\"${PREFIX}'pile_mlm_adamw_post/'\" \
    --gin.USE_CACHED_TASKS=False \
    --alsologtostderr \
    --multiprocess_gpu \
    --coordinator_address=${ADDR} \
    --process_count=${SLURM_NTASKS} \
    --process_index=${SLURM_PROCID}
