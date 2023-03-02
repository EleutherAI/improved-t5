ADDR=$1
MODEL_DIR=$2

python -m t5x.train \
    --gin_file="configs/task/pretrain/pile_mlm.gin" \
    --gin_file="models/scalable_t5/t5_1_1/base.gin" \
    --gin_file="configs/exp/post_layernorm.gin" \
    --gin.DROPOUT_RATE=0.0 \
    --gin.TRAIN_STEPS=125000 \
    --gin.SAVING_PERIOD=25000 \
    --gin.MODEL_DIR=\"'/fsx/lintangsutawika/improved_t5/ckpts/post_layernorm_base/'\" \
    --gin.USE_CACHED_TASKS=False \
    --alsologtostderr \
    --multiprocess_gpu \
    --coordinator_address=${ADDR} \
    --process_count=${SLURM_NTASKS} \
    --process_index=${SLURM_PROCID}
