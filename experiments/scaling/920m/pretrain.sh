ADDR=$1
MODEL_DIR=$2

python -m t5x.train \
    --gin_file="configs/task/pretrain/c4_mlm.gin" \
    --gin_file="configs/size/920m/vanilla.gin" \
    --gin_file="configs/exp/scaling.gin" \
    --gin.TRAIN_STEPS=256000 \
    --gin.SAVING_PERIOD=32000 \
    --gin.MODEL_DIR=\"'/fsx/aran/jax/ckpts/scaling/920m/'\" \
    --gin.USE_CACHED_TASKS=False \
    --alsologtostderr \
    --multiprocess_gpu \
    --coordinator_address=${ADDR} \
    --process_count=${SLURM_NTASKS} \
    --process_index=${SLURM_PROCID}
