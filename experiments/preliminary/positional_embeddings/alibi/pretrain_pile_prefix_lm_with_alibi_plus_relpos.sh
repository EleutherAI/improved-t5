ADDR=$1
MODEL_DIR=$2

python -m t5x.train \
    --gin_file="models/scalable_t5/t5_1_1/base.gin" \
    --gin_file="configs/task/pretrain/pile_prefix_lm.gin" \
    --gin_file="configs/exp/alibi_plus_relpos.gin" \
    --gin.TRAIN_STEPS=125000 \
    --gin.SAVING_PERIOD=5000 \
    --gin.MODEL_DIR=\"'/fsx/lintangsutawika/improved_t5/ckpts/pile_prefix_lm_with_alibi_plus_relpos/'\" \
    --gin.USE_CACHED_TASKS=False \
    --alsologtostderr \
    --multiprocess_gpu \
    --coordinator_address=${ADDR} \
    --process_count=${SLURM_NTASKS} \
    --process_index=${SLURM_PROCID}
