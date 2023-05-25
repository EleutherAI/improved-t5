python -m t5x.train \
    --gin_file="models/scalable_t5/t5_1_1/base.gin" \
    --gin_file="configs/task/pretrain/pile_prefix_lm.gin" \
    --gin_file="configs/task/partition.gin" \
    --gin.BATCH_SIZE=2048 \
    --gin.EVAL_BATCH_SIZE=128 \
    --gin.TRAIN_STEPS=125000 \
    --gin.SAVING_PERIOD=25000 \
    --gin.TASK_FEATURE_LENGTHS="{'inputs': 1024, 'targets': 1024}" \
    --gin.MODEL_DIR=\"'/fsx/lintangsutawika/improved_t5/ckpts/pile_prefix_lm_no_alibi/'\" \
    --gin.USE_CACHED_TASKS=False \
    --alsologtostderr \
    --multiprocess_gpu \
    --coordinator_address=${ADDR} \
    --process_count=${SLURM_NTASKS} \
    --process_index=${SLURM_PROCID}
