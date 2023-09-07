ADDR=$1
MODEL_DIR=$2

python -m t5x.train \
    --gin_file="../t5x/t5x/examples/t5/t5_1_1/xxl.gin" \
    --gin_file="configs/task/finetune/flan2021_t5.gin" \
    --gin.partitioning.PjitPartitioner.model_parallel_submesh="(1, 1, 2, 1)" \
    --gin.TRAIN_STEPS=1_014_000 \
    --gin.SAVING_PERIOD=10_000 \
    --gin.MODEL_DIR=\"gs://improved-t5/ckpts/t5_1_1_xxl/flan2021_finetune\" \
    --gin.INITIAL_CHECKPOINT_PATH=\"gs://t5-data/pretrained_models/t5x/t5_1_1_xxl/checkpoint_1000000\" \
    --gin.USE_CACHED_TASKS=False \
    --alsologtostderr
    # --multiprocess_gpu \
    # --coordinator_address=${ADDR} \
    # --process_count=${SLURM_NTASKS} \
    # --process_index=${SLURM_PROCID}