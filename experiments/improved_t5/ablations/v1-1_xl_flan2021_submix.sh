ADDR=$1
MODEL_DIR=$2

python -m t5x.train \
    --gin_file="../t5x/t5x/examples/t5/t5_1_1/xl.gin" \
    --gin.partitioning.PjitPartitioner.model_parallel_submesh="(1, 1, 2, 1)" \
    --gin_file="configs/task/finetune/flan2021_t5.gin" \
    --gin.MIXTURE_OR_TASK_NAME=\"flan2021_submix_original_t5\" \
    --gin.TRAIN_STEPS=1_038_000 \
    --gin.SAVING_PERIOD=2_000 \
    --gin.INITIAL_CHECKPOINT_PATH=\"gs://t5-data/pretrained_models/t5x/t5_1_1_xl/checkpoint_1000000\" \
    --gin.MODEL_DIR=\"gs://improved-t5/ckpts/ablations/v1_1_xl_flan2021_submix\" \
    --gin.USE_CACHED_TASKS=False \
    --alsologtostderr
    # --multiprocess_gpu \
    # --coordinator_address=${ADDR} \
    # --process_count=${SLURM_NTASKS} \
    # --process_index=${SLURM_PROCID}
