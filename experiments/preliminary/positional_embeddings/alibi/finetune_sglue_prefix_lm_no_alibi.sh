export CACHED_DATA_DIR="/fsx/lintangsutawika/data"

ADDR=$1
MODEL_DIR=$2

python -m t5x.train \
    --gin_file="models/scalable_t5/t5_1_1/base.gin" \
    --gin_file="configs/task/finetune/sglue.gin" \
    --gin.TRAIN_STEPS=135000 \
    --gin.SAVING_PERIOD=2000 \
    --gin.BATCH_SIZE=2048 \
    --gin.MODEL_DIR=\"'/fsx/lintangsutawika/improved_t5/ckpts/sglue_finetune_no_alibi/'\" \
    --gin.INITIAL_CHECKPOINT_PATH=\"'/fsx/lintangsutawika/improved_t5/ckpts/pile_prefix_lm_no_alibi/checkpoint_125000/'\" \
    --seqio_additional_cache_dirs="${CACHED_DATA_DIR}" \
    --alsologtostderr \
    --multiprocess_gpu \
    --coordinator_address=${ADDR} \
    --process_count=${SLURM_NTASKS} \
    --process_index=${SLURM_PROCID}
