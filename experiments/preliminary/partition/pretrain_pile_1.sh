ADDR=$1
MODEL_DIR=$2

rm -rf /fsx/lintangsutawika/improved_t5/ckpts/partition_1/

python -m t5x.train \
    --gin_file="models/scalable_t5/t5_1_1/xl.gin" \
    --gin_file="configs/task/pretrain/pile_mlm.gin" \
    --gin_file="configs/exp/partition.gin" \
    --gin.NUM_PARTITIONS=1 \
    --gin.MODEL_DIR=\"'/fsx/lintangsutawika/improved_t5/ckpts/partition_1/'\" \
    --gin.USE_CACHED_TASKS=False \
    --alsologtostderr \
    --multiprocess_gpu \
    --coordinator_address=${ADDR} \
    --process_count=${SLURM_NTASKS} \
    --process_index=${SLURM_PROCID}
