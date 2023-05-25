# ADDR=$1
# MODEL_DIR=$2

# python -m t5x.train \
#     --gin_file="models/scalable_t5/t5_1_1/base.gin" \
#     --gin_file="configs/task/pretrain/pile_mlm.gin" \
#     --gin_file="configs/exp/LayerNorm/pre_layernorm.gin" \
#     --gin.TRAIN_STEPS=125000 \
#     --gin.SAVING_PERIOD=25000 \
#     --gin.MODEL_DIR=\"'/fsx/lintangsutawika/improved_t5/ckpts/LayerNorm/pile_mlm_adafactor_pre/'\" \
#     --gin.USE_CACHED_TASKS=False \
#     --alsologtostderr \
#     --multiprocess_gpu \
#     --coordinator_address=${ADDR} \
#     --process_count=${SLURM_NTASKS} \
#     --process_index=${SLURM_PROCID}

# export CACHED_DATA_DIR="/fsx/lintangsutawika/data"
export CACHED_DATA_DIR="gs://improved-t5/data/"
export PREFIX="gs://improved-t5/ckpts/test-abs/"
# export PREFIX="/fsx/lintangsutawika/improved_t5/ckpts/"

ADDR=$1
MODEL_DIR=$2

python -m t5x.train \
    --gin_file="models/scalable_t5/t5_1_1/base.gin" \
    --gin_file="configs/task/pretrain/pile_mlm.gin" \
    --gin_file="configs/exp/LayerNorm/pre_layernorm.gin" \
    --gin.TRAIN_STEPS=125000 \
    --gin.SAVING_PERIOD=25000 \
    --gin.MODEL_DIR=\"${PREFIX}'LayerNorm/pile_mlm_adafactor_pre/'\" \
    --gin.USE_CACHED_TASKS=False \
    --alsologtostderr \
    --multiprocess_gpu
    # --coordinator_address=${ADDR} \
    # --process_count=${SLURM_NTASKS} \
    # --process_index=${SLURM_PROCID}
