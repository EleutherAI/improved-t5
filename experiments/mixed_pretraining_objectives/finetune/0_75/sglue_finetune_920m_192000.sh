export CACHED_DATA_DIR="/fsx/lintangsutawika/data"

ADDR=$1
MODEL_DIR=$2

let "TRAIN_STEPS = 192000 + 128000"

python -m t5x.train \
    --gin_file="models/scalable_t5/t5_1_1/base.gin" \
    --gin_file="configs/size/920m/vanilla.gin" \
    --gin_file="configs/task/finetune/sglue.gin" \
    --gin.TRAIN_STEPS=${TRAIN_STEPS} \
    --gin.SAVING_PERIOD=2_000 \
    --gin.MODEL_DIR=\"'/fsx/lintangsutawika/improved_t5/ckpts/920m_pile_ul2_causal_0_75/sglue_192k/'\" \
    --gin.INITIAL_CHECKPOINT_PATH=\"'/fsx/lintangsutawika/improved_t5/ckpts/920m_pile_ul2_causal_0_75/checkpoint_192000/'\" \
    --seqio_additional_cache_dirs="${CACHED_DATA_DIR}" \
    --alsologtostderr \
    --multiprocess_gpu \
    --coordinator_address=${ADDR} \
    --process_count=${SLURM_NTASKS} \
    --process_index=${SLURM_PROCID}