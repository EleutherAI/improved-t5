export CACHED_DATA_DIR="/fsx/lintangsutawika/data"
export PREFIX="/fsx/lintangsutawika/improved_t5/ckpts/LayerNorm/with_abs_pos/"

ADDR=$1
MODEL_DIR=$2

python -m t5x.train \
    --gin_file="models/scalable_t5/t5_1_1/base.gin" \
    --gin_file="configs/task/eval/t0_eval.gin" \
    --gin_file="configs/exp/PositionEmbedding/rotary.gin" \
    --gin_file="configs/exp/PositionEmbedding/relpos.gin" \
    --gin_file="configs/exp/LayerNorm/post_layernorm.gin" \
    --gin_file="configs/exp/LayerNorm/reset_optim.gin" \
    --gin.TRAIN_STEPS=135000 \
    --gin.SAVING_PERIOD=5000 \
    --gin.MODEL_DIR=\"${PREFIX}'pile_mlm_adamw_post_rotary/finetune_t0_eval/'\" \
    --gin.INITIAL_CHECKPOINT_PATH=\"${PREFIX}'pile_mlm_adamw_post_rotary/finetune_t0_train/checkpoint_135000/'\" \
    --seqio_additional_cache_dirs="${CACHED_DATA_DIR}" \
    --gin.USE_CACHED_TASKS=True \
    --alsologtostderr \
    --multiprocess_gpu \
    --coordinator_address=${ADDR} \
    --process_count=${SLURM_NTASKS} \
    --process_index=${SLURM_PROCID}