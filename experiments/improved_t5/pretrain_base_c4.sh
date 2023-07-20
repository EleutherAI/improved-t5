ADDR=$1
MODEL_DIR=$2

python -m t5x.train \
    --gin_file="models/scalable_t5/t5_1_1/base.gin" \
    --gin_file="configs/task/pretrain/c4_mixed_objective.gin" \
    --gin.MIXTURE_OR_TASK_NAME=\""c4_ul2_causal_0_50"\" \
    --gin.TASK_FEATURE_LENGTHS="{'inputs':2048, 'targets': 2048}" \
    --gin.BATCH_SIZE=512 \
    --gin.TRAIN_STEPS=1000000 \
    --gin.SAVING_PERIOD=10000 \
    --gin.MODEL_DIR=\"'gs://improved-t5/ckpts/t5v2/base_c4/'\" \
    --gin.USE_CACHED_TASKS=False \
    --alsologtostderr