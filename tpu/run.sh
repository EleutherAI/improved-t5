pkill train.py
rm -f /tmp/libtpu_lockfile

export TFDS_DATA_DIR=gs://t5x-test/data/c4/raw/raw//
# export TFDS_DATA_DIR=gs://t5x-test/tfds/
# export MODEL_DIR=gs://t5x-test/ckpts/benchmark/base_c4_pretrain/

export T5X_DIR=~/code/architecture-objective
export CONFIG_DIR=~/code/configs

# #mkdir ~/models/
# # Pre-download dataset in multi-host experiments.
# # tfds build wmt_t2t_translate ${TFDS_DATA_DIR}

cd ~/code/architecture-objective

python3 ${T5X_DIR}/t5x/train.py \
    --gin_search_paths=${T5X_DIR} \
    --gin_file="experiments/regular_t5/xl_c4.gin" \
    --gin.MODEL_DIR=\"'gs://t5x-test/ckpts/experiments/test_large_model_error'\" \
    --gin.USE_CACHED_TASKS=False \
    --tfds_data_dir="${TFDS_DATA_DIR}" \
    --alsologtostderr
