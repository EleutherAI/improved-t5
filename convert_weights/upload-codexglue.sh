SIZE=$1

# Go
GCP_BUCKET="gs://improved-t5/ckpts/v2_${SIZE}_mlm/_finetune/checkpoint_1000000/codexglue_go_finetune/checkpoint_1052270"
CHECKPOINT_PATH="/weka/lintangsutawika/01-t5v2/ckpts"
HF_PATH="${CHECKPOINT_PATH}/hf/_t5-v2-${SIZE}-codexglue-go"
mkdir -p "${HF_PATH}"
T5X_PATH="${CHECKPOINT_PATH}/t5x/t5-v2-${SIZE}-codexglue-go"
mkdir -p "${T5X_PATH}"
gsutil -m cp -r "${GCP_BUCKET}/*" $T5X_PATH
bash scripts/convert_v2.sh ${SIZE} $T5X_PATH $HF_PATH
rm -rf "${T5X_PATH}"

# Java
GCP_BUCKET="gs://improved-t5/ckpts/v2_${SIZE}_mlm/_finetune/checkpoint_1000000/codexglue_java_finetune/checkpoint_1051540"
CHECKPOINT_PATH="/weka/lintangsutawika/01-t5v2/ckpts"
HF_PATH="${CHECKPOINT_PATH}/hf/_t5-v2-${SIZE}-codexglue-java"
mkdir -p "${HF_PATH}"
T5X_PATH="${CHECKPOINT_PATH}/t5x/t5-v2-${SIZE}-codexglue-java"
mkdir -p "${T5X_PATH}"
gsutil -m cp -r "${GCP_BUCKET}/*" $T5X_PATH
bash scripts/convert_v2.sh ${SIZE} $T5X_PATH $HF_PATH
rm -rf "${T5X_PATH}"

# Javascript
GCP_BUCKET="gs://improved-t5/ckpts/v2_${SIZE}_mlm/_finetune/checkpoint_1000000/codexglue_javascript_finetune/checkpoint_1018130"
CHECKPOINT_PATH="/weka/lintangsutawika/01-t5v2/ckpts"
HF_PATH="${CHECKPOINT_PATH}/hf/_t5-v2-${SIZE}-codexglue-javascript"
mkdir -p "${HF_PATH}"
T5X_PATH="${CHECKPOINT_PATH}/t5x/t5-v2-${SIZE}-codexglue-javascript"
mkdir -p "${T5X_PATH}"
gsutil -m cp -r "${GCP_BUCKET}/*" $T5X_PATH
bash scripts/convert_v2.sh ${SIZE} $T5X_PATH $HF_PATH
rm -rf "${T5X_PATH}"

# PHP
GCP_BUCKET="gs://improved-t5/ckpts/v2_${SIZE}_mlm/_finetune/checkpoint_1000000/codexglue_php_finetune/checkpoint_1075380"
CHECKPOINT_PATH="/weka/lintangsutawika/01-t5v2/ckpts"
HF_PATH="${CHECKPOINT_PATH}/hf/_t5-v2-${SIZE}-codexglue-php"
mkdir -p "${HF_PATH}"
T5X_PATH="${CHECKPOINT_PATH}/t5x/t5-v2-${SIZE}-codexglue-php"
mkdir -p "${T5X_PATH}"
gsutil -m cp -r "${GCP_BUCKET}/*" $T5X_PATH
bash scripts/convert_v2.sh ${SIZE} $T5X_PATH $HF_PATH
rm -rf "${T5X_PATH}"

# Python
GCP_BUCKET="gs://improved-t5/ckpts/v2_${SIZE}_mlm/_finetune/checkpoint_1000000/codexglue_python_finetune/checkpoint_1078690"
CHECKPOINT_PATH="/weka/lintangsutawika/01-t5v2/ckpts"
HF_PATH="${CHECKPOINT_PATH}/hf/_t5-v2-${SIZE}-codexglue-python"
mkdir -p "${HF_PATH}"
T5X_PATH="${CHECKPOINT_PATH}/t5x/t5-v2-${SIZE}-codexglue-python"
mkdir -p "${T5X_PATH}"
gsutil -m cp -r "${GCP_BUCKET}/*" $T5X_PATH
bash scripts/convert_v2.sh ${SIZE} $T5X_PATH $HF_PATH
rm -rf "${T5X_PATH}"

# Ruby
GCP_BUCKET="gs://improved-t5/ckpts/v2_${SIZE}_mlm/_finetune/checkpoint_1000000/codexglue_ruby_finetune/checkpoint_1007790"
CHECKPOINT_PATH="/weka/lintangsutawika/01-t5v2/ckpts"
HF_PATH="${CHECKPOINT_PATH}/hf/_t5-v2-${SIZE}-codexglue-ruby"
mkdir -p "${HF_PATH}"
T5X_PATH="${CHECKPOINT_PATH}/t5x/t5-v2-${SIZE}-codexglue-ruby"
mkdir -p "${T5X_PATH}"
gsutil -m cp -r "${GCP_BUCKET}/*" $T5X_PATH
bash scripts/convert_v2.sh ${SIZE} $T5X_PATH $HF_PATH
rm -rf "${T5X_PATH}"

