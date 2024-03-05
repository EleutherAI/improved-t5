SIZE=$1
CHECKPOINT_PATH="/scratch/lintang"
# BUCKET="gs://t5v2/v2_${SIZE}/_finetune/checkpoint_2000000"
BUCKET="gs://improved-t5/ckpts/v2_${SIZE}_mlm/_finetune/checkpoint_2000000"
HF_PATH="${CHECKPOINT_PATH}/hf/pile-t5-${SIZE}-codexglue"
T5X_PATH="${CHECKPOINT_PATH}/t5x/pile-t5-${SIZE}-codexglue"

mkdir -p "${HF_PATH}"
git clone https://huggingface.co/lintang/pile-t5-${SIZE}-codexglue "${HF_PATH}"
git -C "${HF_PATH}" remote set-url origin "https://lintang:${HF_KEY}@huggingface.co/lintang/pile-t5-${SIZE}-codexglue"

#################################################
# Go
#################################################
GCP_BUCKET="${BUCKET}/codexglue_go_finetune/checkpoint_2052270"

# Switch branch
git -C "${HF_PATH}" checkout -b "go"
git -C "${HF_PATH}" config http.postBuffer 524288000

mkdir -p "${T5X_PATH}"
gsutil -m cp -r "${GCP_BUCKET}/*" $T5X_PATH
bash scripts/convert_v2.sh ${SIZE} $T5X_PATH $HF_PATH

git -C "${HF_PATH}" add .
git -C "${HF_PATH}" commit -am "add files for finetuning on go"
git -C "${HF_PATH}" push origin "go"
git -C "${HF_PATH}" checkout main
rm -rf "${T5X_PATH}"

#################################################
# Java
#################################################
GCP_BUCKET="${BUCKET}/codexglue_java_finetune/checkpoint_2051540"

# Switch branch
git -C "${HF_PATH}" checkout -b "java"
git -C "${HF_PATH}" config http.postBuffer 524288000

mkdir -p "${T5X_PATH}"
gsutil -m cp -r "${GCP_BUCKET}/*" $T5X_PATH
bash scripts/convert_v2.sh ${SIZE} $T5X_PATH $HF_PATH

git -C "${HF_PATH}" add .
git -C "${HF_PATH}" commit -am "add files for finetuning on java"
git -C "${HF_PATH}" push origin "java"
git -C "${HF_PATH}" checkout main
rm -rf "${T5X_PATH}"

#################################################
# Javascript
#################################################
GCP_BUCKET="${BUCKET}/codexglue_javascript_finetune/checkpoint_2018130"

# Switch branch
git -C "${HF_PATH}" checkout -b "javascript"
git -C "${HF_PATH}" config http.postBuffer 524288000

mkdir -p "${T5X_PATH}"
gsutil -m cp -r "${GCP_BUCKET}/*" $T5X_PATH
bash scripts/convert_v2.sh ${SIZE} $T5X_PATH $HF_PATH

git -C "${HF_PATH}" add .
git -C "${HF_PATH}" commit -am "add files for finetuning on javascript"
git -C "${HF_PATH}" push origin "javascript"
git -C "${HF_PATH}" checkout main
rm -rf "${T5X_PATH}"

#################################################
# PHP
#################################################
GCP_BUCKET="${BUCKET}/codexglue_php_finetune/checkpoint_2075380"

# Switch branch
git -C "${HF_PATH}" checkout -b "php"
git -C "${HF_PATH}" config http.postBuffer 524288000

mkdir -p "${T5X_PATH}"
gsutil -m cp -r "${GCP_BUCKET}/*" $T5X_PATH
bash scripts/convert_v2.sh ${SIZE} $T5X_PATH $HF_PATH

git -C "${HF_PATH}" add .
git -C "${HF_PATH}" commit -am "add files for finetuning on php"
git -C "${HF_PATH}" push origin "php"
git -C "${HF_PATH}" checkout main
rm -rf "${T5X_PATH}"

#################################################
# Python
#################################################
GCP_BUCKET="${BUCKET}/codexglue_python_finetune/checkpoint_2078690"

# Switch branch
git -C "${HF_PATH}" checkout -b "python"
git -C "${HF_PATH}" config http.postBuffer 524288000

mkdir -p "${T5X_PATH}"
gsutil -m cp -r "${GCP_BUCKET}/*" $T5X_PATH
bash scripts/convert_v2.sh ${SIZE} $T5X_PATH $HF_PATH

git -C "${HF_PATH}" add .
git -C "${HF_PATH}" commit -am "add files for finetuning on python"
git -C "${HF_PATH}" push origin "python"
git -C "${HF_PATH}" checkout main
rm -rf "${T5X_PATH}"

#################################################
# Ruby
#################################################
GCP_BUCKET="${BUCKET}/codexglue_ruby_finetune/checkpoint_2007790"

# Switch branch
git -C "${HF_PATH}" checkout -b "ruby"
git -C "${HF_PATH}" config http.postBuffer 524288000

mkdir -p "${T5X_PATH}"
gsutil -m cp -r "${GCP_BUCKET}/*" $T5X_PATH
bash scripts/convert_v2.sh ${SIZE} $T5X_PATH $HF_PATH

git -C "${HF_PATH}" add .
git -C "${HF_PATH}" commit -am "add files for finetuning on ruby"
git -C "${HF_PATH}" push origin "ruby"
git -C "${HF_PATH}" checkout main
rm -rf "${T5X_PATH}"
