SIZE=$1
START=$2
END=$3
GCP_BUCKET="gs://improved-t5/ckpts/v2_${SIZE}_mlm"
CHECKPOINT_PATH="/weka/lintangsutawika/01-t5v2/ckpts"

HF_PATH="${CHECKPOINT_PATH}/hf/t5-v2-${SIZE}"

mkdir -p "${HF_PATH}"
git clone https://huggingface.co/EleutherAI/t5-v2-${SIZE} "${HF_PATH}"
git -C "${HF_PATH}" remote set-url origin "https://lintang:${HF_KEY}@huggingface.co/EleutherAI/t5-v2-${SIZE}"

for STEP in $(eval echo "{$START..$END..10000}")
do
    T5X_PATH="${CHECKPOINT_PATH}/t5x/t5-v2-${SIZE}-step-${STEP}"
    mkdir -p "${T5X_PATH}"

    # Switch branch
    git -C "${HF_PATH}" checkout -b "step_${STEP}"
    git -C "${HF_PATH}" config http.postBuffer 524288000
    # Download from GCP
    gsutil -m cp -r "${GCP_BUCKET}/checkpoint_${STEP}/*" $T5X_PATH
    bash scripts/convert_v2.sh ${SIZE} $T5X_PATH $HF_PATH

    git -C "${HF_PATH}" add .
    git -C "${HF_PATH}" commit -am "add files for step ${STEP}"
    git -C "${HF_PATH}" push origin "step_${STEP}"

    # Delete Branch
    git -C "${HF_PATH}" checkout main
    git -C "${HF_PATH}" branch -D "step_${STEP}"

    rm -rf "${T5X_PATH}"
done
