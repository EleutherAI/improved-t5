SIZE=$1
START=$2
END=$3
HF_MODEL_PATH=$4
HF_PATH=$5
T5X_PATH=$6

mkdir -p "${HF_PATH}"
git lfs install
git clone "https://huggingface.co/${HF_MODEL_PATH}" "${HF_PATH}"
git -C "${HF_PATH}" remote set-url origin "https://${HF_USERNAME}:${HF_KEY}@huggingface.co/${HF_MODEL_PATH}"
huggingface-cli lfs-enable-largefiles "${HF_PATH}"

for STEP in $(eval echo "{$START..$END..10000}")
do
    # Switch branch
    git -C "${HF_PATH}" checkout -b "step_${STEP}"
    git -C "${HF_PATH}" pull "origin/step_${STEP}"
    git -C "${HF_PATH}" config http.postBuffer 524288000

    # Download from GCP
    gsutil -m cp -r "${T5X_PATH}/checkpoint_${STEP}/*" $HF_PATH

    git -C "${HF_PATH}" add .
    git -C "${HF_PATH}" commit -am "add files for step ${STEP}"
    git -C "${HF_PATH}" pull origin "step_${STEP}"
    git -C "${HF_PATH}" push origin "step_${STEP}"

done
