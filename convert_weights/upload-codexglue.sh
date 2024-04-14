SIZE=$1
LANG=$2
HF_MODEL_PATH=$3
HF_PATH=$4
T5X_PATH=$5

mkdir -p "${HF_PATH}"
git lfs install
git clone "https://huggingface.co/${HF_MODEL_PATH}" "${HF_PATH}"
git -C "${HF_PATH}" remote set-url origin "https://${HF_USERNAME}:${HF_KEY}@huggingface.co/${HF_MODEL_PATH}"
huggingface-cli lfs-enable-largefiles "${HF_PATH}"

# Switch branch
git -C "${HF_PATH}" checkout -b "$LANG"
git -C "${HF_PATH}" config http.postBuffer 524288000

bash scripts/convert_v2.sh ${SIZE} $T5X_PATH $HF_PATH

git -C "${HF_PATH}" add .
git -C "${HF_PATH}" commit -am "add files for finetuning on $LANG"
git -C "${HF_PATH}" push origin "$LANG"
git -C "${HF_PATH}" checkout main
