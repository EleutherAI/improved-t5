SIZE=$1
HF_MODEL_PATH=$2
HF_PATH=$3
T5X_PATH=$4

mkdir -p "${HF_PATH}"

git clone "https://huggingface.co/${HF_MODEL_PATH}" "${HF_PATH}"
git -C "${HF_PATH}" remote set-url origin "https://${HF_USERNAME}:${HF_KEY}@huggingface.co/${HF_MODEL_PATH}"

# in main branch
git -C "${HF_PATH}" checkout main
git -C "${HF_PATH}" config http.postBuffer 524288000

bash scripts/convert_v2.sh ${SIZE} $T5X_PATH $HF_PATH

git -C "${HF_PATH}" add .
git -C "${HF_PATH}" commit -am "add files"
git -C "${HF_PATH}" push origin main

