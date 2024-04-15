# Improved T5

## Overview

- `configs/`: Configs for pretraining and finetuning
- `experiments/`: Scripts to launch pretraining and finetuning
- `convert_weights/`: Scripts to convert T5x checkpoints and upload them to HF
- `data/`: Seqio scripts for loading datasets
- `model/`: Gin config for models
- `tpu-scripts/`: Helper scripts for running training jobs on TPUs

## Running Experiments on TPUs

### Setup

Experiments were run on TPUs. The main scripts involve `send.sh`, `run.sh`, `setup.sh` and `kill.sh` in `tpu-scripts/`. To setup a TPU with the required libraries and dependencies, 
```
bash send.sh <TPU Node Name> setup.sh
```
Then run 
```
bash run.sh <TPU Node Name> "bash setup.sh"
```

To run an pretraining/finetuning job,
```
bash run.sh <TPU Node Name> "source env-t5x/bin/activate; cd improved-t5/; bash <script>"
```

If you need to rerun scripts on the node, you could make sure the node is empty by running
```
bash kill.sh <TPU Node Name>
```

If you are using a different zone than `us-central2-b`, you will need to changed the `--zone` argument in all the scripts.

## Convert Checkpoints to HF

In `convert_weights/` you can use scripts to convert T5x checkpoints.
```
bash scripts/convert_v1.sh MODEL_SIZE /PATH/TO/T5X_CHECPOINTS/ /PATH/TO/HF_CHECPOINTS/
```

```
bash scripts/convert_v2.sh MODEL_SIZE /PATH/TO/T5X_CHECPOINTS/ /PATH/TO/HF_CHECPOINTS/
```

Use absolute paths [issue](https://github.com/huggingface/transformers/issues/15464#issuecomment-1160318564)

You can also directly convert T5x checkpoints and upload them to your HF hub. For that see `convert_weights/upload.sh` and `convert_weights/upload-multiple.sh`.
