# Improved T5

### Overview

- `configs/`: Configs for pretraining and finetuning
- `experiments/`: Scripts to launch pretraining and finetuning
- `convert_weights/`: Scripts to convert T5x checkpoints and upload them to HF
- `data/`: Seqio scripts for loading datasets
- `model/`: Gin config for models

### Running Experiments on TPUs

Experiments were run on TPUs.

```
bash run.sh <TPU Node Name> "source env-t5x/bin/activate; cd improved-t5/; bash experiments/script.sh"
```

If you need to rerun scripts on the node, you could make sure the node is empty by running
```
bash kill.sh <TPU Node Name>
```

### Convert Checkpoints to HF


```
bash scripts/convert_v1.sh MODEL_SIZE /PATH/TO/T5X_CHECPOINTS/ /PATH/TO/HF_CHECPOINTS/
```

```
bash scripts/convert_v2.sh MODEL_SIZE /PATH/TO/T5X_CHECPOINTS/ /PATH/TO/HF_CHECPOINTS/
```

Use absolute paths [issue](https://github.com/huggingface/transformers/issues/15464#issuecomment-1160318564)
