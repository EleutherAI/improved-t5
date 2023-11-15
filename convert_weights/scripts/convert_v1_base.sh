mkdir -p $2

python convert_t5v1_checkpoint_to_pytorch.py \
    --config_file configs/base_v1/config.json \
    --t5x_checkpoint_path $1 \
    --pytorch_dump_path $2

cp configs/base_v1/* $2